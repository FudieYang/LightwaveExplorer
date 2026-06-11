"""
Memetic ML-Driven Optical Topology Discovery for LightwaveExplorer
(Genetic Algorithm for Topology + Bayesian Optimization for Parameters)

v6: General optimizer with:
    - Pairwise Ranking Transformer surrogate (replaces RandomForest)
    - Sequence canonicalization (merge adjacent same-type operations)
    - Adaptive mutation probabilities (learned from success history)
    - Topology hash deduplication cache
"""

import numpy as np
from collections import Counter
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz  # NumPy < 2.0 compatibility
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
import random
import copy
import sys
import uuid
import json
from datetime import datetime
import optuna

# Prevent Optuna from spamming logs for every trial
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print("[WARN] cma not found. Install with: pip install cma")

try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not found.")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] PyTorch {torch.__version__} on {TORCH_DEVICE}")
except ImportError:
    HAS_TORCH = False
    TORCH_DEVICE = None
    print("[WARN] PyTorch not found. Ranking Surrogate disabled.")
    print("       Install with: pip install torch")

# =============================================================================
# 0. PATH CONFIG (修改这里就能适配任何机器)
# =============================================================================
PROJECT_ROOT  = "/home/fudie/LightwaveExplorer"
CLI_PATH      = os.path.join(PROJECT_ROOT, "build_cli", "LightwaveExplorer")
DB_FILE       = os.path.join(PROJECT_ROOT, "CrystalDatabase.txt")
OPTUNA_DB     = os.path.join(PROJECT_ROOT, "Source", "Python", "optuna_lwe.journal")
EVAL_LOG      = os.path.join(PROJECT_ROOT, "Source", "Python", "cma_eval_log.jsonl")
TMP_DIR       = "/mnt/d/lwe_tmp"
OUTPUT_FILE   = "/mnt/c/user/12691/Desktop/lwe_memetic_best.txt"

# Monitor files (centralized)
MONITOR_STATUS  = os.path.join(TMP_DIR, "lwe_monitor_status.txt")
MONITOR_COMP    = os.path.join(TMP_DIR, "lwe_monitor_completed.txt")
MONITOR_STOP    = os.path.join(TMP_DIR, "lwe_monitor_early_stops.txt")

# General optimizer constraints
POP_SIZE              = 16   # Population size
N_GENERATIONS         = 30   # Total GA generations
MAX_WORKERS           = 2    # Parallel CPU workers for LWE
SURROGATE_MAX_SEQ_LEN = 20   # Max sequence length for Transformer (pad/truncate)
SURROGATE_MIN_DATA    = 100  # Minimum evaluations before surrogate activates
SURROGATE_OVERGENERATE = 6   # Generate N × this candidates, then screen to N
SURROGATE_TRAIN_EPOCHS = 40  # Training epochs per refit
SURROGATE_LR           = 3e-3  # Learning rate for Transformer

# CMA-ES parameter optimization
CMA_MAX_EVALS = 30    # Max LWE simulations per topology per call
CMA_PATIENCE  = 10    # Early stop if no improvement in this many evals

# Phase-matching angles for BIBO THG (locked, not optimized)
PHASE_MATCH_ANGLES = [
    {"theta": 0.0,   "phi": 0.0},
    {"theta": 40.2,  "phi": 0.0},
    {"theta": 139.8, "phi": 0.0},
    {"theta": 11.0, "phi": 0.0},
    {"theta": 169.0, "phi": 0.0},
    {"theta": 35.2, "phi": 0.0},
    {"theta": 164.8, "phi": 0.0},
]

sys.path.insert(0, os.path.join(PROJECT_ROOT, "Source", "Python", "src"))
import LightwaveExplorer as lwe

# Auto-setup: ensure crystal database is always findable by the CLI
_db_home = os.path.expanduser("~/.LightwaveExplorer")
os.makedirs(_db_home, exist_ok=True)
_db_link = os.path.join(_db_home, "CrystalDatabase.txt")
if not os.path.exists(_db_link):
    os.symlink(DB_FILE, _db_link)
    print(f"[Setup] Linked crystal database: {DB_FILE} → {_db_link}")

# =============================================================================
# 1. Semi-Atomic Instruction Registry
# =============================================================================

ATOMIC_LIBRARY = {
    "BiaxialCrystalBlock": {
        "params": {
            "theta": {"type": "continuous", "min": 0.0, "max": 180.0},
            "phi": {"type": "continuous", "min": 0.0, "max": 180.0},
            "length_um": {"type": "continuous", "min": 10.0, "max": 800.0}
        },
        "template": "rotateIntoBiaxial(d,{theta:.4f},{phi:.4f},d)nonlinear(d,{theta:.4f},{phi:.4f},{length_um:.2f},d)rotateFromBiaxial(d,{theta:.4f},{phi:.4f},d)"
    },

    "NormalCrystalBlock": {
        "params": {
            "length_um": {"type": "continuous", "min": 10.0, "max": 2000.0}
        },
        "template": "nonlinear(6,0.0000,0.0000,{length_um:.2f},d)"
    },

    "RotateFrame": {
        "params": {
            "angle": {"type": "continuous", "min": -180.0, "max": 180.0}
        },
        "template": "rotate({angle:.4f})"
    },

    "Polarizer": {
        "params": {
            "angle": {"type": "continuous", "min": 0.0, "max": 180.0}
        },
        "template": "polarizer({angle:.4f})"
    },

    "LinearPropagation": {
        "params": {
            "length_um": {"type": "continuous", "min": 10.0, "max": 5000.0}
        },
        "template": "linear(0,0,0,{length_um:.2f},d)"
    },

    "SpectralFilter": {
        "params": {
            "f_center_THz": {"type": "continuous", "min": 50.0, "max": 1500.0},
            "f_width_THz": {"type": "continuous", "min": 5.0, "max": 500.0},
            "inBandAmplitude": {"type": "continuous", "min": -1.0, "max": 1.0},
            "outOfBandAmplitude": {"type": "continuous", "min": 0.0, "max": 1.0}
        },
        "template": "filter({f_center_THz:.2f},{f_width_THz:.2f},4,{inBandAmplitude:.4f},{outOfBandAmplitude:.4f})"
    },

    "SphericalMirror": {
        "params": {
            "focal_m": {"type": "continuous", "min": -1.0, "max": 1.0}
        },
        "template": "sphericalMirror({focal_m:.4f})"
    },

    "Aperture": {
        "params": {
            "d_m": {"type": "continuous", "min": 0.0, "max": 0.0002},
            "activation_parameter": {"type": "continuous", "min": 1.0, "max": 100.0}
        },
        "template": "aperture({d_m:.4f},{activation_parameter:.2f})"
    }
}

# =============================================================================
# 2. Gene and Chromosome Objects (Topological Only)
# =============================================================================

class AtomicGene:
    """A purely structural placeholder for a component."""
    def __init__(self, comp_type):
        if comp_type not in ATOMIC_LIBRARY:
            raise ValueError(f"Unknown instruction '{comp_type}'")
        self.comp_type = comp_type
        self.config = ATOMIC_LIBRARY[comp_type]

class TopologyChromosome:
    def __init__(self, genes=None):
        self.genes = genes if genes else []
        self.loss = float("inf")
        self.efficiency = 0.0
        self.best_sequence = ""
        self._last_mutation = None       # For adaptive mutation tracking
        self._parent_efficiency = 0.0    # Parent's efficiency at time of creation
        
    def get_hash(self):
        """Returns a string hash identifying the topological layout."""
        if not self.genes:
            return "empty"
        return "___".join(g.comp_type for g in self.genes)

    def normalize(self):
        """Canonicalize the gene sequence by merging adjacent same-type operations.
        
        Eliminates equivalent sequences without any physics-specific assumptions:
          linear(d1) + linear(d2) ≡ linear(d1+d2)   (one Optuna param covers both)
          rotate(θ1) + rotate(θ2) ≡ rotate(θ1+θ2)   (ditto)
        
        Crystal blocks are NOT merged because adjacent crystals at different
        angles serve distinct physical roles (e.g. walk-off compensation).
        """
        if len(self.genes) <= 1:
            return self
        
        # Types that are physically equivalent when adjacent
        MERGEABLE = {"LinearPropagation", "RotateFrame"}
        
        merged = [self.genes[0]]
        for g in self.genes[1:]:
            if g.comp_type in MERGEABLE and merged[-1].comp_type == g.comp_type:
                continue  # Skip; single gene with Optuna-tuned param is equivalent
            merged.append(g)

        USELESS_AT_TAIL = {"RotateFrame", "Aperture", "LinearPropagation", "Polarizer"}
        while merged and merged[-1].comp_type in USELESS_AT_TAIL:
            merged.pop()

        self.genes = merged
        return self

    def mutate(self, mutation_probs=None):
        """Purely Topological Mutation with adaptive probabilities.
        
        Args:
            mutation_probs: list [p_insert, p_delete, p_swap, p_replace].
                            If None, uses default [0.40, 0.15, 0.15, 0.30].
        """
        child = TopologyChromosome([copy.deepcopy(g) for g in self.genes])
        
        if mutation_probs is None:
            mutation_probs = [0.40, 0.15, 0.15, 0.30]
        
        actions = ["insert_instruction", "delete_instruction", "swap_order", "replace_gene"]
        action = np.random.choice(actions, p=mutation_probs)
        
        if not child.genes:
            action = "insert_instruction"
        
        if action == "insert_instruction":
            comp_type = random.choice(list(ATOMIC_LIBRARY.keys()))
            new_gene = AtomicGene(comp_type)
            idx = random.randint(0, len(child.genes))
            child.genes.insert(idx, new_gene)
            
        elif action == "delete_instruction" and len(child.genes) > 1:
            idx = random.randint(0, len(child.genes) - 1)
            del child.genes[idx]

        elif action == "swap_order" and len(child.genes) > 1:
            idx = random.randint(0, len(child.genes) - 2)
            child.genes[idx], child.genes[idx+1] = child.genes[idx+1], child.genes[idx]

        elif action == "replace_gene" and child.genes:
            # Replace one gene with a DIFFERENT type (guarantees new topology)
            idx = random.randint(0, len(child.genes) - 1)
            old_type = child.genes[idx].comp_type
            other_types = [t for t in ATOMIC_LIBRARY.keys() if t != old_type]
            new_type = random.choice(other_types)
            child.genes[idx] = AtomicGene(new_type)

        child.normalize()
        child._last_mutation = action
        return child

# =============================================================================
# 2.5 Surrogate Model for Topology Pre-Screening
# =============================================================================

# =============================================================================
# Token vocabulary for topology encoding
# =============================================================================
TOKEN_PAD = 0
TOKEN_CLS = 1
COMP_TO_TOKEN = {
    "BiaxialCrystalBlock": 2,
    "NormalCrystalBlock":  3,
    "RotateFrame":         4,
    "Polarizer":           5,
    "LinearPropagation":   6,
    "SpectralFilter":      7,
    "SphericalMirror":     8,
    "Aperture":            9,
}
VOCAB_SIZE = 10  # 0=PAD, 1=CLS, 2-9=components


def tokenize_topology(topo, max_len=SURROGATE_MAX_SEQ_LEN):
    """Convert a TopologyChromosome into a list of token IDs.
    
    Format: [CLS, comp1, comp2, ..., PAD, PAD, ...]
    Truncated or padded to max_len.
    """
    tokens = [TOKEN_CLS]
    for g in topo.genes:
        tok = COMP_TO_TOKEN.get(g.comp_type, TOKEN_PAD)
        tokens.append(tok)
    # Truncate
    tokens = tokens[:max_len]
    # Pad
    while len(tokens) < max_len:
        tokens.append(TOKEN_PAD)
    return tokens


if HAS_TORCH:
    class TopologyTransformer(nn.Module):
        """Lightweight Transformer encoder that maps a topology token sequence
        to a scalar 'potential score'.
        
        Architecture:
            Token Embedding + Sinusoidal PE → TransformerEncoder → CLS pooling → score
        
        Parameters: ~6K (intentionally tiny to avoid overfitting on ~100 samples).
        """
        
        def __init__(self, vocab_size=VOCAB_SIZE, d_model=32, nhead=4,
                     num_layers=2, dim_ff=64, max_len=SURROGATE_MAX_SEQ_LEN):
            super().__init__()
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=TOKEN_PAD)
            
            # Sinusoidal positional encoding (frozen, not learned)
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                            * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Score head: CLS vector → scalar
            self.score_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1)
            )
        
        def forward(self, token_ids):
            """Forward pass.
            
            Args:
                token_ids: (batch, seq_len) LongTensor of token IDs
            Returns:
                scores: (batch,) float tensor of scalar scores
            """
            # Create padding mask: True where token is PAD
            pad_mask = (token_ids == TOKEN_PAD)  # (batch, seq_len)
            
            x = self.embedding(token_ids) + self.pe[:, :token_ids.size(1), :]
            x = self.transformer(x, src_key_padding_mask=pad_mask)
            
            # CLS token is at position 0
            cls_vec = x[:, 0, :]  # (batch, d_model)
            scores = self.score_head(cls_vec).squeeze(-1)  # (batch,)
            return scores


class SurrogateModel:
    """Improvement-Based Pairwise Ranking Transformer Surrogate.
    
    Instead of ranking topologies by absolute efficiency (which collapses to
    reproducing the baseline), this model ranks by IMPROVEMENT over parent.
    
    Training signal: Δη = η(child) - η(parent)
    This creates a 'fake gradient' in topology space: the model learns which
    MUTATIONS tend to produce improvement, not which topologies are already good.
    
    Screening uses UCB acquisition: score + β·(uncertainty + novelty)
    to balance exploitation vs exploration.
    """
    
    def __init__(self):
        # History storage
        self.topo_history = []       # list of TopologyChromosome
        self.y_history = []          # list of float (absolute efficiency)
        self.improvement_history = [] # list of float (η_child - η_parent)
        self.is_fitted = False
        self._model = None
        self._train_pairwise_acc = 0.0  # Track training quality
        
        # Source tracking
        self.optuna_count = 0
        self.jsonl_count = 0
        
        # Preload history from database if available
        self._preload_history()
    
    def _preload_history(self, db_path=OPTUNA_DB):
        """Load past evaluated topologies from Optuna Journal so surrogate starts smart."""
        print(f"  [Surrogate] Attempting to preload history from {db_path}...")
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.ERROR)
            
            if os.path.exists(db_path):
                storage = optuna.storages.JournalStorage(
                    optuna.storages.journal.JournalFileBackend(db_path)
                )
            else:
                print("  [Surrogate] No existing Optuna Journal found. Starting fresh.")
                return
                
            summaries = optuna.get_all_study_summaries(storage=storage)
            loaded_count = 0
            
            for summary in summaries:
                # The study name is the topology hash: "Comp1___Comp2___..."
                if "___" not in summary.study_name and summary.study_name != "empty":
                    continue
                    
                # Find best efficiency in this study
                study = optuna.load_study(study_name=summary.study_name, storage=storage)
                best_eff = 0.0
                try:
                    best_trial = study.best_trial
                    best_eff = best_trial.user_attrs.get("efficiency", 0.0)
                except ValueError:
                    continue  # No completed trials
                    
                # Reconstruct dummy TopologyChromosome just for the tokenizer
                comps = summary.study_name.split("___")
                dummy_topo = TopologyChromosome()
                for comp in comps:
                    if comp in ATOMIC_LIBRARY:
                        dummy_topo.genes.append(AtomicGene(comp))
                
                if dummy_topo.genes:
                    self.topo_history.append(dummy_topo)
                    self.y_history.append(best_eff)
                    # Preloaded data: no parent info, use absolute eff as improvement
                    # (conservative: treats all historical topologies as starting from 0)
                    self.improvement_history.append(best_eff)
                    loaded_count += 1
            
            self.optuna_count = loaded_count
            print(f"  [Surrogate] Preloaded {loaded_count} past evaluations from Optuna DB.")
            
            # Also preload from new JSONL log
            jsonl_count = 0
            if os.path.exists(EVAL_LOG):
                import json
                with open(EVAL_LOG, "r") as f:
                    for line in f:
                        entry = json.loads(line)
                        topo_hash = entry.get("topo", "")
                        eff = entry.get("efficiency", 0.0)
                        if not topo_hash: continue
                            
                        comps = topo_hash.split("___")
                        dummy_topo = TopologyChromosome()
                        for comp in comps:
                            if comp in ATOMIC_LIBRARY:
                                dummy_topo.genes.append(AtomicGene(comp))
                                
                        if dummy_topo.genes:
                            self.topo_history.append(dummy_topo)
                            self.y_history.append(eff)
                            self.improvement_history.append(eff)
                            jsonl_count += 1
                self.jsonl_count = jsonl_count
                print(f"  [Surrogate] Preloaded {jsonl_count} evaluations from JSONL log.")
            
            if HAS_TORCH and len(self.topo_history) >= SURROGATE_MIN_DATA:
                self.train()
                
        except Exception as e:
            print(f"  [Surrogate] Preload failed (ignored): {e}")
        finally:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def _build_model(self):
        """Create a fresh TopologyTransformer."""
        if not HAS_TORCH:
            return None
        model = TopologyTransformer().to(TORCH_DEVICE)
        return model
    
    def _build_pairs(self):
        """Construct pairwise training data based on IMPROVEMENT over parent.
        
        Instead of ranking by absolute efficiency (which collapses to baseline),
        we rank by improvement: Δη = η(child) - η(parent).
        
        A novel topology achieving 0.5% from a 0% parent (Δη = +0.5%) beats
        a baseline child achieving 5.1% from a 5% parent (Δη = +0.1%).
        This teaches the model to value EXPLORATION over EXPLOITATION.
        """
        n = len(self.improvement_history)
        tokens_all = [tokenize_topology(t) for t in self.topo_history]
        improvements = np.array(self.improvement_history)
        
        pairs_win = []   # token sequences of the "winner" (higher improvement)
        pairs_lose = []  # token sequences of the "loser" (lower improvement)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Skip pairs where both have negligible improvement
                if abs(improvements[i]) < 1e-10 and abs(improvements[j]) < 1e-10:
                    continue
                diff_ij = improvements[i] - improvements[j]
                if abs(diff_ij) < 1e-12:
                    continue  # Skip ties
                if diff_ij > 0:
                    pairs_win.append(tokens_all[i])
                    pairs_lose.append(tokens_all[j])
                else:
                    pairs_win.append(tokens_all[j])
                    pairs_lose.append(tokens_all[i])
        
        if not pairs_win:
            return None, None
        
        # Sub-sample if too many pairs (cap at 50000 for speed)
        MAX_PAIRS = 50000
        if len(pairs_win) > MAX_PAIRS:
            idx = np.random.choice(len(pairs_win), MAX_PAIRS, replace=False)
            pairs_win = [pairs_win[i] for i in idx]
            pairs_lose = [pairs_lose[i] for i in idx]
        
        win_tensor = torch.LongTensor(pairs_win).to(TORCH_DEVICE)
        lose_tensor = torch.LongTensor(pairs_lose).to(TORCH_DEVICE)
        return win_tensor, lose_tensor
    
    def train(self):
        """Train the Ranking Transformer on pairwise improvement comparisons."""
        if not HAS_TORCH:
            return
        
        win_data, lose_data = self._build_pairs()
        if win_data is None or len(win_data) < 10:
            print(f"  [Surrogate] Not enough informative pairs ({0 if win_data is None else len(win_data)}), skipping.")
            return
        
        total_samples = len(self.topo_history)
        print(f"  [Surrogate] Training on {total_samples} samples "
              f"(Optuna: {self.optuna_count}, JSONL: {self.jsonl_count}, Session: {total_samples - self.optuna_count - self.jsonl_count})")
        print(f"  [Surrogate] Generated {len(win_data)} improvement-based pairs.")
        
        # Fresh model each time (avoids stale representations)
        self._model = self._build_model()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=SURROGATE_LR)
        
        n_pairs = len(win_data)
        batch_size = min(256, n_pairs)
        
        self._model.train()
        for epoch in range(SURROGATE_TRAIN_EPOCHS):
            # Shuffle
            perm = torch.randperm(n_pairs, device=TORCH_DEVICE)
            total_loss = 0.0
            n_correct = 0
            n_batches = 0
            
            for start in range(0, n_pairs, batch_size):
                end = min(start + batch_size, n_pairs)
                idx = perm[start:end]
                
                s_win = self._model(win_data[idx])
                s_lose = self._model(lose_data[idx])
                
                # Bradley-Terry loss with margin clamping to prevent saturation
                diff = s_win - s_lose
                diff = torch.clamp(diff, min=-5.0, max=5.0)  # Prevent extreme confidence
                loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_correct += (diff > 0).sum().item()
                n_batches += 1
            
            acc = n_correct / n_pairs
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:3d}/{SURROGATE_TRAIN_EPOCHS}: "
                      f"loss={total_loss/n_batches:.4f}, pairwise_acc={acc:.1%}")
        
        self._train_pairwise_acc = acc
        self._model.eval()
        self.is_fitted = True
        print(f"  [Surrogate] Training complete. Final pairwise accuracy: {acc:.1%}")
    
    def update(self, topo, efficiency, parent_efficiency=0.0):
        """Add an evaluated topology with improvement tracking.
        
        Args:
            topo: TopologyChromosome that was evaluated
            efficiency: absolute THG efficiency achieved
            parent_efficiency: efficiency of the parent topology this was mutated from
        """
        self.topo_history.append(copy.deepcopy(topo))
        self.y_history.append(efficiency)
        self.improvement_history.append(efficiency - parent_efficiency)
    
    def novelty_score(self, topo):
        """Compute novelty as minimum distance to all previously seen topologies.
        
        Uses multiset Jaccard distance on component types + length difference.
        Returns 0.0 for identical topology, 1.0 for maximally different.
        
        This acts as a discrete 'gradient direction indicator': topologies
        far from anything seen represent unexplored directions in topology space.
        """
        if not self.topo_history:
            return 1.0
        
        candidate_bag = Counter(g.comp_type for g in topo.genes)
        candidate_len = len(topo.genes)
        
        min_dist = 1.0
        for seen in self.topo_history:
            seen_bag = Counter(g.comp_type for g in seen.genes)
            # Multiset Jaccard distance
            intersection = sum((candidate_bag & seen_bag).values())
            union = sum((candidate_bag | seen_bag).values())
            jaccard_dist = 1.0 - (intersection / union) if union > 0 else 1.0
            
            # Length difference contribution
            seen_len = len(seen.genes)
            len_dist = abs(candidate_len - seen_len) / max(candidate_len, seen_len, 1)
            
            # Combined distance (weighted)
            dist = 0.7 * jaccard_dist + 0.3 * len_dist
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def predict_with_uncertainty(self, topo, n_samples=8):
        """MC Dropout uncertainty estimation.
        
        Run N forward passes with dropout enabled to estimate epistemic
        uncertainty. High std = model is unsure = worth exploring.
        
        Returns:
            (mean_score, std_score)
        """
        if not self.is_fitted or self._model is None:
            return 0.0, 1.0  # High uncertainty when no model
        
        tokens = torch.LongTensor([tokenize_topology(topo)]).to(TORCH_DEVICE)
        
        # Enable dropout for uncertainty estimation
        self._model.train()
        scores = []
        with torch.no_grad():
            for _ in range(n_samples):
                scores.append(self._model(tokens).item())
        self._model.eval()
        
        return float(np.mean(scores)), float(np.std(scores))
    
    def predict(self, topo):
        """Return a scalar 'improvement potential score' for a topology.
        
        Higher score = model thinks this topology shows more improvement potential.
        NOTE: This is trained on improvement-over-parent, not absolute efficiency.
        """
        if not self.is_fitted or self._model is None:
            return 0.0
        
        tokens = tokenize_topology(topo)
        token_tensor = torch.LongTensor([tokens]).to(TORCH_DEVICE)
        
        with torch.no_grad():
            score = self._model(token_tensor).item()
        return score
    
    def screen(self, candidates, top_k, generation=0, max_generations=30):
        """UCB-based screening with diversity guarantee.
        
        acquisition(c) = score(c) + β · (uncertainty(c) + 0.5 · novelty(c))
        
        β decays linearly from 2.0 (explore) to 0.2 (exploit) over generations.
        At least 30% of selected candidates must be 'novel' (novelty > 0.3).
        """
        if not self.is_fitted or len(candidates) <= top_k:
            if len(candidates) > top_k:
                return random.sample(candidates, top_k)
            return candidates
        
        # Exploration coefficient: high early, decays over generations
        beta = max(0.2, 2.0 * (1.0 - generation / max(max_generations, 1)))
        
        # Score all candidates with UCB acquisition
        scored_candidates = []  # (acquisition, score_mean, novelty, candidate)
        for c in candidates:
            score_mean, score_std = self.predict_with_uncertainty(c, n_samples=8)
            novelty = self.novelty_score(c)
            
            # UCB acquisition: exploitation + exploration
            acquisition = score_mean + beta * (score_std + 0.5 * novelty)
            scored_candidates.append((acquisition, score_mean, novelty, c))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Diversity quota: at least 30% of selected must be genuinely novel
        n_novel_quota = max(1, top_k // 3)
        NOVELTY_THRESHOLD = 0.3
        
        selected = []
        novel_count = 0
        non_novel_buffer = []  # Track non-novel selections for potential replacement
        
        for acq, score, nov, c in scored_candidates:
            if len(selected) >= top_k:
                break
            selected.append(c)
            if nov > NOVELTY_THRESHOLD:
                novel_count += 1
            else:
                non_novel_buffer.append((len(selected) - 1, acq))  # (index, acquisition)
        
        # If diversity quota not met, swap worst non-novel with best unseen novel
        if novel_count < n_novel_quota and non_novel_buffer:
            # Find novel candidates not yet selected
            selected_set = set(id(c) for c in selected)
            novel_remaining = [
                (acq, c) for acq, score, nov, c in scored_candidates
                if nov > NOVELTY_THRESHOLD and id(c) not in selected_set
            ]
            
            # Replace from the bottom of non-novel
            non_novel_buffer.sort(key=lambda x: x[1])  # Lowest acquisition first
            n_swap = min(n_novel_quota - novel_count, len(novel_remaining), len(non_novel_buffer))
            for i in range(n_swap):
                idx_to_replace = non_novel_buffer[i][0]
                _, replacement = novel_remaining[i]
                selected[idx_to_replace] = replacement
                novel_count += 1
        
        acq_values = [a for a, _, _, _ in scored_candidates]
        final_novel = sum(1 for c in selected if self.novelty_score(c) > NOVELTY_THRESHOLD)
        print(f"  Surrogate UCB screening: {len(candidates)} → {top_k} "
              f"(β={beta:.2f}, novel={final_novel}/{top_k}, "
              f"acq range: {min(acq_values):.3f} – {max(acq_values):.3f}, "
              f"train acc: {self._train_pairwise_acc:.0%})")
        
        return selected

# =============================================================================
# 2.6 Random Topology Generator (for diversity injection)
# =============================================================================

def random_topology(min_len=2, max_len=6):
    """Generate a completely random topology (not derived from any parent).
    
    Ensures at least one crystal block is present (otherwise no THG possible).
    These are injected into the candidate pool to force the GA to explore
    regions of topology space unreachable by mutation alone.
    """
    length = random.randint(min_len, max_len)
    genes = []
    for _ in range(length):
        comp_type = random.choice(list(ATOMIC_LIBRARY.keys()))
        genes.append(AtomicGene(comp_type))
    
    # Guarantee at least one crystal (otherwise THG is impossible)
    has_crystal = any(g.comp_type in ("BiaxialCrystalBlock", "NormalCrystalBlock") for g in genes)
    if not has_crystal:
        idx = random.randint(0, len(genes) - 1)
        crystal_type = random.choice(["BiaxialCrystalBlock", "NormalCrystalBlock"])
        genes[idx] = AtomicGene(crystal_type)
    
    topo = TopologyChromosome(genes)
    topo.normalize()
    topo._parent_efficiency = 0.0  # No parent
    return topo

# =============================================================================
# 3. LWE & CMA-ES Parameter Optimization
# =============================================================================

def _log_eval(topo_hash, params, efficiency, sequence, n_evals):
    """Append one evaluation result to the JSONL log."""
    entry = {
        "ts": datetime.now().isoformat(),
        "topo": topo_hash,
        "eff": float(efficiency),
        "seq": sequence,
        "n": n_evals,
        "params": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                   for k, v in params.items()},
    }
    try:
        with open(EVAL_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _run_single_lwe(topo, param_dict, cli_path):
    """Run one LWE simulation with explicit parameters. Returns (efficiency, sequence_str)."""
    # Clip filter amplitudes to enforce passive filter constraint (no unphysical gain)
    for i, gene in enumerate(topo.genes):
        if gene.comp_type == "SpectralFilter":
            in_key = f"g{i}_inBandAmplitude"
            out_key = f"g{i}_outOfBandAmplitude"
            if in_key in param_dict and out_key in param_dict:
                in_val = param_dict[in_key]
                out_val = param_dict[out_key]
                total = in_val + out_val
                if abs(total) > 1.0:
                    sign = 1.0 if total >= 0 else -1.0
                    param_dict[in_key] = sign * 1.0 - out_val

    # Build sequence string from param_dict
    parts = []
    for i, gene in enumerate(topo.genes):
        kwargs = {}
        for p_name in gene.config["params"]:
            key = f"g{i}_{p_name}"
            kwargs[p_name] = param_dict[key]
        parts.append(gene.config["template"].format(**kwargs))
    sequence_str = "init()" + "".join(parts)

    run_uuid = str(uuid.uuid4())[:8]
    run_dir = os.path.join(TMP_DIR, f"{topo.get_hash()[:40]}_{run_uuid}")
    os.makedirs(run_dir, exist_ok=True)
    runner = lwe.SimulationRunner(cli_path=cli_path, work_dir=run_dir)

    try:
        runner.set_params(
            sequence=sequence_str,
            pulse_energy1=1.9e-08, frequency1=1.3e14, bandwidth1=1.5e13,
            sg_order1=2, beamwaist1=1.12e-5, delay1=-1.3e-13,
            polarization1=1.5707963267949,
            pulse_energy2=0, frequency2=4.9e14, bandwidth2=2e13,
            sg_order2=4, gdd2=1e-28, beamwaist2=9e-5,
            polarization2=1.5707963267949,
            material_index=4, crystal_theta=0, crystal_phi=0,
            crystal_thickness=0.0004, dz=5e-7,
            grid_width=2.08e-4, grid_height=2.08e-4, dx=8e-6,
            time_span=6e-13, dt=5e-16,
            band_gap=6, effective_mass=1, drude_gamma=5e12,
            propagation_mode=2,
        )
        runner.run(verbose=False, timeout=800)

        # Record completion for monitor
        try:
            with open(MONITOR_COMP, "a") as f:
                f.write(".\n")
        except Exception:
            pass

        freq = runner.result.frequencyVectorSpectrum
        spectrum = runner.result.spectrumTotal
        if spectrum.ndim > 1:
            spectrum = spectrum[-1]

        thg_center = 390e12
        mask_thg = np.abs(freq - thg_center) < 30e12
        thg_pwr = np.trapezoid(spectrum[mask_thg], freq[mask_thg])

        INPUT_ENERGY = 1.9e-08
        eff = thg_pwr / INPUT_ENERGY if INPUT_ENERGY > 0 else 0

        if np.isnan(eff) or np.isinf(eff) or eff > 1.0 or eff < 0:
            return 0.0, sequence_str
        return eff, sequence_str

    except Exception:
        return 0.0, sequence_str
    finally:
        runner.cleanup()
        shutil.rmtree(run_dir, ignore_errors=True)


def _build_default_params(topo):
    """Build a default parameter dict for a topology (used as x0 for CMA-ES).
    
    Crystal theta/phi are locked to phase-matching seeds (deterministic by position).
    All other continuous params get midpoint defaults.
    Categorical params get safe defaults.
    """
    params = {}
    crystal_idx = 0

    for i, gene in enumerate(topo.genes):
        is_crystal = gene.comp_type in ("BiaxialCrystalBlock", "NormalCrystalBlock")

        for p_name, p_def in gene.config["params"].items():
            key = f"g{i}_{p_name}"

            if p_def["type"] == "continuous":
                if p_name in ("theta", "phi") and is_crystal:
                    ref = random.choice(PHASE_MATCH_ANGLES)
                    params[key] = ref.get(p_name, 0.0)
                elif p_name == "angle" and gene.comp_type == "RotateFrame":
                    params[key] = 180.0
                elif p_name == "angle" and gene.comp_type == "Polarizer":
                    params[key] = 90.0
                elif p_name == "focal_m":
                    params[key] = 0.0
                elif p_name == "length_um":
                    params[key] = 250.0
                else:
                    params[key] = (p_def["min"] + p_def["max"]) / 2.0

        if is_crystal:
            crystal_idx += 1

    return params


def evaluate_topology_cmaes(topo: TopologyChromosome, cli_path: str,
                            max_evals: int = CMA_MAX_EVALS,
                            patience: int = CMA_PATIENCE,
                            warm_start: dict = None):
    """CMA-ES parameter optimization for a given topology.
    
    Crystal theta/phi are LOCKED to phase-matching values (not optimized).
    CMA-ES optimizes: lengths, rotation angles, and other continuous params.
    Covariance matrix adaptation automatically discovers parameter couplings.
    Early stops when no improvement is found for `patience` evaluations.
    """
    topo_hash = topo.get_hash()

    # 1. Build parameter dict: fixed (angles, categoricals) + optimizable (lengths, etc.)
    defaults = warm_start if warm_start else _build_default_params(topo)

    fixed_params = {}   # Not touched by CMA-ES
    opt_specs = []      # (key, lower, upper, x0) for CMA-ES

    for i, gene in enumerate(topo.genes):
        is_crystal = gene.comp_type in ("BiaxialCrystalBlock", "NormalCrystalBlock")
        for p_name, p_def in gene.config["params"].items():
            key = f"g{i}_{p_name}"
            lo, hi = p_def["min"], p_def["max"]
            x0_val = max(lo, min(hi, defaults.get(key, (lo + hi) / 2.0)))
                
            # Dynamic boundaries: restrict crystal angles to +/- 10 degrees around the seed
            if is_crystal and p_name in ("theta", "phi"):
                lo = max(0.0, x0_val - 15.0)
                hi = min(180.0, x0_val + 15.0)
                
            opt_specs.append((key, lo, hi, x0_val))

    # 2. If nothing to optimize, just run once
    if not opt_specs:
        eff, seq = _run_single_lwe(topo, fixed_params, cli_path)
        nc = sum(1 for g in topo.genes if g.comp_type in ("BiaxialCrystalBlock", "NormalCrystalBlock"))
        topo.efficiency = eff
        topo.best_sequence = seq
        topo.loss = -(eff * (0.98 ** nc) * 100) + (0.1 * len(topo.genes))
        _log_eval(topo_hash, fixed_params, eff, seq, 1)
        
        # Fill monitor progress for skipped evals (write a newline per skipped eval for monitor.sh)
        try:
            with open(MONITOR_COMP, "a") as f:
                f.write(".\n" * (max_evals - 1))
        except: pass
        return topo

    # 3. CMA-ES optimization
    n_dim = len(opt_specs)
    
    dummy_added = False
    if n_dim == 1:
        # Workaround for cma 1D bug where it throws 0-dimensional array IndexError
        opt_specs.append(("dummy_cma_var", 0.0, 1.0, 0.5))
        n_dim = 2
        dummy_added = True

    x0 = np.array([s[3] for s in opt_specs])
    lower = [s[1] for s in opt_specs]
    upper = [s[2] for s in opt_specs]
    
    # Use CMA_stds to decouple step sizes across mixed domains (angles vs lengths)
    stds = [(u - l) for u, l in zip(upper, lower)]
    if dummy_added:
        stds[-1] = 1.0
        
    sigma0 = 0.25  # Relative step size: 25% of the range defined by CMA_stds
    if warm_start:
        sigma0 *= 0.5  # Tighter search when refining known-good params

    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {
        'bounds': [lower, upper],
        'CMA_stds': stds,
        'maxfevals': max_evals,
        'popsize': max(4, min(8, max_evals // 3)),
        'verbose': -9,
        'seed': random.randint(0, 99999),
    })

    best_eff = max(topo.efficiency, 0.0)
    best_seq = topo.best_sequence
    best_all_params = warm_start.copy() if warm_start else dict(fixed_params)
    improved_in_cma = False
    n_total = 0
    no_improve = 0

    while not es.stop():
        solutions = es.ask()
        fitnesses = []
        for x in solutions:
            all_params = dict(fixed_params)
            for j, (key, lo, hi, _) in enumerate(opt_specs):
                if key == "dummy_cma_var":
                    continue
                all_params[key] = float(np.clip(x[j], lo, hi))
            eff, seq = _run_single_lwe(topo, all_params, cli_path)
            fitnesses.append(-eff)  # CMA-ES minimizes
            n_total += 1
            if eff > best_eff + 1e-6:
                best_eff = eff
                best_seq = seq
                best_all_params = all_params.copy()
                improved_in_cma = True
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break
        if no_improve >= patience:
            # Record early stop for monitor
            try:
                with open(MONITOR_STOP, "a") as f:
                    f.write(f"{topo_hash} | Early stop @ {n_total} evals\n")
            except: pass
            break
        es.tell(solutions, fitnesses)

    # 4. Log and finalize
    _log_eval(topo_hash, best_all_params, best_eff, best_seq, n_total)
    
    # Fill monitor progress for early-stopped evals (write a newline per skipped eval for monitor.sh)
    if n_total < max_evals:
        try:
            with open(MONITOR_COMP, "a") as f:
                f.write(".\n" * (max_evals - n_total))
        except: pass

    nc = sum(1 for g in topo.genes if g.comp_type in ("BiaxialCrystalBlock", "NormalCrystalBlock"))
    topo.efficiency = best_eff
    topo.best_sequence = best_seq
    topo.loss = -(best_eff * (0.98 ** nc) * 100) + (0.1 * len(topo.genes))
    topo._best_params = best_all_params  # For warm-start on retrain
    topo._improved_in_cma = improved_in_cma

    return topo


def get_adaptive_mutation_probs(mutation_stats):
    """Compute adaptive mutation probabilities from historical success rates."""
    actions = ["insert_instruction", "delete_instruction", "swap_order", "replace_gene"]
    rates = []
    for a in actions:
        s = mutation_stats[a]["success"]
        t = mutation_stats[a]["total"]
        rates.append((s + 1) / (t + 3))  # Laplace smoothing
    rates = np.array(rates)
    probs = np.exp(rates / 0.5)  # Softmax with temperature 0.5
    probs /= probs.sum()
    return probs.tolist()

def evaluate_candidates_parallel(candidates, evaluated_hashes, surrogate, mutation_stats, gen, CLI_PATH):
    """Run CMA-ES evaluation for a batch of candidates in parallel."""
    if not candidates:
        return
        
    try:
        with open(MONITOR_STATUS, "w") as f:
            f.write(f"GEN={gen}\n")
            f.write(f"TOTAL={len(candidates) * CMA_MAX_EVALS}\n")
        open(MONITOR_COMP, "w").close()
        open(MONITOR_STOP, "w").close()
    except Exception:
        pass
        
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for i, topo in enumerate(candidates):
            warm = getattr(topo, '_best_params', None)
            futures[pool.submit(evaluate_topology_cmaes, topo, CLI_PATH, CMA_MAX_EVALS, CMA_PATIENCE, warm)] = i
            
        for future in as_completed(futures):
            idx = futures[future]
            updated_topo = future.result()
            
            is_retrain = hasattr(candidates[idx], '_prev_efficiency')
            improved = getattr(updated_topo, '_improved_in_cma', False)
            
            if not is_retrain or improved:
                candidates[idx].loss = updated_topo.loss
                candidates[idx].efficiency = updated_topo.efficiency
                candidates[idx].best_sequence = updated_topo.best_sequence
                candidates[idx]._best_params = getattr(updated_topo, '_best_params', None)
                
                # Update cache and surrogate
                h = candidates[idx].get_hash()
                evaluated_hashes[h] = (candidates[idx].efficiency, candidates[idx].best_sequence)
                surrogate.update(candidates[idx], candidates[idx].efficiency,
                                 parent_efficiency=getattr(candidates[idx], '_parent_efficiency', 0.0))
            
            # Track mutation success (only track once per new topology)
            mut = getattr(candidates[idx], '_last_mutation', None)
            if mut and mut in mutation_stats and not is_retrain:
                mutation_stats[mut]["total"] += 1
                if candidates[idx].efficiency > getattr(candidates[idx], '_parent_efficiency', 0.0):
                    mutation_stats[mut]["success"] += 1
                    
            # Dynamic improvement tracking for retrained topologies
            if is_retrain:
                delta_eff = candidates[idx].efficiency - candidates[idx]._prev_efficiency
                print(f"    Retrain result ({candidates[idx].get_hash()[:8]}): Δη = {delta_eff*100:+.4f}%")

def select_retrain_candidates(population, champion, surrogate):
    """Select topologies that show promise for further CMA-ES tuning."""
    retrain_candidates = []
    if champion.loss < float("inf"):
        retrain_candidates.append(champion)
        
    valid_others = [t for t in population if t.loss < float("inf") and t != champion]
    if valid_others and surrogate.is_fitted:
        best_potential = max(valid_others, key=lambda x: surrogate.predict(x))
        if surrogate.predict(best_potential) > 0:
            retrain_candidates.append(best_potential)
    elif valid_others:
        best_potential = max(valid_others, key=lambda x: surrogate.novelty_score(x))
        if surrogate.novelty_score(best_potential) > 0.2:
            retrain_candidates.append(best_potential)
            
    # Take snapshots for dynamic improvement tracking
    for t in retrain_candidates:
        t._prev_efficiency = t.efficiency
        
    return retrain_candidates

def reproduce_population(valid_parents, survivors, mut_probs, evaluated_hashes, surrogate, gen):
    """Generate offspring, inject random topologies, and screen using the surrogate."""
    ranks = np.arange(len(valid_parents))
    probs = np.exp(-ranks / (max(len(valid_parents), 1) * 1.5))
    probs /= probs.sum()
    
    n_children = POP_SIZE - survivors
    n_overgenerate = n_children * SURROGATE_OVERGENERATE
    
    all_candidates = []
    seen_in_batch = set()
    max_attempts = n_overgenerate * 5
    attempts = 0
    
    while len(all_candidates) < n_overgenerate and attempts < max_attempts:
        attempts += 1
        parent_idx = np.random.choice(len(valid_parents), p=probs)
        parent = valid_parents[parent_idx]
        child = parent.mutate(mutation_probs=mut_probs)
        child._parent_efficiency = parent.efficiency
        
        while random.random() < 0.3:
            child = child.mutate(mutation_probs=mut_probs)
        
        h = child.get_hash()
        if h not in evaluated_hashes and h not in seen_in_batch:
            seen_in_batch.add(h)
            all_candidates.append(child)
        elif attempts > max_attempts * 0.7:
            all_candidates.append(child)
            
    # Diversity injection
    n_random = max(2, n_overgenerate // 5)
    for _ in range(n_random):
        rtopo = random_topology()
        h = rtopo.get_hash()
        if h not in evaluated_hashes and h not in seen_in_batch:
            seen_in_batch.add(h)
            all_candidates.append(rtopo)
            
    novel_count = sum(1 for c in all_candidates if c.get_hash() not in evaluated_hashes)
    random_count = sum(1 for c in all_candidates if getattr(c, '_parent_efficiency', 0.0) == 0.0 and c.efficiency == 0.0)
    print(f"  Reproduction: {len(all_candidates)} candidates generated "
          f"({novel_count} novel, {random_count} random, {attempts} mutation attempts)")
          
    return surrogate.screen(all_candidates, n_children, generation=gen, max_generations=N_GENERATIONS)

def run_evolution():
    """Main Orchestration Loop."""
    print("=" * 60)
    print("Memetic Optical Topology Discovery v5")
    print("  Surrogate Pre-Screening + Adaptive Mutation + CMA-ES")
    print("=" * 60)
    print(f"Population: {POP_SIZE} | Generations: {N_GENERATIONS}")
    print(f"CMA-ES Max Evals: {CMA_MAX_EVALS}")
    print(f"Max Workers: {MAX_WORKERS}")
    print(f"Surrogate activates after {SURROGATE_MIN_DATA} evaluations")
    
    surrogate = SurrogateModel()

def get_adaptive_mutation_probs(mutation_stats):
    """Compute adaptive mutation probabilities from historical success rates."""
    actions = ["insert_instruction", "delete_instruction", "swap_order", "replace_gene"]
    rates = []
    for a in actions:
        s = mutation_stats[a]["success"]
        t = mutation_stats[a]["total"]
        rates.append((s + 1) / (t + 3))  # Laplace smoothing
    rates = np.array(rates)
    probs = np.exp(rates / 0.5)  # Softmax with temperature 0.5
    probs /= probs.sum()
    return probs.tolist()

def evaluate_candidates_parallel(candidates, evaluated_hashes, surrogate, mutation_stats, gen, CLI_PATH):
    """Run CMA-ES evaluation for a batch of candidates in parallel."""
    if not candidates:
        return
        
    try:
        with open(MONITOR_STATUS, "w") as f:
            f.write(f"GEN={gen}\n")
            f.write(f"TOTAL={len(candidates) * CMA_MAX_EVALS}\n")
        open(MONITOR_COMP, "w").close()
        open(MONITOR_STOP, "w").close()
    except Exception:
        pass
        
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for i, topo in enumerate(candidates):
            warm = getattr(topo, '_best_params', None)
            futures[pool.submit(evaluate_topology_cmaes, topo, CLI_PATH, CMA_MAX_EVALS, CMA_PATIENCE, warm)] = i
            
        for future in as_completed(futures):
            idx = futures[future]
            updated_topo = future.result()
            
            is_retrain = hasattr(candidates[idx], '_prev_efficiency')
            improved = getattr(updated_topo, '_improved_in_cma', False)
            
            if not is_retrain or improved:
                candidates[idx].loss = updated_topo.loss
                candidates[idx].efficiency = updated_topo.efficiency
                candidates[idx].best_sequence = updated_topo.best_sequence
                candidates[idx]._best_params = getattr(updated_topo, '_best_params', None)
                
                # Update cache and surrogate
                h = candidates[idx].get_hash()
                evaluated_hashes[h] = (candidates[idx].efficiency, candidates[idx].best_sequence, getattr(candidates[idx], '_best_params', None))
                surrogate.update(candidates[idx], candidates[idx].efficiency,
                                 parent_efficiency=getattr(candidates[idx], '_parent_efficiency', 0.0))
            
            # Track mutation success
            mut = getattr(candidates[idx], '_last_mutation', None)
            if mut and mut in mutation_stats:
                mutation_stats[mut]["total"] += 1
                if candidates[idx].efficiency > getattr(candidates[idx], '_parent_efficiency', 0.0):
                    mutation_stats[mut]["success"] += 1
                    
            # Dynamic improvement tracking for retrained topologies
            if hasattr(candidates[idx], '_prev_efficiency'):
                delta_eff = candidates[idx].efficiency - candidates[idx]._prev_efficiency
                print(f"    Retrain result ({candidates[idx].get_hash()[:8]}): Δη = {delta_eff*100:+.4f}%")

def select_retrain_candidates(population, champion, surrogate):
    """Select topologies that show promise for further CMA-ES tuning."""
    retrain_candidates = []
    if champion.loss < float("inf"):
        retrain_candidates.append(champion)
        
    valid_others = [t for t in population if t.loss < float("inf") and t != champion]
    if valid_others and surrogate.is_fitted:
        best_potential = max(valid_others, key=lambda x: surrogate.predict(x))
        if surrogate.predict(best_potential) > 0:
            retrain_candidates.append(best_potential)
    elif valid_others:
        best_potential = max(valid_others, key=lambda x: surrogate.novelty_score(x))
        if surrogate.novelty_score(best_potential) > 0.2:
            retrain_candidates.append(best_potential)
            
    # Take snapshots for dynamic improvement tracking
    for t in retrain_candidates:
        t._prev_efficiency = t.efficiency
        
    return retrain_candidates

def reproduce_population(valid_parents, survivors, mut_probs, evaluated_hashes, surrogate, gen):
    """Generate offspring, inject random topologies, and screen using the surrogate."""
    ranks = np.arange(len(valid_parents))
    probs = np.exp(-ranks / (max(len(valid_parents), 1) * 1.5))
    probs /= probs.sum()
    
    n_children = POP_SIZE - survivors
    n_overgenerate = n_children * SURROGATE_OVERGENERATE
    
    all_candidates = []
    seen_in_batch = set()
    max_attempts = n_overgenerate * 5
    attempts = 0
    
    while len(all_candidates) < n_overgenerate and attempts < max_attempts:
        attempts += 1
        parent_idx = np.random.choice(len(valid_parents), p=probs)
        parent = valid_parents[parent_idx]
        child = parent.mutate(mutation_probs=mut_probs)
        child._parent_efficiency = parent.efficiency
        
        while random.random() < 0.3:
            child = child.mutate(mutation_probs=mut_probs)
        
        h = child.get_hash()
        if h not in evaluated_hashes and h not in seen_in_batch:
            seen_in_batch.add(h)
            all_candidates.append(child)
        elif attempts > max_attempts * 0.7:
            all_candidates.append(child)
            
    # Diversity injection
    n_random = max(2, n_overgenerate // 5)
    for _ in range(n_random):
        rtopo = random_topology()
        h = rtopo.get_hash()
        if h not in evaluated_hashes and h not in seen_in_batch:
            seen_in_batch.add(h)
            all_candidates.append(rtopo)
            
    novel_count = sum(1 for c in all_candidates if c.get_hash() not in evaluated_hashes)
    random_count = sum(1 for c in all_candidates if getattr(c, '_parent_efficiency', 0.0) == 0.0 and c.efficiency == 0.0)
    print(f"  Reproduction: {len(all_candidates)} candidates generated "
          f"({novel_count} novel, {random_count} random, {attempts} mutation attempts)")
          
    return surrogate.screen(all_candidates, n_children, generation=gen, max_generations=N_GENERATIONS)

def run_evolution():
    """Main Orchestration Loop."""
    print("=" * 60)
    print("Memetic Optical Topology Discovery v5")
    print("  Surrogate Pre-Screening + Adaptive Mutation + CMA-ES")
    print("=" * 60)
    print(f"Population: {POP_SIZE} | Generations: {N_GENERATIONS}")
    print(f"CMA-ES Max Evals: {CMA_MAX_EVALS}")
    print(f"Max Workers: {MAX_WORKERS}")
    print(f"Surrogate activates after {SURROGATE_MIN_DATA} evaluations")
    
    surrogate = SurrogateModel()
    evaluated_hashes = {}
    mutation_stats = {
        "insert_instruction": {"success": 0, "total": 0},
        "delete_instruction": {"success": 0, "total": 0},
        "swap_order":         {"success": 0, "total": 0},
        "replace_gene":       {"success": 0, "total": 0},
    }
    population = []
    
    # Step 0: Base line sequence
    ORIGINAL_SEQUENCE = (
        "init()"
        "rotateIntoBiaxial(d,d,d,d)nonlinear(d,d,d,d,d)rotateFromBiaxial(d,d,d,d)"
        "rotateIntoBiaxial(d,40.2,d,d)nonlinear(d,40.2,d,250,d)rotateFromBiaxial(d,40.2,d,d)"
        "rotate(180)"
        "rotateIntoBiaxial(d,139.8,d,d)nonlinear(d,139.8,d,250,d)rotateFromBiaxial(d,139.8,d,d)"
    )
    print("\n" + "=" * 60)
    print("Step 0: Evaluating ORIGINAL Postdoc Sequence (unmodified)...")
    
    baseline_dir = os.path.join(TMP_DIR, "baseline_original")
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_runner = lwe.SimulationRunner(cli_path=CLI_PATH, work_dir=baseline_dir)
    baseline_eff = 0.0
    try:
        baseline_runner.set_params(
            sequence=ORIGINAL_SEQUENCE,
            pulse_energy1=1.9e-08, frequency1=1.3e14, bandwidth1=1.5e13,
            sg_order1=2, beamwaist1=1.12e-5, delay1=-1.3e-13, polarization1=1.5707963267949,
            pulse_energy2=0, frequency2=4.9e14, bandwidth2=2e13,
            sg_order2=4, gdd2=1e-28, beamwaist2=9e-5, polarization2=1.5707963267949,
            material_index=4, crystal_theta=0, crystal_phi=0,
            crystal_thickness=0.0004, dz=5e-7, grid_width=2.08e-4, grid_height=2.08e-4, dx=8e-6,
            time_span=6e-13, dt=5e-16, band_gap=6, effective_mass=1, drude_gamma=5e12, propagation_mode=2
        )
        baseline_runner.run(verbose=False, timeout=600)
        freq = baseline_runner.result.frequencyVectorSpectrum
        spectrum = baseline_runner.result.spectrumTotal
        if spectrum.ndim > 1: spectrum = spectrum[-1]
        thg_center = 390e12
        mask_thg = np.abs(freq - thg_center) < 30e12
        INPUT_ENERGY = 1.9e-08
        baseline_eff = np.trapezoid(spectrum[mask_thg], freq[mask_thg]) / INPUT_ENERGY if INPUT_ENERGY > 0 else 0.0
        print(f">>> ORIGINAL BASELINE THG Yield: {baseline_eff * 100:.4f}% <<<")
    except Exception as e:
        print(f">>> BASELINE FAILED: {e}")
    finally:
        baseline_runner.cleanup()
        shutil.rmtree(baseline_dir, ignore_errors=True)
        
    # Step 1: Initial Population
    original = TopologyChromosome([
        AtomicGene('BiaxialCrystalBlock'), AtomicGene('BiaxialCrystalBlock'),
        AtomicGene('RotateFrame'), AtomicGene('BiaxialCrystalBlock')
    ])
    original.normalize()
    original.efficiency = baseline_eff
    original.loss = -(baseline_eff * (0.98 ** 3) * 100) + (0.1 * len(original.genes))
    original.best_sequence = ORIGINAL_SEQUENCE
    original._best_params = {
        "g0_theta": 0.0, "g0_phi": 0.0, "g0_length_um": 400.0,
        "g1_theta": 40.2, "g1_phi": 0.0, "g1_length_um": 250.0,
        "g2_angle": 180.0,
        "g3_theta": 139.8, "g3_phi": 0.0, "g3_length_um": 250.0
    }
    population.append(original)
    evaluated_hashes[original.get_hash()] = (original.efficiency, original.best_sequence, getattr(original, '_best_params', None))
    surrogate.update(original, original.efficiency, parent_efficiency=0.0)
    
    for _ in range(POP_SIZE - 1):
        child = copy.deepcopy(original)
        for _ in range(random.randint(1, 3)):
            child = child.mutate(mutation_probs=[0.25]*4)
        population.append(child)
        
    # Step 2: Main Evolution Loop
    for gen in range(N_GENERATIONS):
        print(f"\n{'='*60}\n--- Generation {gen} ---")
        mut_probs = get_adaptive_mutation_probs(mutation_stats)
        print(f"Adaptive mutation: insert={mut_probs[0]:.2f} delete={mut_probs[1]:.2f} swap={mut_probs[2]:.2f} replace={mut_probs[3]:.2f}")
        
        # Phase 1: Screen new candidates
        unevaluated = [t for t in population if t.loss == float("inf")]
        truly_new = []
        cache_hits = 0
        for t in unevaluated:
            h = t.get_hash()
            if h in evaluated_hashes:
                entry = evaluated_hashes[h]
                if len(entry) == 3:
                    t.efficiency, t.best_sequence, t._best_params = entry
                else:
                    t.efficiency, t.best_sequence = entry
                    t._best_params = None
                nc = sum(1 for g in t.genes if g.comp_type in ("BiaxialCrystalBlock", "NormalCrystalBlock"))
                t.loss = -(t.efficiency * (0.98 ** nc) * 100) + (0.1 * len(t.genes))
                cache_hits += 1
            else:
                truly_new.append(t)
        if cache_hits > 0:
            print(f"  Cache hits: {cache_hits} topologies skipped")
            
        elites = truly_new
        if elites:
            print(f"Phase 1: Passing {len(elites)} new candidates to CMA-ES...")
            
        # Phase 1.5: Select retrain candidates
        population.sort(key=lambda x: x.loss)
        champion = population[0]
        retrain_list = select_retrain_candidates(population, champion, surrogate)
        for t in retrain_list:
            reason = "Champion" if t == champion else "Top Potential"
            print(f"Phase 1.5: Retraining {reason} (Hash: {t.get_hash()})")
            elites.append(t)
            
        # Phase 2: Parallel Evaluation
        if elites:
            print(f"Phase 2: CMA-ES Tuning ({len(elites)} topologies, up to {CMA_MAX_EVALS} evals each)...")
            evaluate_candidates_parallel(elites, evaluated_hashes, surrogate, mutation_stats, gen, CLI_PATH)
            
            if HAS_TORCH and len(surrogate.topo_history) >= SURROGATE_MIN_DATA:
                surrogate.train()
                
        # Phase 3: Selection
        population.sort(key=lambda x: x.loss)
        best = population[0]
        survivors = POP_SIZE // 2
        
        # Get trial count
        total_trials = 0
        try:
            if os.path.exists(EVAL_LOG):
                with open(EVAL_LOG, "r") as f:
                    for line in f:
                        entry = json.loads(line)
                        if entry["topo"] == best.get_hash():
                            total_trials += entry.get("n", 1)
        except:
            pass
        total_trials = total_trials if total_trials > 0 else "N/A"
        
        alive = sum(1 for t in population if t.loss < float("inf"))
        print(f"\nGen {gen} Summary:")
        print(f"  Best: {best.efficiency*100:.4f}% | Loss: {best.loss:.6f} | Genes: {len(best.genes)} | Trials: {total_trials}")
        print(f"  Topology: {best.get_hash()}")
        print(f"  Sequence: {best.best_sequence}")
        print(f"  Alive: {alive}/{POP_SIZE} | Unique eval: {len(evaluated_hashes)} | Surrogate active: {surrogate.is_fitted}")
        
        # Phase 4: Reproduction
        valid_parents = [t for t in population[:survivors] if t.loss < float("inf")]
        if not valid_parents:
            valid_parents = [original]
            
        selected_children = reproduce_population(valid_parents, survivors, mut_probs, evaluated_hashes, surrogate, gen)
        for i, child in enumerate(selected_children):
            population[survivors + i] = child
            
    # Final Output
    print("\n" + "=" * 60)
    print("[✔] Memetic Topology Evolution Complete!")
    print(f"Best Topology: {best.get_hash()}")
    print(f"Best Efficiency: {best.efficiency*100:.4f}%")
    print(f"Best Sequence: {best.best_sequence}")
    
    try:
        with open(OUTPUT_FILE, "w") as f:
            f.write(f"Topology Hash: {best.get_hash()}\nSequence: {best.best_sequence}\nYield: {best.efficiency*100:.4f}%\n\n")
            for h, eff in sorted(evaluated_hashes.items(), key=lambda x: -x[1]):
                f.write(f"  {h}: {eff*100:.4f}%\n")
        print(f"[Save] Data written to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Failed to write: {e}")

if __name__ == '__main__':
    run_evolution()
