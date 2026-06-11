import numpy as np

def sellmeier_lorentzian(wavelengthMicrons, a):
    """
    Computes the complex refractive index using the Lorentzian Sellmeier equation.
    Matches LightwaveExplorer.py equationType == 1 exactly.
    """
    np.seterr(divide="ignore", invalid="ignore")
    # w is the angular frequency. In LWE: w = 2*pi*c / lambda
    w = 2 * np.pi * 2.99792458e8 / (1e-6 * wavelengthMicrons)
    k = 3182.607353999257
    a = np.abs(a)
    n = a[0] + k * a[1] / ((a[2] - w**2) + (a[3] * w) * 1j)
    n += k * a[4] / ((a[5] - w**2) + (a[6] * w) * 1j)
    n += k * a[7] / ((a[8] - w**2) + (a[9] * w) * 1j)
    n += k * a[10] / ((a[11] - w**2) + (a[12] * w) * 1j)
    n += k * a[13] / ((a[14] - w**2) + (a[15] * w) * 1j)
    n += k * a[16] / ((a[17] - w**2) + (a[18] * w) * 1j)
    n += k * a[19] / ((a[20] - w**2) + (a[21] * w) * 1j)
    return np.sqrt(n)

# BiBO L (rot 47deg) Sellmeier Coefficients
# These are strictly from CrystalDatabase.txt for crystal #26
a_x = np.array([1.97616, 2.3e+28, 1.0816e+32, 45.4212, 6.94916e+24, 3.24e+28, 0.111029, 1.9e+28, 1.44e+32, 374524, 7.14198e+24, 1e+28, 0.202241, 0, 0, 0, 0, 0, 0, 0, 0, 0])
a_y = np.array([1.7505, 2.296e+28, 1.0816e+32, 7.09512e+09, 1.48205e+24, 3.61e+28, 1.81721e+10, 3.39791e+28, 1.46531e+32, 5.53069e+09, 1.69512e+25, 2.7225e+28, 0.196454, 0, 0, 0, 0, 0, 0, 0, 0, 0])
a_z = np.array([1.67986, 2.3e+28, 1.0816e+32, 2.52303e+09, 1.84423e+21, 3.8416e+28, 1.94065e+13, 5.8e+28, 1.42803e+32, 8.57531e+09, 2.30604e+25, 2.7225e+28, 0.00718396, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def solve_fresnel(lam_um, theta_deg, phi_deg):
    """
    Solves the Fresnel equation for a biaxial crystal to find n_fast and n_slow.
    """
    nx = np.real(sellmeier_lorentzian(lam_um, a_x))
    ny = np.real(sellmeier_lorentzian(lam_um, a_y))
    nz = np.real(sellmeier_lorentzian(lam_um, a_z))
    
    th = np.radians(theta_deg)
    ph = np.radians(phi_deg)
    
    sx = np.sin(th) * np.cos(ph)
    sy = np.sin(th) * np.sin(ph)
    sz = np.cos(th)
    
    vx = 1.0 / nx**2
    vy = 1.0 / ny**2
    vz = 1.0 / nz**2
    
    B = sx**2 * (vy + vz) + sy**2 * (vx + vz) + sz**2 * (vx + vy)
    C = sx**2 * vy * vz + sy**2 * vx * vz + sz**2 * vx * vy
    
    discriminant = max(0.0, B**2 - 4*C)
    
    v1 = (B + np.sqrt(discriminant)) / 2.0
    v2 = (B - np.sqrt(discriminant)) / 2.0
    
    n1 = 1.0 / np.sqrt(v1)
    n2 = 1.0 / np.sqrt(v2)
    
    n_fast = min(n1, n2)
    n_slow = max(n1, n2)
    return n_fast, n_slow

def compute_delta_k(lam1, lam2, lam3, theta_deg, phi_deg):
    """
    Computes Delta k (m^-1) for all combinations of fast and slow axes.
    Process: omega1 (lam1) + omega2 (lam2) -> omega3 (lam3)
    Assuming lam1 >= lam2 > lam3 (lam3 is the generated harmonic)
    """
    n1_f, n1_s = solve_fresnel(lam1, theta_deg, phi_deg)
    n2_f, n2_s = solve_fresnel(lam2, theta_deg, phi_deg)
    n3_f, n3_s = solve_fresnel(lam3, theta_deg, phi_deg)
    
    # Wavevectors k = 2 * pi * n / lambda (in m^-1, so lambda in meters)
    k1_f = 2 * np.pi * n1_f / (lam1 * 1e-6)
    k1_s = 2 * np.pi * n1_s / (lam1 * 1e-6)
    
    k2_f = 2 * np.pi * n2_f / (lam2 * 1e-6)
    k2_s = 2 * np.pi * n2_s / (lam2 * 1e-6)
    
    k3_f = 2 * np.pi * n3_f / (lam3 * 1e-6)
    k3_s = 2 * np.pi * n3_s / (lam3 * 1e-6)
    
    # Delta k = k3 - k1 - k2
    results = {}
    # Type I combinations (same polarization for pumps)
    results["Type I: S + S -> F"] = k3_f - k1_s - k2_s
    results["Type I: F + F -> S"] = k3_s - k1_f - k2_f
    
    # Type II combinations (orthogonal polarizations for pumps)
    results["Type II: S + F -> F"] = k3_f - k1_s - k2_f
    results["Type II: F + S -> F"] = k3_f - k1_f - k2_s
    results["Type II: S + F -> S"] = k3_s - k1_s - k2_f
    results["Type II: F + S -> S"] = k3_s - k1_f - k2_s

    return results

def get_min_dk(lam1, lam2, lam3, theta_deg, phi_deg):
    """
    Returns the minimum absolute Delta k (m^-1) across all Type I and Type II combinations.
    """
    dks = compute_delta_k(lam1, lam2, lam3, theta_deg, phi_deg)
    return min(abs(dk) for dk in dks.values())

def get_walkoff_angle(lam_um, theta_deg, phi_deg, dth=0.01, dph=0.01):
    """
    Computes the spatial walk-off angle (in radians) for the fast and slow axes using finite difference.
    """
    n_f, n_s = solve_fresnel(lam_um, theta_deg, phi_deg)
    n_f_th, n_s_th = solve_fresnel(lam_um, theta_deg + dth, phi_deg)
    n_f_ph, n_s_ph = solve_fresnel(lam_um, theta_deg, phi_deg + dph)
    
    dn_f_dth = (n_f_th - n_f) / np.radians(dth)
    dn_f_dph = (n_f_ph - n_f) / np.radians(dph)
    
    dn_s_dth = (n_s_th - n_s) / np.radians(dth)
    dn_s_dph = (n_s_ph - n_s) / np.radians(dph)
    
    sin_th = np.sin(np.radians(theta_deg))
    if abs(sin_th) < 1e-6:
        sin_th = 1e-6 if sin_th >= 0 else -1e-6
        
    rho_f = np.sqrt(dn_f_dth**2 + (dn_f_dph / sin_th)**2) / n_f
    rho_s = np.sqrt(dn_s_dth**2 + (dn_s_dph / sin_th)**2) / n_s
    
    return np.arctan(rho_f), np.arctan(rho_s)

def get_group_index(lam_um, theta_deg, phi_deg, dlam=0.001):
    """
    Computes the group index n_g = n - lam * dn/dlam
    """
    n_f, n_s = solve_fresnel(lam_um, theta_deg, phi_deg)
    n_f_plus, n_s_plus = solve_fresnel(lam_um + dlam, theta_deg, phi_deg)
    n_f_minus, n_s_minus = solve_fresnel(lam_um - dlam, theta_deg, phi_deg)
    
    dn_f = (n_f_plus - n_f_minus) / (2 * dlam)
    dn_s = (n_s_plus - n_s_minus) / (2 * dlam)
    
    ng_f = n_f - lam_um * dn_f
    ng_s = n_s - lam_um * dn_s
    
    return ng_f, ng_s

