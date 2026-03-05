import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# =============================================================================
# 0) STYLE (optional)
# =============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# =============================================================================
# 1) PHYSICS / SYSTEM PARAMS (you can tweak)
# =============================================================================
L = 2.0                  # half-length used in your rod coordinate s ∈ [-L, L]
omega_p_sq = 25.0
beta_sq = 2.0
K = 2.0
gamma = 1

k1 = np.pi / (2 * L)
Omega_sq = omega_p_sq + beta_sq * k1**2
omega_dark_sq = Omega_sq + K
omega_dark = np.sqrt(omega_dark_sq)
omega_damped = np.sqrt(max(0, omega_dark**2 - (gamma/2)**2))

# incident field direction (unit-ish)
E_EXT_DIRECTION = np.array([1.0, 0.0])
E_EXT_DIRECTION_NORMALIZED = E_EXT_DIRECTION / np.linalg.norm(E_EXT_DIRECTION)

# time to visualize
T_PLOT = 1.0
Amplitude_Factor = 1.0
temporal_amplitude_AD = (Amplitude_Factor * np.exp(-gamma * T_PLOT / 2) *
                         np.sin(omega_damped * T_PLOT))

# =============================================================================
# 2) GEOMETRY: downward triangle (your earlier choice), then use as line charges
# =============================================================================
d = 2 * L
v1 = np.array([0.0, 0.0])
v2 = np.array([d, 0.0])
v3 = np.array([d/2, -d * np.sqrt(3)/2])
vertices = [v1, v2, v3]

# rods (clockwise): v1->v2 (bottom), v2->v3 (right), v3->v1 (left)
rod_endpoints = [
    (v1, v2),  # rod 1
    (v2, v3),  # rod 2
    (v3, v1)   # rod 3
]

# document unit vectors ↔ rods (as in your code)
u_doc_u1 = np.array([ 1.0, 0.0])
u_doc_u2 = np.array([-0.5,  np.sqrt(3)/2])
u_doc_u3 = np.array([-0.5, -np.sqrt(3)/2])

u_vectors_for_calculation = [
    u_doc_u1,  # rod 1 (v1->v2)
    u_doc_u3,  # rod 2 (v2->v3)
    u_doc_u2   # rod 3 (v3->v1)
]

# rod-parameter s and spatial charge profile
s = np.linspace(-L , L , 200)                        # along each rod
charge_density_spatial_profile = -k1 * np.cos(k1 * (s + L))

# time/drive projection (d_i) and per-rod densities
projection_coeffs = np.array([np.dot(E_EXT_DIRECTION_NORMALIZED, u) for u in u_vectors_for_calculation])
rod_amplitudes_fi = temporal_amplitude_AD * projection_coeffs
rod_charge_densities = [amp * charge_density_spatial_profile for amp in rod_amplitudes_fi]
# Each element is shape (len(s),), one per rod in the same order as rod_endpoints.

# =============================================================================
# 3) DISCRETIZE PERIMETER (clockwise) AND BUILD LINE CHARGES
# =============================================================================
def discretize_rods_clockwise(rod_endpoints, s):
    """
    Returns:
      xs, ys : (N_total,) source coordinates along all rods, clockwise
      ds     : (N_total,) arc-length weight for each source sample
      idxs   : slices mapping each rod to its portion in the flat arrays
    Notes:
      Uses midpoints over N equal subsegments for each edge.
    """
    xs_all, ys_all, ds_all = [], [], []
    idxs = []
    N = len(s)
    t_mid = (np.arange(N) + 0.5) / N
    total = 0
    for (P, Q) in rod_endpoints:  # clockwise order
        P = np.asarray(P); Q = np.asarray(Q)
        vec = Q - P
        L_edge = np.hypot(vec[0], vec[1])
        pts = P[None, :] + t_mid[:, None] * vec[None, :]
        xs_all.append(pts[:, 0])
        ys_all.append(pts[:, 1])
        ds_all.append(np.full(N, L_edge / N))
        idxs.append(slice(total, total + N))
        total += N
    xs = np.concatenate(xs_all)
    ys = np.concatenate(ys_all)
    ds = np.concatenate(ds_all)
    return xs, ys, ds, idxs

def assemble_lambda_perimeter(rod_charge_densities, idxs):
    lam = np.empty(sum(sl.stop - sl.start for sl in idxs), dtype=float)
    for i, sl in enumerate(idxs):
        lam[sl] = rod_charge_densities[i]  # shape (len(s),)
    return lam

xs, ys, ds_per, idxs = discretize_rods_clockwise(rod_endpoints, s)
lam_per = assemble_lambda_perimeter(rod_charge_densities, idxs)
# --- Taper rho near vertices (apply per-edge window) ---
margin_frac = 0.10   # 10% of each edge length gets "smudged"; tweak this
N = len(s)           # samples per edge
M = max(1, int(np.ceil(margin_frac * N)))

# Smooth window: 0 at ends, 1 in the middle
i = np.arange(N)
left_ramp  = 0.5 * (1 - np.cos(np.pi * (i+1) / (M+1)))      # rises 0→1 over M pts
right_ramp = 0.5 * (1 - np.cos(np.pi * (N - i) / (M+1)))    # symmetric at the other end
w_edge = np.ones(N)
w_edge[:M]  = left_ramp[:M]
w_edge[-M:] = right_ramp[-M:]

# Expand to all three edges using idxs, then apply to lam_per
w_flat = np.empty_like(lam_per)
for sl in idxs:
    w_flat[sl] = w_edge

lam_per *= w_flat

# =============================================================================
# 4) FIELD / POTENTIAL FROM LINE CHARGES
# =============================================================================
def field_from_line_charge_on_grid(x, y, xs, ys, ds, lam, k=1.0, soft=0.0, skip_radius=0.0):
    X, Y = np.meshgrid(x, y, indexing='xy')
    Ex = np.zeros_like(X, dtype=float)
    Ey = np.zeros_like(Y, dtype=float)
    soft2 = soft**2
    skip2 = skip_radius**2
    for j in range(xs.size):
        rx = X - xs[j]
        ry = Y - ys[j]
        r2 = rx*rx + ry*ry
        if skip_radius > 0:
            mask = r2 >= skip2
        else:
            mask = True
        r3 = (r2 + soft2)**1.5
        Ex = Ex + (k * lam[j] * rx / r3 * ds[j]) if mask is True else Ex + (k * lam[j] * (rx / r3) * ds[j]) * mask
        Ey = Ey + (k * lam[j] * ry / r3 * ds[j]) if mask is True else Ey + (k * lam[j] * (ry / r3) * ds[j]) * mask
    return Ex, Ey

def potential_from_line_charge_on_grid(x, y, xs, ys, ds, lam, k=1.0, soft=0.0):
    X, Y = np.meshgrid(x, y, indexing='xy')
    V = np.zeros_like(X, dtype=float)
    soft2 = soft**2
    for j in range(xs.size):
        r2 = (X - xs[j])**2 + (Y - ys[j])**2 + soft2
        V += k * lam[j] * ds[j] / np.sqrt(r2)
    return V

# =============================================================================
# 5) PLOT GRIDS / BOUNDS
# =============================================================================
min_x = min(v[0] for v in vertices)
max_x = max(v[0] for v in vertices)
min_y = min(v[1] for v in vertices)
max_y = max(v[1] for v in vertices)
padding = d * 0.2

gx = np.linspace(min_x - padding, max_x + padding, 161)
gy = np.linspace(min_y - padding, max_y + padding, 161)
GX, GY = np.meshgrid(gx, gy, indexing='xy')

soft = 0.01 * (gx[1] - gx[0])  # mild softening

Ex, Ey = field_from_line_charge_on_grid(gx, gy, xs, ys, ds_per, lam_per,
                                        k=1.0, soft=soft, skip_radius=0.0)

V = potential_from_line_charge_on_grid(gx, gy, xs, ys, ds_per, lam_per,
                                       k=1.0, soft=soft)

# =============================================================================
# 6) PLOT: Potential (RdBu) + Field Lines
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 8))
ax.set_aspect('equal')

vlim = np.max(np.abs(V));  vlim = 1.0 if vlim == 0 else vlim
levels = np.linspace(-vlim, vlim, 41)
cn = ax.contourf(GX, GY, V, levels=levels, cmap='RdBu_r', alpha=0.9)

# streamlines of E (use constant linewidth for robustness)
ax.streamplot(gx, gy, Ex, Ey, color='k', density=1.2, linewidth=1.1, arrowsize=1.0)

# triangle edges
for (P, Q) in rod_endpoints:
    ax.plot([P[0], Q[0]], [P[1], Q[1]], 'k-', lw=2)

ax.set_xlim(min_x - padding, max_x + padding)
ax.set_ylim(min_y - padding, max_y + padding)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Line-Charge Potential (RdBu) with Electric Field Lines")

cbar = fig.colorbar(cn, ax=ax, pad=0.02, aspect=25)
cbar.set_label('Potential (arb. units)')

plt.tight_layout()
plt.show()

# =============================================================================
# 7) FIELD ENHANCEMENT: PLOT IN NORMALIZED ARBITRARY UNITS
# =============================================================================
E0 = 1.0
# The incident field direction is taken from the top of the script
Einc_x = E0 * E_EXT_DIRECTION_NORMALIZED[0]
Einc_y = E0 * E_EXT_DIRECTION_NORMALIZED[1]

Etot_x = Ex + Einc_x
Etot_y = Ey + Einc_y
mag_tot = np.hypot(Etot_x, Etot_y)
mag_inc = np.hypot(Einc_x, Einc_y)

enh = mag_tot / mag_inc

# <<< CHANGE: NORMALIZE THE ENHANCEMENT FOR AN "ARBITRARY UNITS" PLOT >>>
# Find the maximum enhancement value (clipping the extreme top 0.5% for stability)
max_enh = np.percentile(enh, 99.5)
if max_enh == 0: max_enh = 1.0 # Avoid division by zero

normalized_enh = enh / max_enh

# --- Plot 1: Linear Scale Normalized Enhancement ---
fig_enh, ax_enh = plt.subplots(figsize=(9, 8))
ax_enh.set_aspect('equal')

# Plot the normalized data, clipping at 1 for a clean colorbar
cn2 = ax_enh.contourf(GX, GY, np.clip(normalized_enh, 0, 1), levels=np.linspace(0, 1, 50), cmap='RdBu_r')

for (P, Q) in rod_endpoints:
    ax_enh.plot([P[0], Q[0]], [P[1], Q[1]], 'w-', lw=1.5)

ax_enh.set_xlim(min_x - padding, max_x + padding)
ax_enh.set_ylim(min_y - padding, max_y + padding)
ax_enh.set_xticks([]); ax_enh.set_yticks([])

# <<< CHANGE: Update Title and Colorbar Label >>>
ax_enh.set_title("Normalized Field Enhancement (Linear Scale)", fontsize=16)
cbar2 = fig_enh.colorbar(cn2, ax=ax_enh, pad=0.02, aspect=25)
cbar2.set_label('Relative Enhancement', fontsize=14)
cbar2.set_ticks(np.linspace(0, 1, 5)) # Set ticks from 0 to 1

plt.tight_layout()
plt.show()

# --- Plot 2: Log Scale Enhancement (Dynamic Range) ---
# For the log plot, we still show the original data to see the orders of magnitude,
# but we emphasize in the label that the absolute scale is arbitrary.
eps_log = 1e-6
log_enh = np.log10(enh + eps_log)
vmin, vmax = np.percentile(log_enh, [1, 99.5])

fig_log, ax_log = plt.subplots(figsize=(9, 8))
ax_log.set_aspect('equal')
cn3 = ax_log.contourf(GX, GY, log_enh, levels=40, cmap='viridis', vmin=vmin, vmax=vmax)

for (P, Q) in rod_endpoints:
    ax_log.plot([P[0], Q[0]], [P[1], Q[1]], 'w-', lw=1.2)

ax_log.set_xlim(min_x - padding, max_x + padding)
ax_log.set_ylim(min_y - padding, max_y + padding)
ax_log.set_xticks([]); ax_log.set_yticks([])

# <<< CHANGE: Update Title and Colorbar Label >>>
ax_log.set_title("Field Enhancement (Log Scale)", fontsize=16)
cbar3 = fig_log.colorbar(cn3, ax=ax_log, pad=0.02, aspect=25)
cbar3.set_label('log10(Enhancement) [arb. scale]', fontsize=14)

plt.tight_layout()
plt.show()
