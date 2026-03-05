import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# =============================================================================
# 1. STYLE CONFIGURATION FOR MANUSCRIPT QUALITY
# =============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'


# =============================================================================
# 2. DEFINE PHYSICAL AND SYSTEM PARAMETERS (TARGETING 2 eV RESONANCE)
# =============================================================================
# --- Target Resonance Frequency ---
TARGET_RESONANCE_eV = 2.0
omega_dark_sq = TARGET_RESONANCE_eV**2 # Target is 4.0

# --- Set other physical parameters ---
L = 1.0                  # Half-length of the nanorods (arbitrary units)
beta_sq = 0.2            # Nonlocal response parameter squared
K = 0.5                  # Inter-rod coupling strength
gamma = 0.1              # Damping coefficient (linewidth)

# --- Calculate the required plasma frequency to hit the target resonance ---
# We use the formula: omega_dark_sq = omega_p_sq + beta_sq * k1**2 + K
# And solve for omega_p_sq.
k1 = np.pi / (2 * L)
omega_p_sq = omega_dark_sq - beta_sq * k1**2 - K

# --- Final Derived Parameters ---
omega_dark = np.sqrt(omega_dark_sq) # This will be exactly 2.0


# =============================================================================
# 3. DEFINE SYSTEM GEOMETRY AND ROD ORIENTATIONS
# =============================================================================
d = 2 * L
v1 = np.array([0, 0])
v2 = np.array([d, 0])
v3 = np.array([d/2, -d * np.sqrt(3)/2])
vertices = [v1, v2, v3]

rod_endpoints = [(v1, v2), (v2, v3), (v3, v1)]

u_doc_u1 = np.array([1.0, 0.0])
u_doc_u2 = np.array([-0.5, np.sqrt(3)/2])
u_doc_u3 = np.array([-0.5, -np.sqrt(3)/2])

u_vectors_for_calculation = [u_doc_u1, u_doc_u3, u_doc_u2]

s = np.linspace(-L, L, 200) # Spatial coordinate along a rod


# =============================================================================
# 4. CALCULATE THE SYSTEM'S RESPONSE AT THE RESONANT FREQUENCY
# =============================================================================
# --- SET THE EXTERNAL FIELD DIRECTION ---
E_EXT_DIRECTION = np.array([-1.0, 0.0]) # x-polarized field
E_EXT_DIRECTION_NORMALIZED = E_EXT_DIRECTION / np.linalg.norm(E_EXT_DIRECTION)

# --- Set the driving frequency to be exactly on resonance ---
DRIVING_FREQUENCY = TARGET_RESONANCE_eV

print(f"Visualizing response amplitude at driving frequency ω = {DRIVING_FREQUENCY:.2f} eV")
print(f"Dark mode resonant frequency is set to ω_dark = {omega_dark:.2f} eV")
print(f"External field direction: {E_EXT_DIRECTION_NORMALIZED}")

# --- Step 1: Calculate the frequency-dependent amplitude ---
# The on-resonance amplitude is A(ω=ω_n) = F / (ω_n * γ)
Amplitude_Factor = 1.0
frequency_amplitude = Amplitude_Factor / (omega_dark * gamma)

# --- Step 2: Calculate the amplitude on each rod, f_i(ω) ---
projection_coeffs = np.array([np.dot(E_EXT_DIRECTION_NORMALIZED, u_vec) for u_vec in u_vectors_for_calculation])
rod_amplitudes_fi = frequency_amplitude * projection_coeffs

print(f"Calculated projection coefficients (d_i): {projection_coeffs}")
print(f"On-resonance response amplitude: {frequency_amplitude:.3f}")

# --- Step 3: Define the spatial profile of the charge density ---
charge_density_spatial_profile = -k1 * np.cos(k1 * (s + L))

# --- Step 4: Combine to get the full charge density amplitude on each rod ---
rod_charge_density_amplitudes = [amp * charge_density_spatial_profile for amp in rod_amplitudes_fi]


# =============================================================================
# 5. PLOT THE RESULTS
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')

# --- Color Normalization to Arbitrary Units [-1, 1] ---
# Find the maximum absolute density value across all rods
max_actual_rho = np.max(np.abs(np.concatenate(rod_charge_density_amplitudes)))
if max_actual_rho == 0: max_actual_rho = 1.0 # Avoid division by zero
# Create a scaling factor to normalize all data to the range [-1, 1]
scaling_factor = 1.0 / max_actual_rho
# Set up the colormap normalization
norm = Normalize(vmin=-1, vmax=1)
cmap = plt.get_cmap('RdBu_r')

# --- Plot Each Rod ---
for i, endpoints in enumerate(rod_endpoints):
    start_point, end_point = endpoints
    x_coords = np.linspace(start_point[0], end_point[0], len(s))
    y_coords = np.linspace(start_point[1], end_point[1], len(s))

    # Apply the scaling factor to get the normalized density
    normalized_density = rod_charge_density_amplitudes[i] * scaling_factor

    points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=10)
    lc.set_array(normalized_density)
    ax.add_collection(lc)

# --- Add Atom Markers (at vertices) ---
atom_x = [v[0] for v in vertices]
atom_y = [v[1] for v in vertices]
ax.scatter(atom_x, atom_y, color='k', s=100, zorder=5)

# --- Add and Style Labels, Title, and Ticks ---
ax.set_xlabel("X (arbitrary units)", fontsize=20)
ax.set_ylabel("Y (arbitrary units)", fontsize=20)
ax.set_title(f"Induced Density", fontsize=22, pad=15)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)

# --- Add and Style Colorbar ---
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.03, aspect=25, shrink=0.9)
cbar.set_label('Normalized Induced Density\n(arbitrary units)', fontsize=20, labelpad=15)
cbar.ax.tick_params(labelsize=18)
cbar.set_ticks(np.linspace(-1, 1, 5))

# --- Adjust plot limits ---
min_x, max_x = min(v[0] for v in vertices), max(v[0] for v in vertices)
min_y, max_y = min(v[1] for v in vertices), max(v[1] for v in vertices)
padding = d * 0.2
ax.set_xlim(min_x - padding, max_x + padding)
ax.set_ylim(min_y - padding, max_y + padding)

# --- Final Layout ---
plt.tight_layout()
plt.show()
