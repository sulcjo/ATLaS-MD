import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Parameters of the Realm ---
N = 100                  # Number of particles
L = 10.0                 # Length of the square grid
sigma = 1.0              # LJ length scale
epsilon = 1.0            # LJ energy scale
m = 1.0                  # Particle mass
gamma = 0.5              # Friction coefficient (Langevin damping)
kT = 1.0                 # Thermal energy (k_B * T)
dt = 0.01                # Time step
n_steps = 200            # Number of frames to simulate

np.random.seed(42)       # For reproducibility of thy results

# --- Initialization ---
pos = np.random.uniform(0, L, (N, 2))
vel = np.zeros((N, 2))

# --- The Lennard-Jones Force Engine ---
def compute_lj_force(pos, L, sigma, epsilon):
    # Pairwise displacements with periodic boundary conditions
    dr = pos[:, None, :] - pos[None, :, :]
    dr = dr - L * np.round(dr / L)

    r2 = np.sum(dr**2, axis=2)
    r = np.sqrt(r2 + 1e-12)  # Guard against division by zero

    sigma_r = sigma / r
    sigma_r6 = sigma_r**6
    sigma_r12 = sigma_r6**2

    # Force magnitude: F(r) = (24*epsilon / r) * [2*(sigma/r)^12 - (sigma/r)^6]
    f_mag = 24 * epsilon * (2 * sigma_r12 - sigma_r6) / r

    # Zero out self-interactions (diagonal) to prevent infinities
    np.fill_diagonal(f_mag, 0.0)

    # Vectorial sum: force[i] = Σ f_mag[i,j] * dr[i,j]
    force = np.sum(f_mag[:, :, None] * dr, axis=1)
    return force

# --- The Stage for Animation ---
fig, ax = plt.subplots(figsize=(6, 6))
scatter = ax.scatter(pos[:, 0], pos[:, 1], s=50, color='royalblue', edgecolor='k')
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal')
ax.set_title("2D Langevin Dynamics: A Lennard-Jones Assembly")
ax.set_xlabel("x")
ax.set_ylabel("y")

# --- The Engine of Time (Animation Loop) ---
def animate(i):
    global pos, vel

    # Compute instantaneous forces
    force = compute_lj_force(pos, L, sigma, epsilon)

    # Exact Ornstein-Uhlenbeck discretization for Langevin dynamics
    lam = gamma * dt / m
    exp_lam = np.exp(-lam)

    # Deterministic drift (friction + external force)
    vel = vel * exp_lam + (1 - exp_lam) / gamma * force

    # Stochastic thermal kicks (fluctuation-dissipation theorem)
    vel += np.sqrt(kT * (1 - exp_lam**2) / m) * np.random.randn(N, 2)

    # Advance positions
    pos = pos + vel * dt
    pos = np.mod(pos, L)  # Enforce periodic boundaries

    # Update the visual
    scatter.set_offsets(np.c_[pos[:, 0], pos[:, 1]])
    return scatter,

# --- Commence the Performance ---
anim = FuncAnimation(fig, animate, frames=n_steps, interval=30, blit=True)
plt.tight_layout()
plt.show()
