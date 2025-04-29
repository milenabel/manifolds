import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# Paths 
data_dir = Path('../data')
fig_dir = Path('../figs')
fig_dir.mkdir(parents=True, exist_ok=True)

# Settings 
xi_list = [2, 4, 6, 8]

# Parse all matching files dynamically
files = list(data_dir.glob("torus_N*_xi*_f0.npz"))
xi_to_data = {xi: [] for xi in xi_list}

for file in files:
    name = file.stem  
    parts = name.split('_')
    N_val = int(parts[1][1:])  
    xi_val = int(parts[2][2:]) 

    if xi_val in xi_list:
        data = np.load(file)
        u_true = data['u_true']
        u_num = data['u_num']
        rel_err = np.linalg.norm(u_num - u_true) / np.linalg.norm(u_true)
        xi_to_data[xi_val].append((np.sqrt(N_val), rel_err))

        # Scatter plots of solution
        X = data['X'] * data['X_std'] + data['X_mean']  # De-normalize X for plotting

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=u_true, cmap='viridis', s=5)
        ax1.set_title("True Solution")

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=u_num, cmap='viridis', s=5)
        ax2.set_title("Numerical Solution")

        plt.tight_layout()
        plt.savefig(fig_dir / f"solution_plot_xi{xi_val}_N{N_val}.png", dpi=300)
        plt.close()

# Convergence Plot
plt.figure(figsize=(8,6))

for xi, pairs in xi_to_data.items():
    if not pairs:
        continue
    pairs.sort()  # sort by sqrt(N)
    xvals, yvals = zip(*pairs)
    plt.semilogy(xvals, yvals, marker='o', label=f"$\\xi = {xi}$")

plt.title("RBF-FD Solver Convergence on Torus (Normalized Inputs)")
plt.xlabel(r"$\sqrt{N}$ (mesh resolution)")
plt.ylabel("Relative $L^2$ Error")
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.tight_layout()

# Save plot
plot_path = fig_dir / "solver_convergence.png"
plt.savefig(plot_path, dpi=300)
print(f"Saved convergence plot to {plot_path}")
plt.show()
