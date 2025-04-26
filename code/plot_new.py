import os
import glob
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deeponet_new import DeepONetDualBranch

# Settings 
xi = 4
figs_dir = "../figs/new"
data_dir = "../data"
model_dir = "../results/new"
hidden_dims = [128, 128]
activation = "relu"
spectral_norm = False
results_dir = "../results/new"
figs_dir = "../figs/new"

os.makedirs(figs_dir, exist_ok=True)

# Locate all model files
model_paths = sorted(glob.glob(f"{model_dir}/model_N*_xi{xi}_sensors*.pt"))

for model_path in model_paths:
    # Parse metadata 
    base = os.path.basename(model_path)
    parts = base.replace(".pt", "").split("_")
    N = int(parts[1][1:])                 # N3750 → 3750
    n_sensors = int(parts[3].replace("sensors", ""))  # sensors100 → 100


    # Corresponding data files 
    data_file = f"{data_dir}/torus_N{N}_xi{xi}_f0.npz"
    sensor_file = f"{data_dir}/sensor_indices_{N}_xi{xi}_sensors{n_sensors}.npy"

    if not os.path.exists(data_file) or not os.path.exists(sensor_file):
        print(f"Skipping: {data_file} or {sensor_file} not found.")
        continue

    # Load data
    data = np.load(data_file)
    X = data["X"]
    nr = data["nr"]
    f = data["f"]
    u_true = data["u_num"]
    sensor_idx = np.load(sensor_file)

    # Inputs for prediction
    branch1_input = torch.tensor(f[sensor_idx], dtype=torch.float32).unsqueeze(0).repeat(X.shape[0], 1)
    branch2_input = torch.tensor(nr, dtype=torch.float32)
    trunk_input = torch.tensor(X, dtype=torch.float32)

    # Build model 
    model = DeepONetDualBranch(
        input_dim_branch1=branch1_input.shape[1],
        input_dim_branch2=branch2_input.shape[1],
        input_dim_trunk=trunk_input.shape[1],
        hidden_dims=hidden_dims,
        output_dim=1,
        activation=activation,
        spectral_norm=spectral_norm
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict
    with torch.no_grad():
        u_pred = model(branch1_input, branch2_input, trunk_input).squeeze().numpy()

    # Plot 
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=u_true, cmap='viridis', s=5)
    ax1.set_title("True Solution")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=u_pred, cmap='viridis', s=5)
    ax2.set_title(f"Predicted (N={N}, sensors={n_sensors})")

    plt.suptitle(f"DualBranch DeepONet – Torus Solution\nN={N}, xi={xi}, sensors={n_sensors}", fontsize=14)
    fname = f"{figs_dir}/torus_comparison_dual_N{N}_sensors{n_sensors}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved: {fname}")

# Collect all result files
result_files = sorted(glob.glob(f"{results_dir}/deeponet_dual_results_N*_xi{xi}_sensors*.json"))

# Group by N 
results_by_N = {}

for path in result_files:
    with open(path, "r") as f:
        data = json.load(f)

    N = data["N"]
    n_sensors = data["n_sensors"]
    rel_l2_train = data["train_relative_l2_errors"]
    rel_l2_test = data["test_relative_l2_errors"]
    gen_error = data["generalization_error"]
    epochs = list(range(len(rel_l2_train)))

    if N not in results_by_N:
        results_by_N[N] = []
    results_by_N[N].append((n_sensors, epochs, rel_l2_train, rel_l2_test, gen_error))

# Plot per N 
for N, runs in sorted(results_by_N.items()):
    plt.figure(figsize=(8, 5))

    for n_sensors, epochs, rel_l2_train, rel_l2_test, gen_error in sorted(runs):
        plt.plot(epochs, rel_l2_train, label=f"{n_sensors} sensors (train)")
        plt.plot(epochs, rel_l2_test, linestyle='--', label=f"{n_sensors} sensors (test)")
        plt.axhline(y=gen_error, linestyle=":", color="gray", label=f"{n_sensors} sensors (final test)")


    plt.xlabel("Epoch")
    plt.ylabel("Relative L2 Error")
    plt.title(f"DualBranch DeepONet – Relative L2 Error (N = {N})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{figs_dir}/dual_branch_l2_vs_epoch_N{N}.png", dpi=300)
    plt.close()

print("Dual-branch DeepONet L2 error plots saved.")
