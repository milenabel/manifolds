import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import json
from deeponet_vanilla import DeepONet, BranchNet, TrunkNet

# Settings 
xi = 4
sensor_dir = "../data"
fig_save_dir = "../figs/vanilla"
model_dir = "../results/vanilla"
hidden_dims = [128, 128]
activation = "relu"
spectral_norm = False

os.makedirs(fig_save_dir, exist_ok=True)

# Gather JSON result files for training metrics
result_files = sorted(glob.glob(f"{model_dir}/deeponet_results_torus_N*_xi{xi}_sensors*.json"))
point_sets = {}

plt.figure(figsize=(8, 5))
for result_path in result_files:
    base = os.path.basename(result_path)
    parts = base.replace(".json", "").split("_")

    # Expected format: deeponet_results_torus_N3750_xi4_sensors100.json
    N = int(parts[3][1:]) #'N3750' -> 3750
    n_sensors = int(parts[5].replace("sensors", "")) #'sensors100' -> 100

    with open(result_path, "r") as f:
        results = json.load(f)

    rel_l2 = results["relative_l2_errors"]
    l2_errors = results["l2_errors"]
    epochs = list(range(len(rel_l2)))

    plt.plot(epochs, rel_l2, label=f"N={N}, sensors={n_sensors}")
    point_sets.setdefault(N, []).append((n_sensors, epochs, l2_errors))

plt.xlabel("Epoch")
plt.ylabel("Relative L2 Error")
plt.title("Relative L2 Error vs Epoch (all runs)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{fig_save_dir}/relative_l2_vs_epoch_all.png", dpi=300)
plt.close()

# L2 vs Epoch per N
for N, entries in sorted(point_sets.items()):
    plt.figure(figsize=(8, 5))
    for n_sensors, epochs, l2_errors in sorted(entries):
        plt.plot(epochs, l2_errors, label=f"{n_sensors} sensors")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Error")
    plt.title(f"L2 Error vs Epoch (N={N})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{fig_save_dir}/l2_vs_epoch_N{N}.png", dpi=300)
    plt.close()

# Prediction plots (True vs Predicted)
model_files = sorted(glob.glob(f"{model_dir}/model_N*_xi{xi}_sensors*.pt"))
for model_path in model_files:
    base = os.path.basename(model_path)
    parts = base.replace(".pt", "").split("_")

    # Expected format: model_N3750_xi4_sensors100.pt
    N = int(parts[1][1:])  #'N3750' -> 3750
    n_sensors = int(parts[3].replace("sensors", ""))  #'sensors100' -> 100

    test_file = f"{sensor_dir}/torus_N{N}_xi{xi}_f0.npz"
    sensor_file = f"{sensor_dir}/sensor_indices_{N}_xi{xi}_sensors{n_sensors}.npy"

    if not os.path.exists(test_file) or not os.path.exists(sensor_file):
        print(f"Skipping missing files for N={N}, sensors={n_sensors}")
        continue

    data = np.load(test_file)
    X, f, u_true = data["X"], data["f"], data["u_num"]
    sensor_idx = np.load(sensor_file)

    branch_input = torch.tensor(f[sensor_idx], dtype=torch.float32).unsqueeze(0).repeat(X.shape[0], 1)
    trunk_input = torch.tensor(X, dtype=torch.float32)

    branch_net = BranchNet(n_sensors, hidden_dims, 1, activation, spectral_norm)
    trunk_net = TrunkNet(X.shape[1], hidden_dims, 1, activation, spectral_norm)
    model = DeepONet(branch_net, trunk_net)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        u_pred = model(branch_input, trunk_input).squeeze().numpy()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=u_true, cmap='viridis', s=5)
    ax1.set_title("True Solution")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=u_pred, cmap='viridis', s=5)
    ax2.set_title(f"Predicted (N={N}, sensors={n_sensors})")

    plt.suptitle(f"Torus Poisson: True vs Predicted\n(N={N}, xi={xi}, sensors={n_sensors})", fontsize=14)
    plt.savefig(f"{fig_save_dir}/torus_comparison_N{N}_sensors{n_sensors}.png", dpi=300)
    plt.close()

print("All plots saved.")
