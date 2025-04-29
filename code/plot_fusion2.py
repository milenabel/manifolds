import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import glob
import json
import csv
from deeponet_fusion2 import DeepONetFusion2, BranchNet, TrunkNet, FusionMLP

# Settings
xi = 4
fig_save_dir = "../figs/fusion2"
model_dir = "../results/fusion2"
data_dir = "../data"
hidden_dims = [128, 128]
activation = "relu"
spectral_norm = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(fig_save_dir, exist_ok=True)

# Gather result files
result_files = sorted(glob.glob(f"{model_dir}/deeponet_results_torus_N*_xi{xi}.json"))
point_sets = []

plt.figure(figsize=(10, 6))
for result_path in result_files:
    base = os.path.basename(result_path)
    parts = base.replace(".json", "").split("_")

    N = int(parts[3][1:])

    with open(result_path, "r") as f:
        results = json.load(f)

    train_errors = results["train_errors"]
    test_errors = results["test_errors"]
    gen_error = results["generalization_error"]

    epochs = list(range(len(train_errors)))

    plt.plot(epochs, train_errors, label=f"N={N} Train")
    plt.plot(epochs, test_errors, '--', label=f"N={N} Test")
    plt.hlines(gen_error, xmin=0, xmax=epochs[-1], colors='r', linestyles='dotted', label=f"N={N} Gen ({gen_error:.2e})")

    point_sets.append((N, epochs, train_errors, test_errors, gen_error))

plt.xlabel("Epoch")
plt.ylabel("Relative L2 Error")
plt.title("Train, Test, and Generalization Errors vs Epoch (Fusion DeepONet)")
plt.legend(fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{fig_save_dir}/train_test_gen_vs_epoch_all.png", dpi=300)
plt.close()

# Save per N plots
for N, epochs, train_errors, test_errors, gen_error in sorted(point_sets):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_errors, label="Train Error")
    plt.plot(epochs, test_errors, '--', label="Test Error")
    plt.hlines(gen_error, xmin=0, xmax=epochs[-1], colors='r', linestyles='dotted', label=f"Gen Error = {gen_error:.2e}")

    plt.xlabel("Epoch")
    plt.ylabel("Relative L2 Error")
    plt.title(f"Fusion2 DeepONet (N={N})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{fig_save_dir}/train_test_gen_vs_epoch_N{N}.png", dpi=300)
    plt.close()

# Save CSV summary
os.makedirs(fig_save_dir, exist_ok=True)
csv_path = f"{fig_save_dir}/summary_results.csv"
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["N", "Average Train Error", "Average Test Error", "Generalization Error"])
    for N, _, train_errors, test_errors, gen_error in sorted(point_sets):
        writer.writerow([N, f"{np.mean(train_errors):.4e}", f"{np.mean(test_errors):.4e}", f"{gen_error:.4e}"])

# Predict True vs Predicted plots
model_files = sorted(glob.glob(f"{model_dir}/model_N*_xi{xi}.pt"))
for model_path in model_files:
    base = os.path.basename(model_path)
    parts = base.replace(".pt", "").split("_")

    N = int(parts[1][1:])
    data_file = f"{data_dir}/torus_N{N}_xi{xi}_f0.npz"
    if not os.path.exists(data_file):
        continue

    data = np.load(data_file)
    X = data["X"]
    f = data["f"]
    u_true = data["u_num"]
    nr = data["nr"]

    branch_f_input = torch.tensor(f.flatten(), dtype=torch.float32).unsqueeze(0).repeat(X.shape[0], 1)
    branch_n_input = torch.tensor(nr, dtype=torch.float32)
    trunk_input = torch.tensor(X, dtype=torch.float32)

    branch_f_net = BranchNet(branch_f_input.shape[1], hidden_dims, 128)
    branch_n_net = BranchNet(branch_n_input.shape[1], hidden_dims, 128)
    trunk_net = TrunkNet(trunk_input.shape[1], hidden_dims, 128)
    fusion_net = FusionMLP(256, [128, 128], 128) 

    model = DeepONetFusion2(branch_f_net, branch_n_net, trunk_net, fusion_net).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        preds = model(branch_f_input.to(device), branch_n_input.to(device), trunk_input.to(device)).cpu().squeeze().numpy()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=u_true, cmap='viridis', s=5)
    ax1.set_title("True Solution")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=preds, cmap='viridis', s=5)
    ax2.set_title(f"Predicted Solution (N={N})")

    plt.suptitle(f"Fusion2 DeepONet Prediction (N={N}, xi={xi})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{fig_save_dir}/torus_comparison_N{N}.png", dpi=300)
    plt.close()

print("All Fusion2 DeepONet plots and summary saved!")