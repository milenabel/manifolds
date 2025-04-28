import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json

# === Settings ===
xi = 4
fig_save_dir = "../figs/comparison"
vanilla_dir = "../results/vanilla"
new_dir = "../results/new"

os.makedirs(fig_save_dir, exist_ok=True)

# === Gather Vanilla and New Results ===
vanilla_files = sorted(glob.glob(f"{vanilla_dir}/deeponet_results_torus_N*_xi{xi}.json"))
new_files = sorted(glob.glob(f"{new_dir}/deeponet_results_torus_N*_xi{xi}.json"))

vanilla_results = {}
new_results = {}

# Load vanilla
for fpath in vanilla_files:
    base = os.path.basename(fpath)
    parts = base.replace(".json", "").split("_")
    N = int(parts[3][1:])
    with open(fpath, "r") as f:
        data = json.load(f)
        vanilla_results[N] = data

# Load new
for fpath in new_files:
    base = os.path.basename(fpath)
    parts = base.replace(".json", "").split("_")
    N = int(parts[3][1:])
    with open(fpath, "r") as f:
        data = json.load(f)
        new_results[N] = data

# === Make Comparison Plots ===
common_Ns = sorted(set(vanilla_results.keys()) & set(new_results.keys()))

for N in common_Ns:
    v_data = vanilla_results[N]
    n_data = new_results[N]

    v_train = v_data["train_errors"]
    v_test = v_data["test_errors"]
    v_gen = v_data["generalization_error"]

    n_train = n_data["train_errors"]
    n_test = n_data["test_errors"]
    n_gen = n_data["generalization_error"]

    epochs = list(range(len(v_train)))

    plt.figure(figsize=(10, 6))

    # Vanilla
    plt.plot(epochs, v_train, label=f"Vanilla Train")
    plt.plot(epochs, v_test, '--', label=f"Vanilla Test")
    plt.hlines(v_gen, xmin=0, xmax=epochs[-1], colors='r', linestyles='dotted', label=f"Vanilla Gen ({v_gen:.2e})")

    # New
    plt.plot(epochs, n_train, label=f"New Train")
    plt.plot(epochs, n_test, '--', label=f"New Test")
    plt.hlines(n_gen, xmin=0, xmax=epochs[-1], colors='g', linestyles='dotted', label=f"New Gen ({n_gen:.2e})")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Relative L2 Error", fontsize=12)
    plt.title(f"Comparison: Vanilla vs New DeepONet (N={N}, xi={xi})", fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{fig_save_dir}/compare_vanilla_new_N{N}.png", dpi=300)
    plt.close()

print(f"Saved all comparison plots to {fig_save_dir}")
