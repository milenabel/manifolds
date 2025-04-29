import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json

# Settings
xi = 4
fig_save_dir = "../figs/comparison_all"

# Folders for each architecture
vanilla_dir = "../results/vanilla"
new_dir = "../results/new"
fusion_dir = "../results/fusion"
fusion2_dir = "../results/fusion2"

os.makedirs(fig_save_dir, exist_ok=True)

# Helper to load results
def load_results(folder):
    files = sorted(glob.glob(f"{folder}/deeponet_results_torus_N*_xi{xi}.json"))
    results = {}
    for fpath in files:
        base = os.path.basename(fpath)
        parts = base.replace(".json", "").split("_")
        N = int(parts[3][1:])
        with open(fpath, "r") as f:
            data = json.load(f)
            results[N] = data
    return results

# Load all results
vanilla_results = load_results(vanilla_dir)
new_results = load_results(new_dir)
fusion_results = load_results(fusion_dir)
fusion2_results = load_results(fusion2_dir)

# Common N values
common_Ns = sorted(set(vanilla_results) & set(new_results) & set(fusion_results) & set(fusion2_results))

# Color map per architecture 
colors = {
    "Vanilla": "tab:blue",
    "New": "tab:green",
    "Fusion": "tab:orange",
    "Fusion2": "tab:purple"
}

# For each N, create comparison plots
for N in common_Ns:
    epochs = list(range(len(vanilla_results[N]["train_errors"])))

    plt.figure(figsize=(12, 7))

    # Plot each architecture
    for name, res, color in [
        ("Vanilla", vanilla_results, colors["Vanilla"]),
        ("New", new_results, colors["New"]),
        ("Fusion", fusion_results, colors["Fusion"]),
        ("Fusion2", fusion2_results, colors["Fusion2"])
    ]:
        train_errors = res[N]["train_errors"]
        test_errors = res[N]["test_errors"]
        gen_error = res[N]["generalization_error"]

        plt.plot(epochs, train_errors, label=f"{name} Train", color=color)
        plt.plot(epochs, test_errors, '--', label=f"{name} Test", color=color)
        plt.hlines(gen_error, xmin=0, xmax=epochs[-1], colors=color, linestyles='dotted', label=f"{name} Gen ({gen_error:.2e})")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Relative L2 Error", fontsize=12)
    plt.title(f"Comparison: All Architectures (N={N}, xi={xi})", fontsize=14)
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{fig_save_dir}/compare_all_N{N}.png", dpi=300)
    plt.close()

print(f"Saved all comparison plots to {fig_save_dir}")
