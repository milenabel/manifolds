import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
import numpy as np
from deeponet_new import DeepONetDualBranch
from utils import get_scheduler, save_results
from sklearn.model_selection import train_test_split

# L2 Error 
def compute_relative_l2_error(pred, target):
    l2_error = torch.norm(pred - target, p=2).item()
    norm = torch.norm(target, p=2).item()
    return l2_error / norm

# Data Loader 
def load_dual_branch_data(data_file, sensor_file):
    data = np.load(data_file)
    f = data["f"]
    X = data["X"]
    nr = data["nr"]
    u = data["u_num"]
    sensor_idx = np.load(sensor_file)

    branch1_inputs = []
    branch2_inputs = []
    trunk_inputs = []
    targets = []

    for i in range(len(X)):
        trunk_inputs.append(X[i])                    # (3,)
        branch2_inputs.append(nr[i])                 # (3,)
        targets.append(np.array(u[i], ndmin=1))      # (1,)
        branch1_inputs.append(f[sensor_idx])         # (n_sensors,)

    return (
        torch.tensor(branch1_inputs, dtype=torch.float32),
        torch.tensor(branch2_inputs, dtype=torch.float32),
        torch.tensor(trunk_inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        len(X)
    )

# Training Loop
def train_dual_branch_deeponet(config, save_every=1000):
    model = DeepONetDualBranch(
        input_dim_branch1=config['input_dim_branch1'],
        input_dim_branch2=config['input_dim_branch2'],
        input_dim_trunk=config['input_dim_trunk'],
        hidden_dims=config['hidden_dims'],
        output_dim=config['output_dim'],
        activation=config['activation'],
        spectral_norm=config['spectral_norm']
    )

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = get_scheduler(optimizer, config['schedule_type'])

    # 80/20 split
    b1_train, b1_test, b2_train, b2_test, trunk_train, trunk_test, tgt_train, tgt_test = train_test_split(
        config['branch1_input'], config['branch2_input'],
        config['trunk_input'], config['targets'],
        test_size=0.2, random_state=42
    )

    train_losses, test_losses = [], []
    train_rel_l2s, test_rel_l2s = [], []
    epochs = config['epochs']

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_preds = model(b1_train, b2_train, trunk_train)
        train_loss = nn.MSELoss()(train_preds, tgt_train)
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        train_rel_l2 = compute_relative_l2_error(train_preds, tgt_train)

        model.eval()
        with torch.no_grad():
            test_preds = model(b1_test, b2_test, trunk_test)
            test_loss = nn.MSELoss()(test_preds, tgt_test).item()
            test_rel_l2 = compute_relative_l2_error(test_preds, tgt_test)

        train_losses.append(train_loss.item())
        test_losses.append(test_loss)
        train_rel_l2s.append(train_rel_l2)
        test_rel_l2s.append(test_rel_l2)

        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] Train Loss: {train_loss.item():.4e}, Test Loss: {test_loss:.4e}, Rel L2 Train: {train_rel_l2:.4e}, Rel L2 Test: {test_rel_l2:.4e}")

        if epoch % save_every == 0 or epoch == epochs - 1:
            results = {
                "train_losses": train_losses,
                "test_losses": test_losses,
                "train_relative_l2_errors": train_rel_l2s,
                "test_relative_l2_errors": test_rel_l2s,
                "avg_train_relative_l2_error": sum(train_rel_l2s) / len(train_rel_l2s),
                "avg_test_relative_l2_error": sum(test_rel_l2s) / len(test_rel_l2s),
                "generalization_error": test_rel_l2s[-1],  # <- this line
                "n_sensors": config['n_sensors'],
                "N": config['N'],
                "xi": config['xi']
            }

            os.makedirs("../results/new", exist_ok=True)
            torch.save(model.state_dict(), f"../results/new/model_N{config['N']}_xi{config['xi']}_sensors{config['n_sensors']}.pt")
            save_results(results, f"../results/new/deeponet_dual_results_N{config['N']}_xi{config['xi']}_sensors{config['n_sensors']}.json")

    return min(test_rel_l2s)

# Main
if __name__ == "__main__":
    xi = 4
    sensor_counts = [100, 200, 300]
    data_files = sorted(glob.glob(f"../data/torus_N*_xi{xi}_f0.npz"))

    for file_path in data_files:
        basename = os.path.basename(file_path)
        N = int(basename.split("_")[1][1:])

        for n_sensors in sensor_counts:
            sensor_file = f"../data/sensor_indices_{N}_xi{xi}_sensors{n_sensors}.npy"
            if not os.path.exists(sensor_file):
                print(f"Skipping N={N}, sensors={n_sensors} (sensor file missing)")
                continue

            print(f"\n--- Training DualBranchDeepONet for N={N}, xi={xi}, sensors={n_sensors} ---")
            b1, b2, trunk, target, N_actual = load_dual_branch_data(file_path, sensor_file)

            config = {
                'input_dim_branch1': b1.shape[1],
                'input_dim_branch2': b2.shape[1],
                'input_dim_trunk': trunk.shape[1],
                'hidden_dims': [128, 128],
                'activation': 'relu',
                'spectral_norm': False,
                'output_dim': 1,
                'xi': xi,
                'N': N_actual,
                'n_sensors': n_sensors,
                'lr': 0.001,
                'schedule_type': 'cosine',
                'epochs': 10000,
                'branch1_input': b1,
                'branch2_input': b2,
                'trunk_input': trunk,
                'targets': target
            }

            train_dual_branch_deeponet(config)
