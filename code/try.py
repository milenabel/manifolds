# Corrected train_new.py to properly calculate generalization error only once after training
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
import numpy as np
from deeponet_new import DeepONetDualBranch
from utils import get_scheduler, save_results
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Relative L2 Error
def compute_relative_l2_error(pred, target):
    l2_error = torch.norm(pred - target, p=2).item()
    norm = torch.norm(target, p=2).item()
    return l2_error / norm

# Data Loader for full f (no sensors)
def load_full_f_data(data_file):
    data = np.load(data_file)
    f = data['f']               # (N, 1) or (N,)
    X = data['X']               # (N, 3)
    nr = data['nr']             # (N, 3)
    u = data['u_num']           # (N,)

    N_pts = len(X)
    f_tensor = torch.tensor(f, dtype=torch.float32).T  # shape: (1, N)
    branch_inputs = f_tensor.repeat(N_pts, 1)          # shape: (N, N)
    normals = torch.tensor(nr, dtype=torch.float32)
    trunk_inputs = torch.tensor(X, dtype=torch.float32)
    targets = torch.tensor(u, dtype=torch.float32).unsqueeze(1)

    return branch_inputs, normals, trunk_inputs, targets

# Training Loop
def train_deeponet_encoder(config, generalization_files, save_every=1000):
    model = DeepONetDualBranch(
        input_dim_branch1=config['input_dim_branch1'],
        input_dim_branch2=config['input_dim_branch2'],
        input_dim_trunk=config['input_dim_trunk'],
        hidden_dims=config['hidden_dims'],
        output_dim=config['output_dim'],
        activation=config['activation'],
        spectral_norm=config['spectral_norm'],
        trunk_activation=config['trunk_activation']
    )

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = get_scheduler(optimizer, config['schedule_type'])

    train_loader = DataLoader(config['train_dataset'], batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(config['test_dataset'], batch_size=config['batch_size'], shuffle=False)

    train_losses, test_losses = [], []
    test_rel_l2s = []

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        for b1, b2, x, y in train_loader:
            optimizer.zero_grad()
            preds = model(b1, b2, x)
            loss = nn.MSELoss()(preds, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
        scheduler.step()

        avg_train_loss = running_loss / len(config['train_dataset'])

        model.eval()
        test_preds, test_targets = [], []
        with torch.no_grad():
            for b1, b2, x, y in test_loader:
                preds = model(b1, b2, x)
                test_preds.append(preds)
                test_targets.append(y)
        test_preds = torch.cat(test_preds, dim=0)
        test_targets = torch.cat(test_targets, dim=0)
        test_loss = nn.MSELoss()(test_preds, test_targets).item()
        test_rel_l2 = compute_relative_l2_error(test_preds, test_targets)

        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        test_rel_l2s.append(test_rel_l2)

        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4e}, Test Loss: {test_loss:.4e}, Rel L2 Test: {test_rel_l2:.4e}")

    # === Generalization error evaluation ===
    model.eval()
    generalization_errors = []
    with torch.no_grad():
        for file_path in generalization_files:
            b1_gen, b2_gen, trunk_gen, target_gen = load_full_f_data(file_path)
            preds_gen = model(b1_gen, b2_gen, trunk_gen)
            gen_error = compute_relative_l2_error(preds_gen, target_gen)
            generalization_errors.append(gen_error)

    avg_generalization_error = sum(generalization_errors) / len(generalization_errors)

    results = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_relative_l2_errors": test_rel_l2s,
        "avg_test_relative_l2_error": sum(test_rel_l2s) / len(test_rel_l2s),
        "final_test_relative_l2_error": test_rel_l2s[-1],
        "avg_generalization_error": avg_generalization_error,
        "N_train_rhs": config['n_train'],
        "N_test_rhs": config['n_test'],
        "N_generalization_rhs": config['n_generalization'],
        "xi": config['xi']
    }

    os.makedirs("../results/encoder", exist_ok=True)
    torch.save(model.state_dict(), f"../results/encoder/model_xi{config['xi']}_train{config['n_train']}_test{config['n_test']}.pt")
    save_results(results, f"../results/encoder/results_xi{config['xi']}_train{config['n_train']}_test{config['n_test']}.json")

    print(f"\nFinal Generalization Error: {avg_generalization_error:.4e}")

    return avg_generalization_error

# === Main ===
if __name__ == "__main__":
    xi = 4
    data_files = sorted(glob.glob(f"../data/torus_N*_xi{xi}_f*.npz"))

    # First split: 60% train, 40% temp
    train_files, temp_files = train_test_split(data_files, test_size=0.4, random_state=42)
    # Second split: 20% validation, 20% generalization
    test_files, generalization_files = train_test_split(temp_files, test_size=0.5, random_state=24)

    train_data, test_data = [], []

    for file_path in train_files:
        b1, b2, trunk, target = load_full_f_data(file_path)
        train_data.append(TensorDataset(b1, b2, trunk, target))

    for file_path in test_files:
        b1, b2, trunk, target = load_full_f_data(file_path)
        test_data.append(TensorDataset(b1, b2, trunk, target))

    train_dataset = torch.utils.data.ConcatDataset(train_data)
    test_dataset = torch.utils.data.ConcatDataset(test_data)

    sample = train_data[0][0]

    config = {
        'input_dim_branch1': sample[0].shape[0],
        'input_dim_branch2': sample[1].shape[0],
        'input_dim_trunk': sample[2].shape[0],
        'hidden_dims': [128, 128],
        'activation': 'relu',
        'trunk_activation': 'tanh',
        'spectral_norm': False,
        'output_dim': 1,
        'xi': xi,
        'lr': 0.001,
        'schedule_type': 'cosine',
        'epochs': 5000,
        'batch_size': 256,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'n_train': len(train_files),
        'n_test': len(test_files),
        'n_generalization': len(generalization_files)
    }

    train_deeponet_encoder(config, generalization_files)
