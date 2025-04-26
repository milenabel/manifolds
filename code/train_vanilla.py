import torch
import torch.optim as optim
from deeponet_vanilla import DeepONet, BranchNet, TrunkNet
from utils import get_scheduler, save_results
import numpy as np
import glob
import os

def compute_relative_l2_error(predictions, targets):
    l2_error = torch.norm(predictions - targets, p=2).item()
    target_norm = torch.norm(targets, p=2).item()
    return l2_error / target_norm

def load_torus_data(data_file, sensor_path):
    data = np.load(data_file)
    f = data['f']
    X = data['X']
    u = data['u_num']
    sensor_idx = np.load(sensor_path)

    trunk_inputs = []
    branch_inputs = []
    targets = []

    for i in range(len(X)):
        trunk_inputs.append(X[i])
        targets.append(np.array(u[i], ndmin=1))
        branch_inputs.append(f[sensor_idx])

    return (
        torch.tensor(np.array(branch_inputs), dtype=torch.float32),
        torch.tensor(np.array(trunk_inputs), dtype=torch.float32),
        torch.tensor(np.array(targets), dtype=torch.float32),
        len(X)
    )

def train_deeponet(config, save_every=1000):
    output_dim = config['targets'].shape[1]

    branch_net = BranchNet(config['input_dim_branch'], config['hidden_dim'], output_dim,
                           config['activation'], config['spectral_norm'])
    trunk_net = TrunkNet(config['input_dim_trunk'], config['hidden_dim'], output_dim,
                         config['activation'], config['spectral_norm'])
    deeponet = DeepONet(branch_net, trunk_net)

    optimizer = optim.Adam(deeponet.parameters(), lr=config['lr'])
    scheduler = get_scheduler(optimizer, config['schedule_type'])

    epochs = config['epochs']
    losses, l2_errors, relative_l2_errors = [], [], []
    num_layers = len(config['hidden_dim'])
    num_neurons = sum(config['hidden_dim'])

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = deeponet(config['branch_input'], config['trunk_input'])
        loss = torch.nn.MSELoss()(predictions, config['targets'])
        loss.backward()
        optimizer.step()
        scheduler.step()

        l2_error = torch.norm(predictions - config['targets'], dim=1).mean().item()
        relative_l2_error = compute_relative_l2_error(predictions, config['targets'])

        losses.append(loss.item())
        l2_errors.append(l2_error)
        relative_l2_errors.append(relative_l2_error)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4e}, Rel L2: {relative_l2_error:.4e}")

        if epoch % save_every == 0 or epoch == epochs - 1:
            results = {
                "losses": losses,
                "l2_errors": l2_errors,
                "relative_l2_errors": relative_l2_errors,
                "avg_relative_l2_error": sum(relative_l2_errors) / len(relative_l2_errors),
                "num_layers": num_layers,
                "num_neurons": num_neurons,
                "N": config['N'],
                "xi": config['xi'],
                "n_sensors": config['n_sensors']
            }
            result_path = f"../results/vanilla/deeponet_results_torus_N{config['N']}_xi{config['xi']}_sensors{config['n_sensors']}.json"
            save_results(results, result_path)
            torch.save(deeponet.state_dict(),
                       f"../results/vanilla/model_N{config['N']}_xi{config['xi']}_sensors{config['n_sensors']}.pt")

if __name__ == "__main__":
    xi = 4
    sensor_counts = [100, 200, 300]
    data_files = sorted(glob.glob(f"../data/torus_N*_xi{xi}_f0.npz"))

    for file_path in data_files:
        base = os.path.basename(file_path)
        N = int(base.split("_")[1][1:])

        for n_sensors in sensor_counts:
            sensor_path = f"../data/sensor_indices_{N}_xi{xi}_sensors{n_sensors}.npy"
            if not os.path.exists(sensor_path):
                print(f"Skipping N={N}, sensors={n_sensors} (missing sensor indices)")
                continue

            print(f"\n--- Training DeepONet for N={N}, xi={xi}, sensors={n_sensors} ---")
            branch_input, trunk_input, targets, N_actual = load_torus_data(file_path, sensor_path)

            config = {
                'input_dim_branch': branch_input.shape[1],
                'input_dim_trunk': trunk_input.shape[1],
                'hidden_dim': [128, 128],
                'activation': 'relu',
                'spectral_norm': False,
                'xi': xi,
                'N': N_actual,
                'n_sensors': n_sensors,
                'lr': 0.001,
                'schedule_type': 'cosine',
                'epochs': 10000,
                'branch_input': branch_input,
                'trunk_input': trunk_input,
                'targets': targets
            }

            train_deeponet(config)
