import torch
import torch.optim as optim
from deeponet_vanilla import DeepONet, BranchNet, TrunkNet
from utils import get_scheduler, save_results
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split

# === Helper Functions ===
def compute_relative_l2_error(predictions, targets):
    l2_error = torch.norm(predictions - targets, p=2).item()
    target_norm = torch.norm(targets, p=2).item()
    return l2_error / target_norm

def load_torus_data_full(data_file):
    data = np.load(data_file)
    f = data['f']
    X = data['X']
    u = data['u_num']

    trunk_inputs = []
    branch_inputs = []
    targets = []

    for i in range(len(X)):
        trunk_inputs.append(X[i])
        targets.append(np.array(u[i], ndmin=1))
        branch_inputs.append(f)  # Full forcing vector (no subsampling!)

    return (
        torch.tensor(np.array(branch_inputs), dtype=torch.float32),
        torch.tensor(np.array(trunk_inputs), dtype=torch.float32),
        torch.tensor(np.array(targets), dtype=torch.float32),
        len(X)
    )

# === Training Loop ===
def train_deeponet(config, generalization_files, save_every=1000):
    output_dim = config['targets'].shape[1]

    branch_net = BranchNet(
        input_dim=config['input_dim_branch'],
        hidden_dims=config['hidden_dim'],
        output_dim=output_dim,
        activation=config['branch_activation'],
        spectral_norm=config['spectral_norm'],
        use_layernorm=config['use_layernorm']
    )

    trunk_net = TrunkNet(
        input_dim=config['input_dim_trunk'],
        hidden_dims=config['hidden_dim'],
        output_dim=output_dim,
        activation=config['trunk_activation'],
        spectral_norm=config['spectral_norm'],
        use_layernorm=config['use_layernorm']
    )

    deeponet = DeepONet(branch_net, trunk_net)

    optimizer = optim.Adam(deeponet.parameters(), lr=config['lr'])
    scheduler = get_scheduler(optimizer, config['schedule_type'])

    epochs = config['epochs']
    losses, l2_errors, relative_l2_errors, learning_rates = [], [], [], []
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
        current_lr = optimizer.param_groups[0]['lr']

        losses.append(loss.item())
        l2_errors.append(l2_error)
        relative_l2_errors.append(relative_l2_error)
        learning_rates.append(current_lr)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4e}, Rel L2: {relative_l2_error:.4e}")

    # === Generalization Error Evaluation ===
    deeponet.eval()
    generalization_errors = []

    with torch.no_grad():
        for file_path in generalization_files:
            branch_input_gen, trunk_input_gen, targets_gen, _ = load_torus_data_full(file_path)
            preds_gen = deeponet(branch_input_gen, trunk_input_gen)
            gen_error = compute_relative_l2_error(preds_gen, targets_gen)
            generalization_errors.append(gen_error)

    avg_generalization_error = sum(generalization_errors) / len(generalization_errors)

    print(f"\nFinal Generalization Error (Vanilla): {avg_generalization_error:.4e}")

    results = {
        "losses": losses,
        "l2_errors": l2_errors,
        "relative_l2_errors": relative_l2_errors,
        "avg_relative_l2_error": sum(relative_l2_errors) / len(relative_l2_errors),
        "avg_generalization_error": avg_generalization_error,
        "num_layers": num_layers,
        "num_neurons": num_neurons,
        "N": config['N'],
        "xi": config['xi'],
        "learning_rates": learning_rates
    }

    os.makedirs("../results/vanilla", exist_ok=True)
    result_path = f"../results/vanilla/deeponet_results_torus_N{config['N']}_xi{config['xi']}.json"
    save_results(results, result_path)
    torch.save(deeponet.state_dict(),
               f"../results/vanilla/model_N{config['N']}_xi{config['xi']}.pt")

# === Main Driver ===
if __name__ == "__main__":
    xi = 4
    data_files = sorted(glob.glob(f"../data/torus_N*_xi{xi}_f0.npz"))

    # First split: 60% train, 40% temp
    train_files, temp_files = train_test_split(data_files, test_size=0.4, random_state=42)
    # Second split: 20% test, 20% generalization
    test_files, generalization_files = train_test_split(temp_files, test_size=0.5, random_state=24)

    for file_path in train_files:
        base = os.path.basename(file_path)
        N = int(base.split("_")[1][1:])

        print(f"\n--- Training Vanilla DeepONet for N={N}, xi={xi} ---")
        branch_input, trunk_input, targets, N_actual = load_torus_data_full(file_path)

        config = {
            'input_dim_branch': branch_input.shape[1],
            'input_dim_trunk': trunk_input.shape[1],
            'hidden_dim': [128, 128],
            'branch_activation': 'relu',
            'trunk_activation': 'tanh',
            'spectral_norm': False,
            'use_layernorm': False,
            'xi': xi,
            'N': N_actual,
            'lr': 0.001,
            'schedule_type': 'cosine',
            'epochs': 10000,
            'branch_input': branch_input,
            'trunk_input': trunk_input,
            'targets': targets
        }

        train_deeponet(config, generalization_files)
