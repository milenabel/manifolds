import torch
import torch.optim as optim
from deeponet_vanilla import DeepONet, BranchNet, TrunkNet
from encoder import SimpleEncoder
from utils import get_scheduler, save_results
import numpy as np
import glob
import os

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Helper Functions
def compute_relative_l2_error(predictions, targets):
    l2_error = torch.norm(predictions - targets, p=2).item()
    target_norm = torch.norm(targets, p=2).item()
    return l2_error / target_norm

def load_torus_data_split(file_path, max_points=6000, generalization_fraction=0.02, test_fraction=0.2):
    data = np.load(file_path)
    f = data['f']
    X = data['X']
    u = data['u_num']

    N_total = len(X)
    if N_total > max_points:
        idx_selected = np.random.choice(N_total, max_points, replace=False)
        X = X[idx_selected]
        u = u[idx_selected]

    forcing_vector = f.flatten()
    branch_inputs = torch.tensor(np.array([forcing_vector for _ in range(len(X))]), dtype=torch.float32)
    trunk_inputs = torch.tensor(X, dtype=torch.float32)
    targets = torch.tensor(u[:, None], dtype=torch.float32)

    N_pts = len(X)
    N_gen = max(1, int(generalization_fraction * N_pts))
    idx = np.random.permutation(N_pts)

    idx_gen = idx[:N_gen]
    idx_remain = idx[N_gen:]

    branch_gen = branch_inputs[idx_gen]
    trunk_gen = trunk_inputs[idx_gen]
    targets_gen = targets[idx_gen]

    N_remain = len(idx_remain)
    N_test = int(test_fraction * N_remain)
    idx_test = idx_remain[:N_test]
    idx_train = idx_remain[N_test:]

    branch_train = branch_inputs[idx_train]
    trunk_train = trunk_inputs[idx_train]
    targets_train = targets[idx_train]

    branch_test = branch_inputs[idx_test]
    trunk_test = trunk_inputs[idx_test]
    targets_test = targets[idx_test]

    return (branch_train, trunk_train, targets_train,
            branch_test, trunk_test, targets_test,
            branch_gen, trunk_gen, targets_gen)

# Training Loop 
def train_deeponet(config, save_every=1000):
    output_dim = 128

    encoder_f = SimpleEncoder(config['input_dim_branch'], latent_dim=128).to(device)

    branch_net = BranchNet(128, [128, 128], 128, activation='relu', spectral_norm=False, use_layernorm=False)
    trunk_net = TrunkNet(config['input_dim_trunk'], [128, 128], 128, activation='tanh', spectral_norm=False, use_layernorm=False)

    deeponet = DeepONet(branch_net, trunk_net).to(device)

    with torch.no_grad():
        encoded_f = encoder_f(config['branch_input'][0:1, :].to(device)).squeeze(0)

    optimizer = optim.Adam(deeponet.parameters(), lr=config['lr'])
    scheduler = get_scheduler(optimizer, config['schedule_type'])

    branch_input = config['branch_input'].to(device)
    trunk_input = config['trunk_input'].to(device)
    targets = config['targets'].to(device)

    trunk_input_test = config['trunk_input_test'].to(device)
    targets_test = config['targets_test'].to(device)

    trunk_input_gen = config['trunk_input_gen'].to(device)
    targets_gen = config['targets_gen'].to(device)

    train_dataset = torch.utils.data.TensorDataset(branch_input, trunk_input, targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    epochs = config['epochs']
    losses, train_errors, test_errors, learning_rates = [], [], [], []

    for epoch in range(epochs):
        deeponet.train()
        running_loss = 0.0

        for _, x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = deeponet(encoded_f, x_batch)
            loss = torch.nn.MSELoss()(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_batch)

        scheduler.step()
        avg_loss = running_loss / len(train_dataset)

        deeponet.eval()
        with torch.no_grad():
            preds_train = deeponet(encoded_f, trunk_input)
            preds_test = deeponet(encoded_f, trunk_input_test)
            train_error = compute_relative_l2_error(preds_train, targets)
            test_error = compute_relative_l2_error(preds_test, targets_test)
            current_lr = optimizer.param_groups[0]['lr']

        losses.append(avg_loss)
        train_errors.append(train_error)
        test_errors.append(test_error)
        learning_rates.append(current_lr)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Rel L2: {train_error:.4e}, Test Error: {test_error:.4e}")

    # Generalization Error after training ends
    deeponet.eval()
    with torch.no_grad():
        preds_gen = deeponet(encoded_f, trunk_input_gen)
        gen_error = compute_relative_l2_error(preds_gen, targets_gen)

    print(f"\nFinal Generalization Error (Vanilla): {gen_error:.4e}")

    results = {
        "losses": losses,
        "train_errors": train_errors,
        "test_errors": test_errors,
        "generalization_error": gen_error,
        "avg_train_error": sum(train_errors) / len(train_errors),
        "avg_test_error": sum(test_errors) / len(test_errors),
        "N": config['N'],
        "xi": config['xi'],
        "learning_rates": learning_rates
    }

    os.makedirs("../results/vanilla", exist_ok=True)
    result_path = f"../results/vanilla/deeponet_results_torus_N{config['N']}_xi{config['xi']}.json"
    save_results(results, result_path)
    torch.save(deeponet.state_dict(), f"../results/vanilla/model_N{config['N']}_xi{config['xi']}.pt")

# Main Driver
if __name__ == "__main__":
    xi = 4
    data_files = sorted(glob.glob(f"../data/torus_N*_xi{xi}_f0.npz"))

    for file_path in data_files:
        base = os.path.basename(file_path)
        N = int(base.split("_")[1][1:])

        print(f"\n--- Training Vanilla DeepONet for N={N}, xi={xi} ---")
        (branch_train, trunk_train, targets_train,
         branch_test, trunk_test, targets_test,
         branch_gen, trunk_gen, targets_gen) = load_torus_data_split(file_path)

        config = {
            'input_dim_branch': branch_train.shape[1],
            'input_dim_trunk': trunk_train.shape[1],
            'xi': xi,
            'N': N,  # <- use the ORIGINAL N here
            'lr': 0.001,
            'schedule_type': 'cosine',
            'epochs': 10000,
            'batch_size': 512,
            'branch_input': branch_train,
            'trunk_input': trunk_train,
            'targets': targets_train,
            'branch_input_test': branch_test,
            'trunk_input_test': trunk_test,
            'targets_test': targets_test,
            'branch_input_gen': branch_gen,
            'trunk_input_gen': trunk_gen,
            'targets_gen': targets_gen
        }

        train_deeponet(config)
