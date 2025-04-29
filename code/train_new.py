import torch
import torch.optim as optim
from deeponet_new import DeepONetDualBranch, BranchNet, TrunkNet
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
    nr = data['nr']  # normal vectors
    u = data['u_num']

    N_total = len(X)
    if N_total > max_points:
        idx_selected = np.random.choice(N_total, max_points, replace=False)
        X = X[idx_selected]
        nr = nr[idx_selected]
        u = u[idx_selected]

    branch_f_inputs = torch.tensor(np.array([f.flatten() for _ in range(len(X))]), dtype=torch.float32)
    branch_n_inputs = torch.tensor(nr, dtype=torch.float32)
    trunk_inputs = torch.tensor(X, dtype=torch.float32)
    targets = torch.tensor(u[:, None], dtype=torch.float32)

    N_pts = len(X)
    N_gen = max(1, int(generalization_fraction * N_pts))
    idx = np.random.permutation(N_pts)

    idx_gen = idx[:N_gen]
    idx_remain = idx[N_gen:]

    idx_test = idx_remain[:int(test_fraction * len(idx_remain))]
    idx_train = idx_remain[int(test_fraction * len(idx_remain)) :]

    def slice_data(idxs):
        return (branch_f_inputs[idxs], branch_n_inputs[idxs], trunk_inputs[idxs], targets[idxs])

    return slice_data(idx_train) + slice_data(idx_test) + slice_data(idx_gen)

# Training Loop
def train_deeponet(config, save_every=1000):
    branch_f_net = BranchNet(
        input_dim=config['input_dim_branch_f'],
        hidden_dims=[128, 128],
        output_dim=128
    )

    branch_n_net = BranchNet(
        input_dim=config['input_dim_branch_n'],
        hidden_dims=[128, 128],
        output_dim=128
    )

    trunk_net = TrunkNet(
        input_dim=config['input_dim_trunk'],
        hidden_dims=[128, 128],
        output_dim=128
    )

    deeponet = DeepONetDualBranch(branch_f_net, branch_n_net, trunk_net).to(device)

    optimizer = optim.Adam(deeponet.parameters(), lr=config['lr'])
    scheduler = get_scheduler(optimizer, config['schedule_type'])

    branch_f_train = config['branch_f_train'].to(device)
    branch_n_train = config['branch_n_train'].to(device)
    trunk_train = config['trunk_train'].to(device)
    targets_train = config['targets_train'].to(device)

    branch_f_test = config['branch_f_test'].to(device)
    branch_n_test = config['branch_n_test'].to(device)
    trunk_test = config['trunk_test'].to(device)
    targets_test = config['targets_test'].to(device)

    branch_f_gen = config['branch_f_gen'].to(device)
    branch_n_gen = config['branch_n_gen'].to(device)
    trunk_gen = config['trunk_gen'].to(device)
    targets_gen = config['targets_gen'].to(device)

    train_dataset = torch.utils.data.TensorDataset(branch_f_train, branch_n_train, trunk_train, targets_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    epochs = config['epochs']
    losses, train_errors, test_errors, learning_rates = [], [], [], []

    for epoch in range(epochs):
        deeponet.train()
        running_loss = 0.0

        for b1_batch, b2_batch, x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = deeponet(b1_batch, b2_batch, x_batch)
            loss = torch.nn.MSELoss()(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_batch)

        scheduler.step()
        avg_loss = running_loss / len(train_dataset)

        deeponet.eval()
        with torch.no_grad():
            preds_train = deeponet(branch_f_train, branch_n_train, trunk_train)
            preds_test = deeponet(branch_f_test, branch_n_test, trunk_test)

            train_error = compute_relative_l2_error(preds_train, targets_train)
            test_error = compute_relative_l2_error(preds_test, targets_test)

            current_lr = optimizer.param_groups[0]['lr']

        losses.append(avg_loss)
        train_errors.append(train_error)
        test_errors.append(test_error)
        learning_rates.append(current_lr)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Rel L2: {train_error:.4e}, Test Rel L2: {test_error:.4e}")

    # Generalization Error After Full Training
    deeponet.eval()
    with torch.no_grad():
        preds_gen = deeponet(branch_f_gen, branch_n_gen, trunk_gen)
        gen_error = compute_relative_l2_error(preds_gen, targets_gen)

    print(f"\nFinal Generalization Error (New DeepONet): {gen_error:.4e}")

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

    os.makedirs("../results/new", exist_ok=True)
    result_path = f"../results/new/deeponet_results_torus_N{config['N']}_xi{config['xi']}.json"
    save_results(results, result_path)
    torch.save(deeponet.state_dict(), f"../results/new/model_N{config['N']}_xi{config['xi']}.pt")

# Main Driver
if __name__ == "__main__":
    xi = 4
    data_files = sorted(glob.glob(f"../data/torus_N*_xi{xi}_f0.npz"))

    for file_path in data_files:
        base = os.path.basename(file_path)
        N = int(base.split("_")[1][1:])  

        print(f"\n--- Training New DeepONet for N={N}, xi={xi} ---")
        (branch_f_train, branch_n_train, trunk_train, targets_train,
         branch_f_test, branch_n_test, trunk_test, targets_test,
         branch_f_gen, branch_n_gen, trunk_gen, targets_gen) = load_torus_data_split(file_path)

        config = {
            'input_dim_branch_f': branch_f_train.shape[1],
            'input_dim_branch_n': branch_n_train.shape[1],
            'input_dim_trunk': trunk_train.shape[1],
            'xi': xi,
            'N': N,   
            'lr': 0.001,
            'schedule_type': 'cosine',
            'epochs': 10000,
            'batch_size': 512,
            'branch_f_train': branch_f_train,
            'branch_n_train': branch_n_train,
            'trunk_train': trunk_train,
            'targets_train': targets_train,
            'branch_f_test': branch_f_test,
            'branch_n_test': branch_n_test,
            'trunk_test': trunk_test,
            'targets_test': targets_test,
            'branch_f_gen': branch_f_gen,
            'branch_n_gen': branch_n_gen,
            'trunk_gen': trunk_gen,
            'targets_gen': targets_gen
        }

        train_deeponet(config)
