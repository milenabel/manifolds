import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csc_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve

# Global torus parameters
R, r = 1.0, 0.3

# Mesh parameters
N_list   = [25, 30, 40, 60, 80]
xi_list  = [2, 4, 6, 8]
M_random = 20

def normalize(arr):
    mean = np.mean(arr, axis=0, keepdims=True)
    std  = np.std(arr, axis=0, keepdims=True)
    return (arr - mean) / (std + 1e-8), mean, std

def compute_torus_nodes(N):
    theta = np.linspace(0, 2*np.pi, 3*N, endpoint=False)
    phi   = np.linspace(0, 2*np.pi,   N, endpoint=False)
    dθ = theta[1]-theta[0]; θ2 = theta + dθ/2
    dφ = phi[1]-phi[0]; φ2 = phi + dφ/2

    θ,  φ  = np.meshgrid(theta,  phi)
    θ2, φ2 = np.meshgrid(θ2,     φ2)
    θ = np.r_[θ.ravel(),  θ2.ravel()]
    φ = np.r_[φ.ravel(),  φ2.ravel()]

    X = np.stack([
        (R + r*np.cos(φ)) * np.cos(θ),
        (R + r*np.cos(φ)) * np.sin(θ),
        r*np.sin(φ)
    ], axis=1)

    nr = np.stack([
        r*np.cos(φ)*(R + r*np.cos(φ))*np.cos(θ),
        r*np.cos(φ)*(R + r*np.cos(φ))*np.sin(θ),
        r*np.sin(φ)*(R + r*np.cos(φ))
    ], axis=1)
    nr /= np.linalg.norm(nr, axis=1, keepdims=True)
    return X, nr, θ, φ

def u_manufactured(X):
    x,y,z = X[:,0], X[:,1], X[:,2]
    return (1/8)*( x*(x**4 - 10*x**2*y**2 + 5*y**4) * (x**2 + y**2 - 60*z**2) )

def f_manufactured(X):
    x,y,z = X[:,0], X[:,1], X[:,2]
    rxy = np.sqrt(x**2 + y**2)
    return -(3./(8*rxy**2))*x*(x**4 - 10*x**2*y**2 + 5*y**4)*(10248*rxy**4 - 34335*rxy**3 + 41359*rxy**2 - 21320*rxy + 4000)

def generate_random_forcings(theta, phi, M):
    fs = []
    for _ in range(M):
        a  = np.random.uniform(-1,1)
        fθ = np.random.randint(1,6)
        fφ = np.random.randint(1,6)
        pθ = np.random.uniform(0,2*np.pi)
        pφ = np.random.uniform(0,2*np.pi)
        fs.append( a * np.sin(fθ*theta + pθ) * np.sin(fφ*phi + pφ) )
    return fs

def build_laplacian(X, nr, l, k=None):
    if k is None:
        num_poly = (l + 1)*(l + 2) // 2
        k = num_poly + 5

    tree = cKDTree(X)
    N = X.shape[0]
    L = lil_matrix((N,N))
    monos = [(i,j) for i in range(l+1) for j in range(l+1-i)]

    def P_basis(xw):
        return np.stack([ xw[:,0]**i * xw[:,1]**j for i,j in monos ], axis=1)

    for i in range(N):
        idx = tree.query(X[i], k=k)[1]
        Xnbr = X[idx]
        n = nr[i]
        e2 = np.array([0,1,0]) if abs(n[0]) > abs(n[1]) else np.array([1,0,0])
        t1 = e2 - (n @ e2) * n
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        Rel = Xnbr - X[i]
        xw  = np.stack([Rel @ t1, Rel @ t2], axis=1)
        r   = np.linalg.norm(xw[:,None,:] - xw[None,:,:], axis=2)

        ep  = 1.0
        A   = np.exp(-(ep*r)**2)
        Lrbf= (4*ep**4*r**2 - 6*ep**2)*np.exp(-(ep*r)**2)

        P   = P_basis(xw)
        A_full = np.block([[A, P],[P.T, np.zeros((P.shape[1], P.shape[1]))]])
        rhs = np.r_[Lrbf[0], np.zeros(P.shape[1])]

        try:
            w = np.linalg.solve(A_full, rhs)
            L[i, idx] = w[:k]
        except np.linalg.LinAlgError:
            pass

    return csc_matrix(L)

def solve_poisson(X, nr, f, l):
    L = build_laplacian(X, nr, l)
    N = X.shape[0]
    z = np.ones((N,1))
    Lh = vstack([hstack([L, csc_matrix(z)]), csc_matrix(np.r_[z, [[0]]]).T]).tocsc()
    rhs_h = np.r_[f, 0]
    sol = spsolve(Lh, rhs_h)
    return sol[:-1]

if __name__ == '__main__':
    out_dir = '../data'
    os.makedirs(out_dir, exist_ok=True)

    for N in N_list:
        X, nr, θ, φ = compute_torus_nodes(N)
        f0      = f_manufactured(X)
        u_true0 = u_manufactured(X)
        X_norm, X_mean, X_std = normalize(X)
        f0_norm, f_mean, f_std = normalize(f0[:, None])

        for xi in xi_list:
            try:
                u_num0 = solve_poisson(X, nr, f0, l=xi)
                fname0 = f"torus_N{X.shape[0]}_xi{xi}_f0.npz"
                np.savez(os.path.join(out_dir, fname0),
                         X=X_norm, nr=nr, f=f0_norm, u_true=u_true0, u_num=u_num0,
                         X_mean=X_mean, X_std=X_std, f_mean=f_mean, f_std=f_std)
                print("saved", fname0)
            except Exception as e:
                print(f"[Skipped] N={X.shape[0]}, xi={xi}, f0 → {e}")
                continue

            f_list = generate_random_forcings(θ, φ, M_random)
            for j, fj in enumerate(f_list, start=1):
                try:
                    fj = fj[:, None]
                    fj_norm = (fj - f_mean) / (f_std + 1e-8)
                    uj = solve_poisson(X, nr, fj.ravel(), l=xi)
                    fname = f"torus_N{X.shape[0]}_xi{xi}_f{j}.npz"
                    np.savez(os.path.join(out_dir, fname),
                             X=X_norm, nr=nr, f=fj_norm, u_num=uj,
                             X_mean=X_mean, X_std=X_std, f_mean=f_mean, f_std=f_std)
                    print("saved", fname)
                except Exception as e:
                    print(f"[Skipped] N={X.shape[0]}, xi={xi}, f{j} → {e}")
                    continue