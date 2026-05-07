"""J. Fuentes Aguilar, 2025-2026."""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import linear_sum_assignment

def torus_parametrisation(u_param, v_param, R_param=1.0, r_param=1.0):
    X = (R_param + r_param * np.cos(v_param)) * np.cos(u_param)
    Y = (R_param + r_param * np.cos(v_param)) * np.sin(u_param)
    Z = r_param * np.sin(v_param)
    return X, Y, Z

def torus_tangent(u_param, v_param, R_param=1.0, r_param=1.0):

    du = np.stack([
        -(R_param + r_param * np.cos(v_param)) * np.sin(u_param),
         (R_param + r_param * np.cos(v_param)) * np.cos(u_param),
         np.zeros_like(u_param)
    ], axis=-1)

    dv = np.stack([
        -r_param * np.sin(v_param) * np.cos(u_param),
        -r_param * np.sin(v_param) * np.sin(u_param),
         r_param * np.cos(v_param)
    ], axis=-1)

    norm_du = np.linalg.norm(du, axis=-1, keepdims=True)
    norm_du = np.maximum(norm_du, 1e-12)
    e1 = du / norm_du

    dv_proj = dv - np.sum(dv * e1, axis=-1, keepdims=True) * e1
    norm_dv = np.linalg.norm(dv_proj, axis=-1, keepdims=True)
    norm_dv = np.maximum(norm_dv, 1e-12)
    e2 = dv_proj / norm_dv

    return np.stack([e1, e2], axis=-2)

def ground_truth_field(u, v):
    return np.sin(2*u + 4*np.sin(v)) + 0.5*np.cos(8*v - u)

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_ensemble(u_samples, v_samples, targets, R_param=1.0, r_param=1.0,
                   n_models=10, batch_size=64, learning_rate=1e-2, epochs=200,
                   seed0=0, weight_decay=0.0):

    x_samples, y_samples, z_samples = torus_parametrisation(
        u_samples, v_samples, R_param=R_param, r_param=r_param
    )

    X_data = torch.tensor(
        np.vstack([x_samples, y_samples, z_samples]).T,
        dtype=torch.float32
    )
    y_data = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_data, y_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    models = []
    for seed in range(n_models):
        torch.manual_seed(seed0 + seed)
        model = SimpleMLP()
        optimiser = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            for xb, yb in dataloader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        models.append(model)

    return models

def evaluate_ensemble(models, u_eval, v_eval, R_param=1.0, r_param=1.0):
    u_flat = u_eval.ravel()
    v_flat = v_eval.ravel()

    x_eval, y_eval, z_eval = torus_parametrisation(
        u_flat, v_flat, R_param=R_param, r_param=r_param
    )

    X_eval = torch.tensor(
        np.vstack([x_eval, y_eval, z_eval]).T,
        dtype=torch.float32
    )

    E_all = torus_tangent(u_flat, v_flat, R_param=R_param, r_param=r_param)

    predictions = []
    jacobians = []

    for model in models:
        model.eval()
        Xi = X_eval.detach().clone().requires_grad_(True)
        y_pred = model(Xi)

        grad = torch.autograd.grad(
            outputs=y_pred,
            inputs=Xi,
            grad_outputs=torch.ones_like(y_pred),
            retain_graph=False,
            create_graph=False
        )[0]

        y_np = y_pred.detach().cpu().numpy().squeeze()
        g_np = grad.detach().cpu().numpy()

        e1 = E_all[:, 0, :]
        e2 = E_all[:, 1, :]
        g1 = np.sum(g_np * e1, axis=1)
        g2 = np.sum(g_np * e2, axis=1)
        g_tan = np.stack([g1, g2], axis=1)

        predictions.append(y_np)
        jacobians.append(g_tan)

    return np.stack(predictions), np.stack(jacobians)

def rotation_SO2_closest(R):
    U, _, Vt = np.linalg.svd(R)
    Rso = U @ Vt
    if np.linalg.det(Rso) < 0:
        U[:, 1] *= -1
        Rso = U @ Vt
    return Rso

def rotation_AB(i_A, j_A, i_B, j_B, E_all, n_eval):
    k_A = i_A * n_eval + j_A
    k_B = i_B * n_eval + j_B
    E_A = E_all[k_A]
    E_B = E_all[k_B]
    R = E_A @ E_B.T
    return rotation_SO2_closest(R)

def jet_costm(fibre_A, fibre_B, s_y, s_g, R_AB):
    G_B_rot = fibre_B[:, 1:3] @ R_AB.T
    dY = (fibre_A[:, None, 0] - fibre_B[None, :, 0]) / s_y
    dG = (fibre_A[:, None, 1:3] - G_B_rot[None, :, :]) / s_g
    C = dY**2 + np.sum(dG**2, axis=2)
    return C

def logsumexp(A, axis=None, keepdims=False):
    m = np.max(A, axis=axis, keepdims=True)
    S = np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True)) + m
    if keepdims:
        return S
    return np.squeeze(S, axis=axis)

def sinkhorn_uniform(C, eps=0.5, max_iter=1000, tol=1e-9):
    n = C.shape[0]
    eps = max(float(eps), 1e-8)

    Cc = C - np.min(C)
    log_a = -np.log(n)
    log_b = -np.log(n)

    f = np.zeros(n)
    g = np.zeros(n)

    for _ in range(max_iter):
        f_prev, g_prev = f.copy(), g.copy()

        S = (f[:, None] + g[None, :] - Cc) / eps
        lse_rows = logsumexp(S, axis=1)
        f = f + eps * (log_a - lse_rows)

        S = (f[:, None] + g[None, :] - Cc) / eps
        lse_cols = logsumexp(S, axis=0)
        g = g + eps * (log_b - lse_cols)

        if max(np.max(np.abs(f - f_prev)), np.max(np.abs(g - g_prev))) < tol:
            break

    P_log = (f[:, None] + g[None, :] - Cc) / eps
    P = np.exp(P_log)
    P = np.maximum(P, 0)
    return P

def transfer_operator_entropic(C, eps=0.5):
    n = C.shape[0]
    P = sinkhorn_uniform(C, eps=eps)
    T = n * P
    return T

def transfer_operator_hungarian(C):
    n = C.shape[0]
    row_ind, col_ind = linear_sum_assignment(C)
    P = np.zeros((n, n))
    P[row_ind, col_ind] = 1.0
    T = P
    return T

def fibre_jets(i, j, predictions, jacobians, n_eval):
    k = i * n_eval + j
    return np.hstack([
        predictions[:, k:k+1],
        jacobians[:, k, :]
    ])

def loop_holonomy(i0, j0, side, predictions, jacobians, E_all, n_eval,
                  s_y, s_g, eps=0.5, mode='entropic'):
    loop_indices = [
        (i0, j0),
        (i0, j0 + side),
        (i0 + side, j0 + side),
        (i0 + side, j0),
        (i0, j0)
    ]

    Ts = []
    for (ia, ja), (ib, jb) in zip(loop_indices[:-1], loop_indices[1:]):
        A = fibre_jets(ia, ja, predictions, jacobians, n_eval)
        B = fibre_jets(ib, jb, predictions, jacobians, n_eval)
        R_AB = rotation_AB(ia, ja, ib, jb, E_all, n_eval)
        C = jet_costm(A, B, s_y, s_g, R_AB)

        if mode == 'entropic':
            T = transfer_operator_entropic(C, eps=eps)
        elif mode == 'hungarian':
            T = transfer_operator_hungarian(C)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        Ts.append(T)

    H = Ts[-1].T @ Ts[-2].T @ Ts[-3].T @ Ts[-4].T
    I = np.eye(H.shape[0])
    h_fro = np.linalg.norm(H - I, ord='fro') / H.shape[0]

    Z0 = fibre_jets(i0, j0, predictions, jacobians, n_eval)
    Z_loop = H @ Z0

    DY = (Z_loop[:, 0] - Z0[:, 0]) / s_y
    DG1 = (Z_loop[:, 1] - Z0[:, 1]) / s_g
    DG2 = (Z_loop[:, 2] - Z0[:, 2]) / s_g
    per_model_disp = np.sqrt(DY**2 + DG1**2 + DG2**2)
    disp = per_model_disp.mean()

    return h_fro, disp

def holonomy_field(predictions, jacobians, E_all, n_eval, s_y, s_g,
                   eps=0.5, side=1, mode='entropic'):
    hol_map = np.zeros((n_eval - side, n_eval - side))
    for i in range(n_eval - side):
        for j in range(n_eval - side):
            hol_map[i, j], _ = loop_holonomy(
                i, j, side, predictions, jacobians, E_all, n_eval,
                s_y, s_g, eps=eps, mode=mode
            )
    return hol_map

def permutation_distance_entropy(T):

    eps_safe = 1e-12
    entropies = []
    for row in T:
        p = np.maximum(row / row.sum(), eps_safe)
        h = -np.sum(p * np.log(p))
        entropies.append(h)
    return np.mean(entropies)

def permutation_distance_frobenius(T, T_hungarian):
    return np.linalg.norm(T - T_hungarian, ord='fro')

def randomly_permute_ensemble(predictions, jacobians, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    n_models, n_points = predictions.shape
    preds_perm = predictions.copy()
    jacs_perm = jacobians.copy()

    for k in range(n_points):
        perm = rng.permutation(n_models)
        preds_perm[:, k] = predictions[perm, k]
        jacs_perm[:, k, :] = jacobians[perm, k, :]

    return preds_perm, jacs_perm

def plot_style_setup():
    FC = "#777777"
    FS = 16
    return FC, FS

def style_3d_axes(ax, FC="#777777", FS=16):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.9, 0.9, 0.9, 0.5))
        axis._axinfo["grid"]["linewidth"] = 1
    ax.grid(True)

    ax.set_xlabel("x", color=FC, fontsize=FS)
    ax.set_ylabel("y", color=FC, fontsize=FS)
    ax.set_zlabel("z", color=FC, fontsize=FS)
    ax.tick_params(axis="both", which="major", colors=FC, labelsize=FS-1)

    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=3))

    ax.set_box_aspect((1, 1, 0.8))

def style_colorbar(cb, label, FC="#777777", FS=16):
    cb.set_label(label, color=FC, fontsize=FS)
    cb.ax.yaxis.set_tick_params(color=FC, labelsize=FS-1)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=FC)
