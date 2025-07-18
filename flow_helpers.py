from flow_imports import *

def sample_x0(sample_size, dim):
    """
    Generate a batch of N i.i.d. samples from a standard normal distribution using NumPy.
    
    Args:
        N (int): Batch size.
        dim (int): Dimension of each sample (default is 1 for scalar values).
        
    Returns:
        np.ndarray: A (N, dim) array of standard normal samples.
    """
    return torch.randn(sample_size, dim)


def sample_x1(sample_size, device='cpu'):
    """
    Sample N i.i.d. uniform values on [0,1] and map them to points on the unit circle in 2D.
    
    Args:
        N (int): Number of samples.
        device (str): Device to place the tensor on ('cpu' or 'cuda').
        
    Returns:
        torch.Tensor: Tensor of shape (N, 2) with points on the unit circle.
    """
    u = torch.rand(sample_size, device=device)             # Uniform samples in [0,1]
    angles = 2 * torch.pi * u                     # Map to [0, 2π]
    x = torch.cos(angles)
    y = torch.sin(angles)
    return torch.stack([x, y], dim=1)



def construct_training_data(x0, x1, j, device='cpu'):
    """
    Create flow matching training dataset from paired 2D points and interpolation times.
    
    Args:
        x0 (torch.Tensor): (N, 2) Gaussian samples.
        x1 (torch.Tensor): (N, 4) Unit circle samples with measurements h-stacked.
        j (int): Number of interpolation times.
        device (str): Device to place tensors.
        
    Returns:
        torch.Tensor: (N*j, 7) tensor with rows [x0(2), x1(2), t, x_t(2)].
    """
    N = x0.shape[0]
    assert x0.shape[1] == 2 and x1.shape[1] == 4, "x0 and x1 must be 2D."
    
    # # t = torch.linspace(0, 1, steps=j, device=device)       # (j,)
    # t = torch.rand(j,)
    # t_exp = t.unsqueeze(0).expand(N, j)                    # (N, j)
    
    # x0_exp = x0.unsqueeze(1).expand(N, j, 2)               # (N, j, 2)
    # x1_exp = x1.unsqueeze(1).expand(N, j, 4)               # (N, j, 4)

    t = torch.rand(j, device=device)                          # (j,)
    t_exp = t.repeat(N, 1)                                    # (N, j)

    x0_exp = x0.unsqueeze(1).repeat(1, j, 1)                   # (N, j, 2)
    x1_exp = x1.unsqueeze(1).repeat(1, j, 1)                   # (N, j, 4)

    x_t = (1 - t_exp.unsqueeze(2)) * x0_exp + t_exp.unsqueeze(2) * x1_exp[:, :, :2]   # (N, j, 2)
    
    # Flatten all to (N*j, feature)
    x0_flat = x0_exp.reshape(N*j, 2)
    assert x1_exp.shape == (N, j, 4)
    x1_flat = x1_exp.reshape(-1, 4)  # Let PyTorch infer first dim

    # x1_flat = x1_exp.reshape(N*j, 4)
    t_flat = t_exp.reshape(N*j, 1)
    x_t_flat = x_t.reshape(N*j, 2)
    
    dataset = torch.cat([x0_flat, x1_flat, t_flat, x_t_flat], dim=1)  # (N*j, 7)
    return dataset


class FlowMatchingDataset(TensorDataset):
    def __init__(self, data_tensor):
        """
        Args:
            data_tensor (torch.Tensor): shape (num_samples, feature_dim)
                Each row is [x0(2), x1(2), t(1), x_t(2)] as per your dataset.
        """
        self.data = data_tensor

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return the sample at index idx as a dictionary or tuple
        # Here we return a dict with separate fields for convenience
        sample = self.data[idx]
        x0 = sample[0:2]
        x1 = sample[2:4]
        t = sample[4]
        xt = sample[5:7]
        return {
            "x0": x0,
            "x1": x1,
            "t": t,
            "xt": xt
        }


# class FlowMLP(torch.nn.Module):
#     def __init__(self, hidden_dim=128):
#         super().__init__()
#         # self.net = nn.Sequential(
#         #     nn.Linear(3, hidden_dim),  # 2 for x_t + 1 for t
#         #     nn.ReLU(),
#         #     nn.Linear(hidden_dim, hidden_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(hidden_dim, 2)   # output: vector field in 2D
#         # )


#         self.input_width = int(5)
#         self.output_width = 2
#         self.width = int(4*self.input_width)

#         self.net = nn.Sequential(nn.Linear(self.input_width,self.width), nn.SiLU(),
#                                     nn.Linear(self.width,4*self.width), nn.SiLU(),
#                                     nn.Linear(4*self.width,self.width), nn.SiLU(),
#                                     nn.Linear(self.width,self.output_width)).to(device='cpu')

#     def forward(self, xt, y1, t):
#         # xt: (batch_size, 2)
#         # t: (batch_size,) scalar time, expand to (batch_size, 1)
#         t = t.unsqueeze(-1)
#         inp = torch.cat([xt, y1, t], dim=-1)  # (batch_size, 3)
#         return self.net(inp)               # (batch_size, 2)
    


# def generate_grid_with_time(t, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), steps=20, device='cpu'):
#     """
#     Generate a uniform 2D grid and append a scalar time value to each point.

#     Args:
#         t (float): Scalar time value to append to each (x, y) position.
#         x_range (tuple): (min_x, max_x)
#         y_range (tuple): (min_y, max_y)
#         steps (int): Number of grid steps per axis.
#         device (str): PyTorch device.

#     Returns:
#         input_tensor (torch.Tensor): (steps*steps, 3), where each row is (x, y, t)
#         grid_X, grid_Y: meshgrid arrays for visualization
#     """
#     x = torch.linspace(x_range[0], x_range[1], steps)
#     y = torch.linspace(y_range[0], y_range[1], steps)
#     grid_X, grid_Y = torch.meshgrid(x, y, indexing='ij')  # shape: (steps, steps)

#     xy_points = torch.stack([grid_X, grid_Y], dim=-1).reshape(-1, 2)  # (N, 2)
#     t_column = torch.full((xy_points.shape[0], 1), t, device=device)  # (N, 1)
#     input_tensor = torch.cat([xy_points, t_column], dim=1).to(device)  # (N, 3)

#     return input_tensor, grid_X, grid_Y

import torch
import torch.nn as nn

class FlowMLP(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.input_width = 5   # [xt(2), y1(2), t(1)]
        self.output_width = 2  # Output vector field (2D)
        self.width = 4 * self.input_width

        self.net = nn.Sequential(
            nn.Linear(self.input_width, self.width), nn.SiLU(),
            nn.Linear(self.width, 10 * self.width), nn.SiLU(),
            nn.Linear(10 * self.width, self.width), nn.SiLU(),
            nn.Linear(self.width, self.output_width)
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 9)
           Format: [x0(2), x1(2), y1(2), t(1), xt(2)]
        """
        # Parse inputs
        size_x = x.shape[0]
        try:
            x0 = x[:, :2]
            x1 = x[:, 2:4]
            y1 = x[:, 4:6]
            t = x[:, 6].unsqueeze(-1)  # (batch_size, 1)
            xt = x[:, 7:]              # (batch_size, 2)
            eval_mode = False
        except:
            xt = x[:,:2]
            y1 = x[:,2:4]
            t = x[:,4].reshape(size_x,1)
            eval_mode = True

        # Prepare input to the network
        inp = torch.cat([xt, y1, t], dim=-1)  # (batch_size, 5)

        # Compute predicted vector field
        v_pred = self.net(inp)

        if not eval_mode:
            return v_pred, x1 - x0  # Also return v_target for convenience
        else:
            return v_pred



def generate_grid_with_time(t, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), steps=20, measurement=None, device='cpu'):
    """
    Generate a uniform 2D grid and append fixed measurement and scalar time to each point.

    Args:
        t (float): Scalar time value to append to each position.
        x_range (tuple): (min_x, max_x)
        y_range (tuple): (min_y, max_y)
        steps (int): Number of grid steps per axis.
        measurement (torch.Tensor or None): Tensor of shape (2,), fixed per point.
        device (str): PyTorch device.

    Returns:
        input_tensor (torch.Tensor): (steps*steps, 5), each row [x, y, measurement_1, measurement_2, t]
        grid_X, grid_Y: meshgrid arrays for visualization
    """
    x = torch.linspace(x_range[0], x_range[1], steps, device=device)
    y = torch.linspace(y_range[0], y_range[1], steps, device=device)
    grid_X, grid_Y = torch.meshgrid(x, y, indexing='ij')  # (steps, steps)

    xy_points = torch.stack([grid_X, grid_Y], dim=-1).reshape(-1, 2)  # (N, 2)
    
    if measurement is None:
        raise ValueError("measurement tensor of shape (2,) must be provided")
    measurement = measurement.to(device)
    assert measurement.shape == (2,), "measurement must be shape (2,)"

    measurement_rep = measurement.unsqueeze(0).expand(xy_points.shape[0], -1)  # (N, 2)
    t_column = torch.full((xy_points.shape[0], 1), t, device=device)           # (N, 1)

    input_tensor = torch.cat([xy_points, measurement_rep, t_column], dim=1)   # (N, 5)

    return input_tensor, grid_X, grid_Y


def repeated_linspace(start=0.0, end=1.0, steps=10, repeats=5, device='cpu'):
    """
    Create a tensor that linearly interpolates from `start` to `end` with `steps` points,
    and repeats it `repeats` times along rows.

    Returns:
        Tensor of shape (repeats, steps)
    """
    base = torch.linspace(start, end, steps, device=device)  # shape: (steps,)
    repeated = base.repeat(repeats, 1)  # shape: (repeats, steps)
    return repeated



def rk4(positions, velocity_model, t_tensor, dt):
    """
    RK4 integrator for positions using a velocity model.

    Args:
        positions (N, 2): Current positions.
        velocity_model (callable): Function or nn.Module taking (N, 2) → (N, 2) velocities.
        dt (float): Time step.

    Returns:
        (N, 2): Updated positions.
    """
    k1 = velocity_model(positions,t_tensor)
    k2 = velocity_model(positions + 0.5 * dt * k1, t_tensor)
    k3 = velocity_model(positions + 0.5 * dt * k2, t_tensor)
    k4 = velocity_model(positions + dt * k3, t_tensor)

    return positions + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)



# def animate_trajectories_and_vector_field(
#     positions,         # [N, 2, T]
#     velocities,        # [400, 2, T]
#     grid_X, grid_Y,    # [20, 20] meshgrid
#     interval=100,
#     save_path=None
# ):
#     """
#     Animates 2D particle trajectories and velocity vector fields over time.

#     Args:
#         positions (Tensor or ndarray): shape [N, 2, T]
#         velocities (Tensor or ndarray): shape [400, 2, T]
#         grid_X, grid_Y (Tensor or ndarray): meshgrid of shape [20, 20]
#         interval (int): milliseconds between frames
#         save_path (str): optional file path to save as mp4
#     """
#     import numpy as np

#     # Convert tensors to NumPy
#     if isinstance(positions, torch.Tensor):
#         positions = positions.cpu().numpy()
#     if isinstance(velocities, torch.Tensor):
#         velocities = velocities.cpu().numpy()
#     if isinstance(grid_X, torch.Tensor):
#         grid_X = grid_X.cpu().numpy()
#     if isinstance(grid_Y, torch.Tensor):
#         grid_Y = grid_Y.cpu().numpy()

#     N, _, T = positions.shape
#     H, W = grid_X.shape
#     assert velocities.shape == (H*W, 2, T), "Velocity tensor must be [400, 2, T]"

#     fig, ax = plt.subplots(figsize=(6, 6))
#     scat = ax.scatter([], [], s=8, color="#00ec47", label='Particles')
#     quiv = ax.quiver(grid_X, grid_Y,
#                      np.zeros_like(grid_X), np.zeros_like(grid_Y),
#                      color="#000000", scale=40, alpha=0.5)

#     ax.set_xlim(-1.5, 1.5)
#     ax.set_ylim(-1.5, 1.5)
#     ax.set_aspect('equal')
#     ax.grid(False)
#     ax.set_title("t = 0")
#     ax.legend(loc='upper right')

#     def update(frame):
#         pos_t = positions[:, :, frame]  # [N, 2]
#         scat.set_offsets(pos_t)

#         vel_t = velocities[:, :, frame]  # [400, 2]
#         U = vel_t[:, 0].reshape(H, W)
#         V = vel_t[:, 1].reshape(H, W)
#         quiv.set_UVC(U, V)

#         ax.set_title(f"Time Step: {frame}/{T - 1}")
#         return scat, quiv

#     anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)

#     if save_path:
#         anim.save(save_path, writer='ffmpeg', fps=1000 // interval)
#         print(f"Saved animation to {save_path}")
#     else:
#         plt.show()


# def animate_trajectories_and_vector_field(
#     positions,         # [N, 2, T]
#     velocities,        # [400, 2, T]
#     grid_X, grid_Y,    # [20, 20] meshgrid
#     x1=None,           # [N, 2]
#     y1=None,           # [N, 2]
#     measurement=None,  # [1, 2] or [2,]
#     interval=100,
#     save_path=None
# ):
#     """
#     Animates 2D particle trajectories and velocity vector fields over time,
#     with optional fixed inputs plotted (x1, y1, measurement).

#     Args:
#         positions (Tensor or ndarray): shape [N, 2, T]
#         velocities (Tensor or ndarray): shape [400, 2, T]
#         grid_X, grid_Y (Tensor or ndarray): meshgrid of shape [20, 20]
#         x1 (Tensor or ndarray): fixed points [N, 2]
#         y1 (Tensor or ndarray): fixed points [N, 2]
#         measurement (Tensor or ndarray): fixed point [1, 2] or [2,]
#         interval (int): milliseconds between frames
#         save_path (str): optional file path to save as mp4
#     """
#     # Convert to NumPy
#     if isinstance(positions, torch.Tensor): positions = positions.cpu().numpy()
#     if isinstance(velocities, torch.Tensor): velocities = velocities.cpu().numpy()
#     if isinstance(grid_X, torch.Tensor): grid_X = grid_X.cpu().numpy()
#     if isinstance(grid_Y, torch.Tensor): grid_Y = grid_Y.cpu().numpy()
#     if isinstance(x1, torch.Tensor): x1 = x1.cpu().numpy()
#     if isinstance(y1, torch.Tensor): y1 = y1.cpu().numpy()
#     if isinstance(measurement, torch.Tensor): measurement = measurement.cpu().numpy()

#     N, _, T = positions.shape
#     H, W = grid_X.shape
#     assert velocities.shape == (H * W, 2, T), "Velocity tensor must be [400, 2, T]"

#     fig, ax = plt.subplots(figsize=(6, 6))
    
#     # Moving particles
#     scat = ax.scatter([], [], s=8, color="#0070d2", label=r'$p(x_k \mid y_k)$')

#     # Static vector field
#     quiv = ax.quiver(grid_X, grid_Y,
#                     np.zeros_like(grid_X), np.zeros_like(grid_Y),
#                     color="#9A9A9A", scale=100, alpha=0.5)

#     # Fixed reference points
#     if x1 is not None:
#         ax.scatter(x1[:, 0], x1[:, 1], c="#00E7A9", s=10, label=r'$p(x_k \mid x_{k-1})$')
#     if y1 is not None:
#         ax.scatter(y1[:, 0], y1[:, 1], c="#9DFF00", s=10, label=r'$p(y_k \mid x_k)$', alpha=0.5)
#     if measurement is not None:
#         if measurement.ndim == 2 and measurement.shape[0] == 1:
#             measurement = measurement[0]  # flatten from (1, 2) to (2,)
#         ax.scatter(measurement[0], measurement[1], c="#FF00B3", s=50, marker="*", label="measurement")

#     ax.set_xlim(-1.5, 1.5)
#     ax.set_ylim(-1.5, 1.5)
#     ax.set_aspect('equal')
#     ax.grid(False)
#     ax.set_title("t = 0")
#     ax.legend(loc='upper right')

#     def update(frame):
#         pos_t = positions[:, :, frame]  # [N, 2]
#         scat.set_offsets(pos_t)

#         vel_t = velocities[:, :, frame]  # [400, 2]
#         U = vel_t[:, 0].reshape(H, W)
#         V = vel_t[:, 1].reshape(H, W)
#         quiv.set_UVC(U, V)

#         ax.set_title(f"Time Step: {frame}/{T - 1}")
#         return scat, quiv

#     anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)

#     if save_path:
#         anim.save(save_path, writer='ffmpeg', fps=1000 // interval)
#         print(f"Saved animation to {save_path}")
#     else:
#         plt.show()

def animate_trajectories_and_vector_field(
    positions,         # [N, 2, T]
    velocities,        # [400, 2, T]
    grid_X, grid_Y,    # [20, 20] meshgrid
    x1=None,           # [N, 2]
    y1=None,           # [N, 2]
    measurement=None,  # [1, 2] or [2,]
    interval=100,
    save_path=None,    # for MP4 video
    frame_dir=None     # optional: folder to save each frame as PNG
):
    """
    Animates 2D particle trajectories and velocity vector fields over time.
    Saves an MP4 animation (if save_path is provided) and/or individual PNG frames (if frame_dir is provided).
    """

    # Convert tensors to NumPy
    if isinstance(positions, torch.Tensor): positions = positions.cpu().numpy()
    if isinstance(velocities, torch.Tensor): velocities = velocities.cpu().numpy()
    if isinstance(grid_X, torch.Tensor): grid_X = grid_X.cpu().numpy()
    if isinstance(grid_Y, torch.Tensor): grid_Y = grid_Y.cpu().numpy()
    if isinstance(x1, torch.Tensor): x1 = x1.cpu().numpy()
    if isinstance(y1, torch.Tensor): y1 = y1.cpu().numpy()
    if isinstance(measurement, torch.Tensor): measurement = measurement.cpu().numpy()

    N, _, T = positions.shape
    H, W = grid_X.shape
    assert velocities.shape == (H * W, 2, T), "Velocity tensor must be [400, 2, T]"

    fig, ax = plt.subplots(figsize=(6, 6))
    
    scat = ax.scatter([], [], s=8, color="#0070d2", label=r'Posterior $p(x \mid y)$')
    quiv = ax.quiver(grid_X, grid_Y,
                     np.zeros_like(grid_X), np.zeros_like(grid_Y),
                     color="#9A9A9A", scale=100, alpha=0.5)

    if x1 is not None:
        ax.scatter(x1[:, 0], x1[:, 1], c="#00E7A9", s=10, label=r'Prior $p(x)$')
    if y1 is not None:
        ax.scatter(y1[:, 0], y1[:, 1], c="#9DFF00", s=10, label=r'Marginal $p(y)$', alpha=0.5)
    if measurement is not None:
        if measurement.ndim == 2 and measurement.shape[0] == 1:
            measurement = measurement[0]
        ax.scatter(measurement[0], measurement[1], c="#FF00B3", s=50, marker="*", label="Observation")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.legend(loc='upper right')

    def update(frame):
        pos_t = positions[:, :, frame]
        scat.set_offsets(pos_t)

        vel_t = velocities[:, :, frame]
        U = vel_t[:, 0].reshape(H, W)
        V = vel_t[:, 1].reshape(H, W)
        quiv.set_UVC(U, V)

        ax.set_title(f"Time Step: {frame}/{T - 1}")
        return scat, quiv

    # Save individual frames (PNG images)
    if frame_dir is not None:
        os.makedirs(frame_dir, exist_ok=True)
        print(f"Saving individual frames to '{frame_dir}'")
        for frame in range(T):
            update(frame)
            fig.canvas.draw()
            frame_filename = os.path.join(frame_dir, f"frame_{frame:04d}.png")
            fig.savefig(frame_filename, dpi=150)
        print("Finished saving frames.")

    # Create and save animation (MP4)
    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)

    if save_path:
        print(f"Saving animation to '{save_path}'")
        anim.save(save_path, writer='ffmpeg', fps=1000 // interval)
        print("Finished saving animation.")
    elif frame_dir is None:
        plt.show()


def measurement_operator(samples_x):
    """
    Generate n samples from a fixed 2D multivariate Gaussian distribution.

    Parameters:
    - n (int): Number of samples
    - d (int): Dimensionality (must be 2)

    Returns:
    - samples (ndarray): 2 x n array of samples
    """
    
    sample_size, dim = samples_x.shape


    if dim != 2:
        raise ValueError("This function only supports 2D Gaussian samples (d must be 2).")
    
    # Fixed mean and covariance matrix
    mu = np.array([0, 0])
    # cov = np.array([[1, 0.125],
    #                 [0.125, .25]])  # s11=1.0, s12=s21=0.8, s22=2.0
    cov = 0.01*torch.eye(2)

    # Ensure positive semi-definiteness
    if not np.all(np.linalg.eigvals(cov) >= 0):
        raise ValueError("Covariance matrix is not positive semi-definite.")

    samples = samples_x + np.random.multivariate_normal(mu, cov, sample_size)  # Shape: 2 x n
    return samples.float()

def construct_joint_dist(samples_x, samples_y):

    """
    Stack original and noisy samples into an (N x 4) array.

    Parameters:
    - samples_x: (N x 2) ndarray of original samples
    - samples_y: (N x 2) ndarray of noisy samples

    Returns:
    - stacked: (N x 4) ndarray with [x1, x2, y1, y2] per row
    """
    # if samples_x.shape != samples_y.shape:
    #     raise ValueError("samples_x and samples_y must have the same shape.")
    
    if samples_x.shape[1] != 2:
        raise ValueError("Each sample must be 2-dimensional.")

    stacked = np.hstack((samples_x, samples_y))  # Shape: (N, 4)
    return torch.from_numpy(stacked).float()


def jacobian_(th1, th2):

    j = torch.tensor([[-np.sin(th1)-np.sin(th1+th2), -np.sin(th1+th2)],
                     [np.cos(th1)+np.cos(th1+th2), np.cos(th1+th2)]])
    
    return j


if __name__ == "__main__":

    j = jacobian_(0, np.pi)
    jt = j.T
    tau = torch.tensor([1,1])
    f = torch.inverse(jt)
    print(f)