import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "ligandsbm_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Dummy Data Generation (Simulating Ligand Poses) ---
def get_dummy_data(n_samples=1000):
    """
    Simulates:
    - x0: Initial pose distribution (e.g., generic docking funnel)
    - x1_a: Target pose A (Binding Mode 1)
    - x1_b: Target pose B (Binding Mode 2)
    """
    # Source: Cluster around (0, -2)
    x0 = torch.randn(n_samples, 2) * 0.5 + torch.tensor([0.0, -2.0])
    
    # Target A: Cluster around (-2, 2)
    x1_a = torch.randn(n_samples, 2) * 0.3 + torch.tensor([-2.0, 2.0])
    
    # Target B: Cluster around (2, 2)
    x1_b = torch.randn(n_samples, 2) * 0.3 + torch.tensor([2.0, 2.0])
    
    return x0.to(DEVICE), [x1_a.to(DEVICE), x1_b.to(DEVICE)]

def smiles_to_poses(smiles: str):
    """
    Mock function to demonstrate interface.
    In a real app, this would use RDKit to generate conformers/poses.
    """
    print(f"Processing SMILES: {smiles}")
    print("Generating initial pose ensemble...")
    return get_dummy_data(n_samples=100)

# --- Model Components ---

class GeoPathMLP(nn.Module):
    """
    Learns the non-linear deviation from the straight line path.
    G(x0, x1, t)
    """
    def __init__(self, dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x0, x1, t):
        # t shape: [B, 1]
        # x0, x1 shape: [B, D]
        x = torch.cat([x0, x1, t], dim=-1)
        return self.net(x)

class VelocityNet(nn.Module):
    """
    Learns the vector field v(x, t).
    """
    def __init__(self, dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, t):
        # x: [B, D], t: [B, 1]
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)

class BranchSBM(nn.Module):
    """
    Simplified implementation of Branched Schrödinger Bridge Matching.
    """
    def __init__(self, geopath_nets, alpha=1.0):
        super().__init__()
        self.geopath_nets = nn.ModuleList(geopath_nets)
        self.alpha = alpha
        self.branches = len(geopath_nets)

    def gamma(self, t):
        # Bridge function: 0 at t=0 and t=1, non-zero in between.
        # Using the form from the repo: 1 - t^2 - (1-t)^2 = 2t - 2t^2 = 2t(1-t)
        # Repo uses: 1 - ((t-tmin)/(tmax-tmin))^2 - ... assuming t in [0,1]
        return t * (1 - t) # Simplified bell shape

    def d_gamma(self, t):
        # Derivative of gamma(t) = t - t^2 -> 1 - 2t
        return 1 - 2 * t

    def compute_mu_t(self, x0, x1, t, branch_idx):
        # Linear interpolation
        linear_path = (1 - t) * x0 + t * x1
        
        if self.alpha == 0:
            return linear_path
        
        # Add non-linear deviation
        # G(x0, x1, t)
        g_out = self.geopath_nets[branch_idx](x0, x1, t)
        
        # mu_t = linear + gamma(t) * G
        return linear_path + self.gamma(t) * g_out

    def compute_conditional_flow(self, x0, x1, t, xt, branch_idx):
        # u_t = d/dt (mu_t)
        # u_t = (x1 - x0) + gamma'(t) * G + gamma(t) * dG/dt
        
        # For simplicity, we ignore dG/dt term (assuming G varies slowly or we detach)
        # The repo includes dG/dt if time_geopath is True.
        # Here we approximate or compute gradients if needed.
        
        # Re-compute G to be safe
        t.requires_grad_(True)
        g_out = self.geopath_nets[branch_idx](x0, x1, t)
        
        # dG/dt
        dg_dt = torch.autograd.grad(
            outputs=g_out,
            inputs=t,
            grad_outputs=torch.ones_like(g_out),
            create_graph=True,
            retain_graph=True
        )[0]
        
        linear_vel = (x1 - x0)
        bridge_vel = self.d_gamma(t) * g_out + self.gamma(t) * dg_dt
        
        return linear_vel + bridge_vel

    def sample_location_and_flow(self, x0, x1, t, branch_idx):
        # Sample xt
        mu_t = self.compute_mu_t(x0, x1, t, branch_idx)
        sigma_t = 0.1 # Fixed small noise for SBM
        eps = torch.randn_like(x0)
        xt = mu_t + sigma_t * eps
        
        # Compute target vector field u_t at xt
        # Note: The repo computes u_t at mu_t or xt? 
        # Repo: compute_conditional_flow takes xt but deletes it immediately.
        # It returns flow based on x0, x1, t. 
        # So u_t is the velocity of the *mean* path, not the sample?
        # Actually, standard CFM targets the conditional flow u_t(x|x0,x1).
        # If sigma is constant, u_t is just d/dt mu_t.
        ut = self.compute_conditional_flow(x0, x1, t, xt, branch_idx)
        
        return xt, ut

# --- Training Functions ---

def train_geopath(sbm, x0, x1_list, n_steps=1000):
    """
    Phase 1: Train Geodesic Interpolants to minimize energy.
    Energy = Kinetic + Potential.
    Potential = Obstacle cost.
    """
    optimizers = [optim.Adam(net.parameters(), lr=1e-3) for net in sbm.geopath_nets]
    
    # Define an obstacle at (0, 0)
    def obstacle_cost(x):
        # Gaussian hill at (0, 0)
        dist = torch.sum(x**2, dim=1)
        return 10.0 * torch.exp(-dist / 0.5)

    print("Training Geodesic Interpolants...")
    for step in range(n_steps):
        for i in range(sbm.branches):
            optimizer = optimizers[i]
            optimizer.zero_grad()
            
            # Sample t
            t = torch.rand(x0.shape[0], 1, device=DEVICE, requires_grad=True)
            
            # Compute path position xt (deterministic part for energy)
            # We want to minimize energy of the *mean* path
            mu_t = sbm.compute_mu_t(x0, x1_list[i], t, i)
            
            # Compute velocity ut
            ut = sbm.compute_conditional_flow(x0, x1_list[i], t, mu_t, i)
            
            # Kinetic Energy: ||u||^2
            kinetic = torch.sum(ut**2, dim=1).mean()
            
            # Potential Energy: V(x)
            potential = obstacle_cost(mu_t).mean()
            
            loss = kinetic + potential
            loss.backward()
            optimizer.step()
            
        if step % 200 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

def train_flow(sbm, flow_nets, x0, x1_list, n_steps=1000):
    """
    Phase 2: Train Flow Matching Networks to match the SBM vector field.
    """
    optimizers = [optim.Adam(net.parameters(), lr=1e-3) for net in flow_nets]
    
    print("Training Flow Networks...")
    for step in range(n_steps):
        for i in range(sbm.branches):
            optimizer = optimizers[i]
            optimizer.zero_grad()
            
            t = torch.rand(x0.shape[0], 1, device=DEVICE)
            
            # Get target flow from SBM
            xt, ut_target = sbm.sample_location_and_flow(x0, x1_list[i], t, i)
            
            # Predict flow
            v_pred = flow_nets[i](xt, t)
            
            loss = torch.mean((v_pred - ut_target)**2)
            loss.backward()
            optimizer.step()
            
        if step % 200 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

# --- Main Execution ---

def main():
    print("Initializing LigandSBM (Branched Schrödinger Bridge Matching)...")
    
    # 1. Data
    x0, x1_list = get_dummy_data()
    
    # 2. Models
    # Two branches: 0 -> A, 0 -> B
    geopath_nets = [GeoPathMLP(dim=2).to(DEVICE) for _ in range(2)]
    sbm = BranchSBM(geopath_nets).to(DEVICE)
    
    flow_nets = [VelocityNet(dim=2).to(DEVICE) for _ in range(2)]
    
    # 3. Train
    train_geopath(sbm, x0, x1_list, n_steps=500)
    train_flow(sbm, flow_nets, x0, x1_list, n_steps=500)
    
    # 4. Visualize
    print("Generating visualization...")
    plt.figure(figsize=(8, 8))
    
    # Plot Data
    plt.scatter(x0.cpu().numpy()[:, 0], x0.cpu().numpy()[:, 1], alpha=0.1, label='Source (Initial Poses)', c='gray')
    plt.scatter(x1_list[0].cpu().numpy()[:, 0], x1_list[0].cpu().numpy()[:, 1], alpha=0.1, label='Target A', c='blue')
    plt.scatter(x1_list[1].cpu().numpy()[:, 0], x1_list[1].cpu().numpy()[:, 1], alpha=0.1, label='Target B', c='red')
    
    # Plot Trajectories (Inference)
    # Take a few source points
    test_x0 = x0[:10]
    
    for branch_idx in range(2):
        color = 'blue' if branch_idx == 0 else 'red'
        flow_net = flow_nets[branch_idx]
        
        # Euler Integration
        traj = [test_x0]
        curr_x = test_x0.clone()
        dt = 0.05
        for t_val in np.arange(0, 1.0, dt):
            t_tensor = torch.full((curr_x.shape[0], 1), t_val, device=DEVICE)
            v = flow_net(curr_x, t_tensor)
            curr_x = curr_x + v * dt
            traj.append(curr_x)
            
        traj = torch.stack(traj).cpu().detach().numpy()
        
        # Plot lines
        for i in range(traj.shape[1]):
            plt.plot(traj[:, i, 0], traj[:, i, 1], c=color, alpha=0.5, linewidth=1)

    # Plot Obstacle
    circle = plt.Circle((0, 0), 0.5, color='black', fill=False, linestyle='--', label='Obstacle')
    plt.gca().add_patch(circle)
    
    plt.legend()
    plt.title("LigandSBM: Branched Transport of Ligand Poses")
    plt.savefig(f"{OUTPUT_DIR}/ligandsbm_demo.png")
    print(f"Visualization saved to {OUTPUT_DIR}/ligandsbm_demo.png")

if __name__ == "__main__":
    # Example usage with "SMILES"
    smiles_to_poses("CCO")
    main()
