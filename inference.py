#!/usr/bin/env python
import argparse
import os
import random
from typing import Optional
import pandas as pd

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, Subset

# Assuming these modules exist locally
from model.spflow import Flow
from dataload.dataset import SolarDataModule

torch.set_float32_matmul_precision('high')

# Set plotting style
plt.style.use('dark_background') 
plt.rcParams['axes.unicode_minus'] = False 

def ensure_dir(file_path):
    """Ensure the directory exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def capture_inference_trajectory(model, dataloader, device, num_samples=30):
    """
    Run inference and capture intermediate trajectories x_t_seq.
    Also returns the physical input (x_nsrdb) to generate the mask later.
    """
    model.eval()
    model.to(device)
    
    # Get a full batch
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        raise ValueError("Dataloader is empty.")
    
    # Get batch size
    batch_size = batch['y'].shape[0]
    
    # If batch size is smaller than requested samples, use all; otherwise, sample randomly
    if num_samples == -1 or batch_size <= num_samples:
        indices = torch.arange(batch_size)
        actual_samples = batch_size
        if num_samples != -1:
            print(f"Warning: Batch size ({batch_size}) is smaller or equal to requested samples ({num_samples}). Using all available.")
        else:
            print(f"Using all {batch_size} samples in the batch (max_samples=-1).")
    else:
        # Generate random unique indices
        indices = torch.randperm(batch_size)[:num_samples]
        actual_samples = num_samples
        print(f"Randomly selected {actual_samples} samples from batch of size {batch_size}.")

    # Move data to device and slice based on random indices
    batch_subset = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            # Slice first then move to GPU to save memory
            batch_subset[k] = v[indices].to(device)
            
    with torch.no_grad():
        # 1. Get RNN Prior
        rnn_pred = model.get_rnn_prior(batch_subset)
        noise = torch.randn_like(rnn_pred)
        source_dist_samples = 1 * rnn_pred + 0.1 * noise
        x_final, x_t_seq, _ = model.generate_reconstructions(
            x0=source_dist_samples,
            x=batch_subset['y'], 
            y=batch_subset['x_nsrdb'],
            x_future=rnn_pred,
            num_flow_steps=model.hparams.num_flow_steps,
            result_device='cpu' 
        )
        x_t_seq_cpu = [t.cpu() for t in x_t_seq] 
        traj_stack = torch.stack(x_t_seq_cpu).numpy().squeeze(-1) 

        ground_truths = batch_subset['y'].cpu().numpy().squeeze(-1)
        rnn_priors = rnn_pred.cpu().numpy().squeeze(-1)
        physical_inputs = batch_subset['x_nsrdb'].cpu().numpy()
        
    return traj_stack, ground_truths, rnn_priors, physical_inputs

# --- Chart 1: Spectral Evolution Flow ---
def plot_spectral_evolution(trajectory, ground_truth, rnn_prior, sample_idx=0, save_path=None):
    steps, _, seq_len = trajectory.shape
    data = trajectory[:, sample_idx, :] 
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    bg_color = "#575b65"
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    
    # Plot reference lines
    ax.plot(ground_truth[sample_idx], color='white', linestyle='--', linewidth=2, label='Ground Truth', zorder=100)
    ax.plot(rnn_prior[sample_idx], color='#22d3ee', linestyle=':', linewidth=2, label='RNN Prior (Masked)', zorder=90)

    # Plot Flow gradients
    colors = cm.plasma(np.linspace(0, 1, steps))
    for i in range(steps):
        alpha = 0.3 + 0.7 * (i / steps)
        ax.plot(data[i], color=colors[i], linewidth=1.5, alpha=alpha)

    ax.set_title(f'Flow Matching Trajectory (Masked) - Sample {sample_idx}', color='white', fontsize=14)
    ax.set_xlabel('Time Step (Horizon)', color='#94a3b8')
    ax.set_ylabel('Solar Irradiance (W/m^2)', color='#94a3b8') 
    ax.grid(True, color='#334155', linestyle='--', alpha=0.5)
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
    
    sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Flow Time $t$ (0 $\to$ 1)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, facecolor=bg_color)
        print(f"[Saved] Spectral Plot -> {save_path}")
    plt.close()

# --- Chart 2: Spatiotemporal Heatmap ---
def plot_flow_heatmap(trajectory, sample_idx=0, save_path=None):
    steps, _, seq_len = trajectory.shape
    data = trajectory[:, sample_idx, :] 
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    
    plt.imshow(data, aspect='auto', cmap='inferno', origin='lower', 
               extent=[0, seq_len, 0, 1])
    
    cbar = plt.colorbar()
    cbar.set_label('Irradiance Value (W/m^2)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    
    plt.title(f'Flow Dynamics Heatmap (Masked) - Sample {sample_idx}', color='white', fontsize=14)
    plt.xlabel('Forecast Horizon', color='white')
    plt.ylabel('Flow Time $t$', color='white')
    plt.tick_params(colors='white')
    
    plt.tight_layout()
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path, facecolor='black')
        print(f"[Saved] Heatmap -> {save_path}")
    plt.close()

# --- Chart 3: Dynamic GIF ---
def create_flow_gif(trajectory, ground_truth, rnn_prior, sample_idx=0, save_path=None):
    if not save_path: return

    ensure_dir(save_path)
    
    steps, _, seq_len = trajectory.shape
    data = trajectory[:, sample_idx, :]
    gt = ground_truth[sample_idx]
    prior = rnn_prior[sample_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    bg_color = '#0f172a'
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    ax.plot(gt, color='white', linestyle='--', linewidth=2, label='Target', alpha=0.6)
    ax.plot(prior, color='#22d3ee', linestyle=':', linewidth=2, label='RNN Prior', alpha=0.6)
    
    line, = ax.plot([], [], color='#facc15', linewidth=3, label='Flow State $x_t$')
    title_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', color='white', fontsize=14)
    
    # Dynamically set Y-axis range
    y_min, y_max = data.min(), data.max()
    margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlim(0, seq_len - 1)
    
    ax.grid(True, color='#334155', linestyle='--')
    ax.legend(facecolor='#1e293b', labelcolor='white', loc='upper right')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values(): spine.set_edgecolor('#334155')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        current_data = data[frame]
        line.set_data(range(seq_len), current_data)
        t_val = frame / (steps - 1)
        title_text.set_text(f'Flow Step: {frame}/{steps} | t={t_val:.2f}')
        line.set_color(cm.plasma(t_val))
        return line, title_text

    ani = animation.FuncAnimation(fig, update, frames=steps, init_func=init, blit=False)
    ani.save(save_path, writer='pillow', fps=20)
    print(f"[Saved] GIF -> {save_path}")
    plt.close()

def main(args: argparse.Namespace):
    # 1. Initialize DataModule
    dm = SolarDataModule(args.config_path)
    dm.prepare_data()
    
    # 2. Load Model
    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
        
    print(f"Loading model from {args.ckpt_path}...")
    model: Flow = Flow.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        map_location='cpu',
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        num_flow_steps=args.num_flow_steps,
        ema_decay=args.ema_decay,
        eps=args.eps,
        test_results_path=args.output_dir,  
        rnn_checkpoint_path=args.rnn_checkpoint_path,
        scaler=None 
    )

    # 3. Setup WandB Logger (if enabled)
    logger = None
    if args.enable_wandb:
        logger = WandbLogger(
            project=args.wandb_project_name,
            group=args.wandb_group,
            id=args.wandb_id
        )

    # 4. Initialize Trainer
    trainer = Trainer(
        accelerator='auto',
        devices=args.num_gpus,
        logger=logger,
        precision=args.precision
    )

    # =========================================================
    # PHASE 1: Run Standard Test Set Evaluation
    # =========================================================
    print("\n" + "="*50)
    print("PHASE 1: Running Standard Test Evaluation...")
    print("="*50 + "\n")
    
    # trainer.test(model, datamodule=dm)

    # =========================================================
    # PHASE 2: Custom Visualization & Inference
    # =========================================================
    print("\n" + "="*50)
    print("PHASE 2: Running Custom Visualization...")
    print("="*50 + "\n")

    print("Ensuring Scaler is loaded (running setup('fit'))...")
    dm.setup(stage='fit') 
    
    scaler = getattr(dm, 'scaler', None)
    
    # Attach Scaler to Model
    if scaler is not None:
        model.scaler = scaler
        print("Scaler successfully attached to model.")
    else:
        raise RuntimeError("Could not load Scaler from DataModule. Inference cannot proceed without inverse_transform.")
    
    # Set device for custom inference
    device = 'cuda' if torch.cuda.is_available() and args.num_gpus > 0 else 'cpu'
    model.to(device)
    
    # Prepare DataLoader
    if args.use_test_split:
        dm.setup(stage='test')
        if hasattr(dm, 'test_dataloader'):
            dl = dm.test_dataloader()
        else:
            dl = dm.val_dataloader()
    else:
        dl = dm.val_dataloader()

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    csv_dir = os.path.join(args.output_dir, "csv_data")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    print(f"Capturing inference trajectory for {args.max_samples if args.max_samples != -1 else 'ALL'} samples...")
    
    traj_stack, gt, rnn, phys_inputs = capture_inference_trajectory(
        model=model, 
        dataloader=dl, 
        device=device, 
        num_samples=args.max_samples
    )
    
    # --- Denormalization & Masking ---
    print("Denormalizing trajectories and applying Physical Mask...")
    
    # 1. Denormalize Physical Inputs to calculate Mask
    phys_tensor = torch.tensor(phys_inputs, device=device, dtype=torch.float32)
    with torch.no_grad():
        # phys_denorm shape: (30, 288, 14)
        phys_denorm = model.scaler.inverse_transform(phys_tensor, 'surfrad').cpu().numpy()
    
    # 2. Extract the correct slice for the mask
    # Target shape is (30, 96). Physical Input is (30, 288, 14).
    # We assume the target corresponds to the LAST 96 steps of the input sequence.
    # We also assume the Physical Irradiance (GHI) is at index 0.
    target_seq_len = traj_stack.shape[2] # 96
    
    # Slice: Take last 'target_seq_len' steps and 0th channel
    phys_denorm_sliced = phys_denorm[:, -target_seq_len:, 0] 
    
    # Create Mask: If Physical Model > 1.0, it is Day. Else Night (0).
    mask = (phys_denorm_sliced > 1.0).astype(np.float32)
    print(f"Mask created. Shape: {mask.shape}") # Should be (30, 96)

    # 3. Denormalize Trajectory
    traj_denorm_list = []
    steps = traj_stack.shape[0]
    
    for t in range(steps):
        data_t_numpy = traj_stack[t]
        data_t_tensor = torch.tensor(data_t_numpy, device=device, dtype=torch.float32)
        
        with torch.no_grad():
            data_t_denorm = model.scaler.inverse_transform(data_t_tensor, 'surfrad').cpu().numpy()
        
        traj_denorm_list.append(data_t_denorm)
    
    traj_denorm = np.stack(traj_denorm_list) # (Steps, Samples, Seq_Len)
    
    # 4. Apply Mask to Trajectory
    # Broadcast mask: (Samples, Seq_Len) -> (1, Samples, Seq_Len)
    # traj_denorm: (51, 30, 96)
    # mask: (30, 96)
    traj_denorm = traj_denorm * mask[None, :, :]

    # 5. Denormalize and Mask Ground Truth and RNN Prior
    gt_tensor = torch.tensor(gt, device=device, dtype=torch.float32)
    rnn_tensor = torch.tensor(rnn, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        gt_denorm = model.scaler.inverse_transform(gt_tensor, 'surfrad').cpu().numpy()
        rnn_denorm = model.scaler.inverse_transform(rnn_tensor, 'surfrad').cpu().numpy()
    
    # Apply mask to RNN Prior
    rnn_denorm = rnn_denorm * mask

    # Save and Plot
    for i in range(traj_denorm.shape[1]):
        print(f"--- Processing Sample {i+1}/{traj_denorm.shape[1]} ---")
        
        # Save CSV
        sample_traj_data = traj_denorm[:, i, :]
        df = pd.DataFrame(sample_traj_data)
        df.index.name = 'flow_step'
        df.columns.name = 'forecast_horizon_step'
        
        df.loc['target'] = gt_denorm[i]
        df.loc['rnn_prior'] = rnn_denorm[i]
        df.loc['physical_mask'] = mask[i] 
        
        csv_path = os.path.join(csv_dir, f"sample_{i}_trajectory.csv")
        df.to_csv(csv_path)

        # Plot
        path_spectral = os.path.join(vis_dir, f"sample_{i}_spectral.png")
        path_heatmap = os.path.join(vis_dir, f"sample_{i}_heatmap.png")
        path_gif = os.path.join(vis_dir, f"sample_{i}_flow.gif")
        
        plot_spectral_evolution(traj_denorm, gt_denorm, rnn_denorm, sample_idx=i, save_path=path_spectral)
        plot_flow_heatmap(traj_denorm, sample_idx=i, save_path=path_heatmap)
        create_flow_gif(traj_denorm, gt_denorm, rnn_denorm, sample_idx=i, save_path=path_gif)

    print(f"All outputs saved to: {args.output_dir}")
    rank_zero_info("Inference finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_checkpoint_path', type=str, required=False, default='/home/joty/code/FM_interpolation/checkpoints/flow_matching',
                        help='rnn_checkpoint_path')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to a trained checkpoint (.ckpt).')
    parser.add_argument('--config_path', type=str, default='/home/joty/code/FM_interpolation/config/config.yaml',
                        help='YAML config path for SolarDataModule.')
    parser.add_argument('--output_dir', type=str, default='/home/joty/code/FM_interpolation/outputs/test_infer',
                        help='Directory to save inference results.')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32', choices=['bf16', '32', '16-mixed', 'bf16-mixed'])
    parser.add_argument('--use_test_split', action='store_true', help='Use test split if available (default).')
    parser.add_argument('--max_samples', type=int, default=30, 
                        help='Number of samples to visualize. Set to -1 to use the entire batch.')

    parser.set_defaults(use_test_split=True)

    # Model args
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument('--num_flow_steps', type=int, default=50)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--eps', type=float, default=0.0)

    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--wandb_project_name', type=str, default='flow_matching')
    parser.add_argument('--wandb_group', type=str, default='inference')
    parser.add_argument('--wandb_id', type=str, default=None)

    args = parser.parse_args()
    main(args)