import math
import warnings
from typing import Union, Optional, List, Tuple, Callable
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
from contextlib import contextmanager, nullcontext
from pathlib import Path
from dataload.dataset import SolarScaler
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch_ema import ExponentialMovingAverage as EMA
from torch.nn.functional import sigmoid
from torchdyn.core import NeuralODE
from model.dete_model.RNN import SolarSeq2Seq
from model.dete_model.SPTransformer import Model
import wandb
import os
import pandas as pd
import torch.distributed as dist
from model.cfm import (
    ConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from model.interpolant import Interpolant
from model.denoising.SSSD.SSSD import BackboneModel
# ----------------- CFM -----------------
def quantile_loss_pinball(y_pred: torch.Tensor, y_true: torch.Tensor, q: torch.Tensor) -> torch.Tensor:  
    # y_pred, y_true: same shape
    # q: scalar tensor or broadcastable to y_pred
    e = y_true - y_pred
    return torch.maximum(q * e, (q - 1.0) * e)

def tsflow_like_schedule(sigmin: float = 1e-3, sigmax: float = 1.0) -> Callable[[torch.Tensor], torch.Tensor]:  # NEW

    dsig = sigmin - sigmax  
    def sch(t_b):
        sig_t = (1.0 - t_b) * sigmax + t_b * sigmin
        scale = (-dsig) * sig_t  # >=0
        return scale
    return sch

class GuidedVF(torch.nn.Module): 

    def __init__(
        self,
        denoiser: torch.nn.Module,
        cond: torch.Tensor,
        x_future: torch.Tensor,
        guidance_scale: float = 0.0
    ):
        super().__init__()
        self.denoiser = denoiser
        self.cond = cond
        self.x_future = x_future
        self.guidance_scale = float(guidance_scale)
        self.y_obs: Optional[torch.Tensor] = None    
        self.obs_mask: Optional[torch.Tensor] = None 
        self.quantiles: Optional[Union[List[float], torch.Tensor]] = None  
        self.schedule: Optional[Callable[[torch.Tensor], torch.Tensor]] = None  

    def forward(self, t: Union[float, torch.Tensor], x: torch.Tensor, args: Optional[dict] = None) -> torch.Tensor:  
        B = x.shape[0]
        if torch.is_tensor(t):
            t_batch = t.to(dtype=x.dtype, device=x.device).expand(B)
        else:
            t_batch = torch.full((B,), float(t), device=x.device, dtype=x.dtype)
        v = self.denoiser(input=x, t=t_batch, features=self.cond, x_future=self.x_future)
        if self.guidance_scale <= 0.0 or (self.y_obs is None) or (self.obs_mask is None): 
            return v

        y_obs = self.y_obs
        obs_mask = self.obs_mask
        if self.quantiles is None:
            q = torch.linspace(0.1, 0.9, 9, device=x.device, dtype=x.dtype)  
        else:
            q = torch.as_tensor(self.quantiles, device=x.device, dtype=x.dtype)  
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            t_bcast = t_batch.view(B, *([1] * (x.dim() - 1)))
            v_here = self.denoiser(input=x, t=t_batch, features=self.cond, x_future=self.x_future)
            pred = x_req + (1.0 - t_bcast) * v_here  
            loss_accum = 0.0
            denom = (obs_mask.sum() + 1e-8) if obs_mask is not None else torch.tensor(pred.numel(), device=x.device, dtype=x.dtype)
            for qi in q:
                loss_q = quantile_loss_pinball(pred, y_obs, qi)
                if obs_mask is not None:
                    loss_q = loss_q * obs_mask
                loss_accum = loss_accum + loss_q.sum() / denom
            E = loss_accum / q.numel()
            score = -torch.autograd.grad(E, x_req, retain_graph=False, create_graph=False)[0]
        schedule = self.schedule  
        tb = t_batch.view(B, *([1] * (x.ndim - 1)))  
        if schedule is None: 
            sch = tb * (1.0 - 0.5 * tb)  
        else:
            sch = schedule(tb)  
        final_v = v - self.guidance_scale * sch * score 
        return final_v.detach()
class VelocityFieldWrapper(torch.nn.Module):
    def __init__(self, denoiser, his_csi):
        super().__init__()
        self.denoiser = denoiser
        self.his_csi = his_csi

    def forward(self, t, x, args=None, kwargs=None):
        B = x.shape[0]
        if torch.is_tensor(t):
            t_batch = t.to(dtype=x.dtype, device=x.device).expand(B)
        else:
            t_batch = torch.full((B,), float(t), device=x.device, dtype=x.dtype)
        return self.denoiser(sample=x, timestep=t_batch, his_seq=self.his_csi)



def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# ----------------- RectifiedFlow -----------------
class Flow(LightningModule):
    def __init__(
        self,
        rnn_checkpoint_path: str,
        lr=5e-4,
        weight_decay=1e-3,
        betas=(0.9, 0.95),
        num_flow_steps=50,
        ema_decay=0.9999,
        test_results_path=None,
        matcher_type="base", 
        matcher_sigma=0.0, 
        scaler: SolarScaler = None,
        input_dim=1,
        hidden_dim=256,
        output_dim=1,
        step_emb=100,
        num_residual_blocks=8,
        num_features=14,
        use_ode_infer: bool = False,     
        ode_solver: str = "euler",       
        guidance_scale: float = 0,  
        
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['scaler'])
        self.scaler = scaler 
        self.sigma = 0.1      
        self.sigma_min = 1e-4
        print(f"Loading pre-trained RNN from {rnn_checkpoint_path}...")
        self.rnn_model = Model.load_from_checkpoint(rnn_checkpoint_path, scaler=scaler, test_results_path=test_results_path)
        self.rnn_model.eval()
        self.rnn_model.freeze() # Ensure no gradients flow to RNN
        print("RNN loaded and frozen.")
        # --------- UNet（velocity field） ---------
        self.model = BackboneModel(
            input_dim=input_dim, 
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            step_emb=step_emb,
            num_residual_blocks=num_residual_blocks,
            num_features=num_features
            )
        self.num_flow_steps = num_flow_steps

        self.ema = EMA(self.model.parameters(), decay=ema_decay) if self.ema_wanted else None

        

        self.matcher_type = matcher_type.lower()
        self.matcher_sigma = float(matcher_sigma)
        self.interpolant = Interpolant(sigma_coef=self.matcher_sigma, beta_fn="t^2")
        self.flow_matcher = self._build_flow_matcher(self.matcher_type, self.matcher_sigma)
        self.test_results_path = test_results_path
        self.test_outputs = []
        self.use_ode_infer = use_ode_infer        
        self.ode_solver = ode_solver          
        self.guidance_scale_default = float(guidance_scale)  

    # --------- Flow matcher ---------
    def _build_flow_matcher(self, matcher_type: str, sigma: float):
        mt = matcher_type.lower()
        if mt == "base":
            return ConditionalFlowMatcher(sigma=sigma)
        elif mt == "target":
            return TargetConditionalFlowMatcher(sigma=sigma)
        elif mt == "sb":
        
            return SchrodingerBridgeConditionalFlowMatcher(
                sigma=max(1e-3, sigma if sigma > 0 else 0.1)
            )
        elif mt == "exact":
            return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        else:
            raise ValueError(f"Unknown matcher_type: {matcher_type}. Choose from base|target|sb|exact.")

    # --------- EMA ---------
    @property
    def ema_wanted(self):
        return self.hparams.ema_decay != -1

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_wanted:
            checkpoint["ema"] = self.ema.state_dict()
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_wanted and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
        return super().on_load_checkpoint(checkpoint)

    def on_before_zero_grad(self, optimizer) -> None:
        if self.ema_wanted:
            self.ema.update(self.model.parameters())
        return super().on_before_zero_grad(optimizer)

    def to(self, *args, **kwargs):
        if self.ema_wanted:
            self.ema.to(*args, **kwargs)
        self.rnn_model.to(*args, **kwargs) 
        return super().to(*args, **kwargs)

    @contextmanager
    def maybe_ema(self):
        ema = self.ema
        ctx = nullcontext if ema is None else ema.average_parameters
        yield ctx

    def forward(self, x_t, timestep, x_con, x_future):
        return self.model(input=x_t, t=timestep, features=x_con, x_future=x_future)
    def get_rnn_prior(self, batch):
        with torch.no_grad():
            x_nsrdb = batch['x_nsrdb']
            x_hrrr = batch['x_hrrr']
            rnn_out = self.rnn_model(x_nsrdb, x_hrrr)
        return rnn_out
    @staticmethod
    def stratified_uniform(bs, group=0, groups=1, dtype=None, device=None):
        if groups <= 0:
            raise ValueError(f"groups must be positive, got {groups}")
        if group < 0 or group >= groups:
            raise ValueError(f"group must be in [0, {groups})")
        n = bs * groups
        offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
        u = torch.rand(bs, dtype=dtype, device=device)
        return ((offsets + u) / n).view(bs, 1, 1)

    def generate_random_t(self, bs, dtype=None):
        return self.stratified_uniform(bs, self.trainer.global_rank, self.trainer.world_size, dtype=dtype,
                                           device=self.device) * (1.0 - self.hparams.eps) + self.hparams.eps


    def get_paths_and_targets(self, target, source, t, mask=None):
        desired_shape = [target.shape[0]] + [1] * (target.ndim - 1)
        timestamps = t.view(desired_shape)
        target_latents = target
        reference_latents = source
        noise = torch.randn_like(target_latents)
        if mask is not None:
            noise = noise * mask
        sigma = self.sigma
        sigma_min = self.sigma_min
        interpolated_vectors = timestamps * target_latents + (1 - timestamps) * reference_latents
    
        term_std = torch.sqrt(timestamps * (1 - timestamps) * sigma**2 + sigma_min**2)
        input_latents = term_std * noise + interpolated_vectors
        numerator = 0.5 * (sigma**2) * (1 - 2 * timestamps) * (input_latents - interpolated_vectors)
        denominator = sigma**2 * timestamps * (1 - timestamps) + sigma_min**2
        target_vectors = (target_latents - reference_latents) + numerator / denominator
        return input_latents, target_vectors


    def training_step(self, batch, batch_idx):

        x_con = batch['x_nsrdb']
        x_0 = batch['x']
        target = batch['y']
        with torch.no_grad():
            x_hrrr_raw = batch['x_raw']
            mask = (x_hrrr_raw > 5.0).float()
        t = self.generate_random_t(target.shape[0], dtype=target.dtype)
        rnn_pred = self.get_rnn_prior(batch) 
        prior_noise = torch.randn_like(rnn_pred)
        prior_noise = prior_noise * mask
        source_dist_samples = 1 * rnn_pred + 0.1 * prior_noise
        # source_dist_samples = 1 * rnn_pred + 0.1 * torch.randn_like(rnn_pred)
        x_t, v_target = self.get_paths_and_targets(target, source_dist_samples, t, mask=mask)

        # x_t = t * target + (1.0 - t) * source_dist_samples
        t = t.view(-1)
        v_pred = self(x_t, t, x_con, rnn_pred)
        loss_elementwise = F.mse_loss(v_pred, v_target, reduction="none")
        loss = (loss_elementwise * mask).sum() / (mask.sum() + 1e-8)
        #loss = F.mse_loss(v_pred, v_target, reduction="mean")
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        return loss
    @torch.no_grad()
    def generate_reconstructions(self, x0, x, y, num_flow_steps, result_device, x_future):
        with self.maybe_ema():
            source_dist_samples = x0
            dt = (1.0 / num_flow_steps) * (1.0 - self.hparams.eps)
            x_t_next = source_dist_samples.clone()
            x_t_seq = [x_t_next]
            t_one = torch.ones(x.shape[0], device=self.device)
            for i in range(num_flow_steps):
                num_t = (i / num_flow_steps) * (1.0 - self.hparams.eps) + self.hparams.eps
                v_t_next = self(x_t_next, t_one * num_t, y, x_future).to(x_t_next.dtype)
                x_t_next = x_t_next.clone() + v_t_next * dt
                x_t_seq.append(x_t_next.to(result_device))
            xhat = x_t_seq[-1].to(torch.float32)
            source_dist_samples = source_dist_samples.to(result_device)
            return xhat.to(result_device), x_t_seq, source_dist_samples
    def validation_step(self, batch, batch_idx):
        x_con = batch['x_nsrdb']
        x_0 = batch['x']
        target = batch['y']
        with torch.no_grad():
            x_hrrr_raw = batch['x_raw']
            mask = (x_hrrr_raw > 5.0).float()
        t = self.generate_random_t(target.shape[0], dtype=target.dtype)
        rnn_pred = self.get_rnn_prior(batch) 
        prior_noise = torch.randn_like(rnn_pred)
        prior_noise = prior_noise * mask # Mask noise
        source_dist_samples = 1 * rnn_pred + 0.1 * prior_noise
        x_t, v_target = self.get_paths_and_targets(target, source_dist_samples, t, mask=mask)
        # source_dist_samples = 1 * rnn_pred + 0.1 * torch.randn_like(rnn_pred)
        #x_t = t * target + (1.0 - t) * source_dist_samples
        # x_t, v_target = self.get_paths_and_targets(target, source_dist_samples, t)
        t = t.view(-1)
        v_pred = self(x_t, t, x_con, rnn_pred)
        loss_elementwise = F.mse_loss(v_pred, v_target, reduction="none")
        loss = (loss_elementwise * mask).sum() / (mask.sum() + 1e-8)
        num_steps = int(self.num_flow_steps) 
        with torch.enable_grad():
            with self.maybe_ema():
                y_hat_norm, _, _ = self.generate_reconstructions(
                    x0=source_dist_samples,
                    x=target,
                    y=x_con,
                    x_future=rnn_pred, 
                    num_flow_steps=num_steps,
                    result_device=self.device,
                    )
        rec_loss = F.mse_loss(y_hat_norm, target, reduction="mean")
        self.log(
            "val_loss",
            rec_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return rec_loss

    def sample_with_ode(   
        self,
        x0: torch.Tensor,          # (B, ...)
        x_con: torch.Tensor,       # (B, features, ...)
        steps: int = 50,
        solver: str = "rk4",
        guidance_scale: float = 0.0,
        y_obs: Optional[torch.Tensor] = None,     # (B, ...)
        obs_mask: Optional[torch.Tensor] = None,  # (B, ...)
        quantiles: Optional[torch.Tensor] = None, # 1D
        schedule: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_trajectory: bool = False,
        device_out: Optional[torch.device] = None,
    ):
        device_out = device_out or torch.device("cpu")
        x0 = x0.to(self.device)
        x_con = x_con.to(self.device)
        if y_obs is not None:
            y_obs = y_obs.to(self.device)
        if obs_mask is not None:
            obs_mask = obs_mask.to(self.device)
        vf = GuidedVF(self.model, x_con, x0, guidance_scale=float(guidance_scale)) 
        if guidance_scale > 0.0:
            vf.y_obs = y_obs                     
            vf.obs_mask = obs_mask               
            vf.quantiles = (quantiles.detach().cpu().tolist()
                            if isinstance(quantiles, torch.Tensor) else quantiles)  
            vf.schedule = schedule               

        node = NeuralODE(vf, solver=solver)                                     
        t_span = torch.linspace(0.0, 1.0, steps + 1, device=self.device)

        if return_trajectory:
            traj = node.trajectory(x0, t_span)                                  
            xT = traj[-1]
            traj_cpu = [z.to(device_out).detach().to(torch.float32) for z in traj]
            return xT.to(device_out).to(torch.float32), traj_cpu
        else:
            xT = node.trajectory(x0, t_span)[-1]                                 
            return xT.to(device_out).to(torch.float32)

    def test_step(self, batch, batch_idx):
        if self.scaler is None:
            if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                self.scaler = self.trainer.datamodule.scaler
            else:
                raise ValueError("Scaler is missing! Ensure DataModule is attached to Trainer.")
        x_con = batch['x_nsrdb']
        x_0 = batch['x']
        x_sta = batch['x_static']
        target = batch['y']
        with torch.no_grad():
            x_hrrr_raw = batch['x_raw']
            mask = (x_hrrr_raw > 5.0).float() 
        with torch.no_grad():
            t = self.generate_random_t(target.shape[0], dtype=target.dtype)
            rnn_pred = self.get_rnn_prior(batch) 
            noise = torch.randn_like(rnn_pred)
            noise = noise * mask  
            
            source_dist_samples = 1 * rnn_pred + 0.1 * noise
            # source_dist_samples = 1 * rnn_pred + 0.1 * torch.randn_like(rnn_pred)
        x1 = target
        num_steps = int(self.num_flow_steps) 
        with torch.enable_grad():
            with self.maybe_ema():
                y_hat_norm, _, _ = self.generate_reconstructions(
                    x0=source_dist_samples,
                    x=x1,
                    y=x_con,
                    x_future=rnn_pred, 
                    num_flow_steps=num_steps,
                    result_device=torch.device("cpu"),
                    )
        y_hat_norm = y_hat_norm.detach()
        y_pred = self.scaler.inverse_transform(y_hat_norm, 'surfrad')
        y_pred = y_pred * mask.cpu() 
        y_true = batch['y_raw'].to('cpu')
        stations = []
        years = []
        day_idxs = []
        for m in batch['meta']: # list of strings
            s, y, d = m.split('-')
            stations.append(s)
            years.append(int(y))
            day_idxs.append(int(d))
        self.test_outputs.append({
            'y_pred': y_pred.cpu().numpy(), # (Batch, Seq, 1)
            'y_true': y_true.cpu().numpy(), # (Batch, Seq, 1)
            'stations': stations,           # (Batch,) - list/tuple
            'years': np.asarray(years, dtype=int)                 # (Batch,) - numpy array
        })
        test_recon_mse = ((y_pred - y_true) ** 2).mean()
        self.log("test/loss_recon", test_recon_mse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"test/loss_vec": test_recon_mse.detach()}
    def on_test_epoch_end(self):
        preds_arr = np.concatenate([x['y_pred'] for x in self.test_outputs], axis=0)  # (N, T, 1)
        trues_arr = np.concatenate([x['y_true'] for x in self.test_outputs], axis=0)  # (N, T, 1)
        all_stations = []
        all_years = []
        for x in self.test_outputs:
            all_stations.extend(x['stations'])  # list of str, len = N
            all_years.extend(x['years'])        # numpy array or list, len = N
        all_stations = np.array(all_stations)   # (N,)
        all_years = np.array(all_years)         # (N,)
        if preds_arr.ndim == 3 and preds_arr.shape[-1] == 1:
            preds_flat_full = preds_arr.reshape(-1)  # (N*T,)
        else:
            preds_flat_full = preds_arr.reshape(-1)

        if trues_arr.ndim == 3 and trues_arr.shape[-1] == 1:
            trues_flat_full = trues_arr.reshape(-1)  # (N*T,)
        else:
            trues_flat_full = trues_arr.reshape(-1)

        N, T = preds_arr.shape[0], preds_arr.shape[1] if preds_arr.ndim >= 2 else 1
        stations_flat_full = np.repeat(all_stations, T)  # (N*T,)
        years_flat_full = np.repeat(all_years, T)        # (N*T,)

        df_full = pd.DataFrame({
        'Station': stations_flat_full,
        'Year': years_flat_full,
        'True': trues_flat_full,
        'Pred': preds_flat_full
        })

        def calc_metrics(g):
            rmse = np.sqrt(np.mean((g['Pred'] - g['True'])**2))
            mae = np.mean(np.abs(g['Pred'] - g['True']))
            return pd.Series({'RMSE': rmse, 'MAE': mae, 'Count': len(g)})

        station_metrics_full = df_full.groupby('Station').apply(calc_metrics).sort_values('RMSE')

        if not os.path.exists(self.test_results_path):
            os.makedirs(self.test_results_path, exist_ok=True)
        path_metrics_full = os.path.join(self.test_results_path, "test_metrics_by_station_full.csv")
        station_metrics_full.to_csv(path_metrics_full)
        print(f"Saved UNFILTERED metrics summary to: {path_metrics_full}")

        stations_dir = os.path.join(self.test_results_path, "station_predictions")
        if not os.path.exists(stations_dir):
            os.makedirs(stations_dir, exist_ok=True)
            print(f"Created directory for individual stations: {stations_dir}")
        print("Saving individual station files (UNFILTERED)...")
        unique_stations = df_full['Station'].unique()
        for station in unique_stations:
            station_df = df_full[df_full['Station'] == station].copy()
            safe_station_name = str(station).replace('/', '_').replace('\\', '_')
            file_path = os.path.join(stations_dir, f"station_{safe_station_name}.csv")
            station_df.to_csv(file_path, index=False)
        print(f"Successfully saved {len(unique_stations)} individual station files in {stations_dir}")
        mask_nonzero = trues_flat_full != 0  # (N*T,)
        if mask_nonzero.any():
            df_filtered = df_full[mask_nonzero].copy()
            station_metrics_filtered = df_filtered.groupby('Station').apply(calc_metrics).sort_values('RMSE')

            path_metrics_filtered = os.path.join(self.test_results_path, "test_metrics_by_station_filtered_nonzero.csv")
            station_metrics_filtered.to_csv(path_metrics_filtered)
            print(f"Saved FILTERED (True!=0) metrics summary to: {path_metrics_filtered}")

            stations_dir_filtered = os.path.join(self.test_results_path, "station_predictions_filtered_nonzero")
            if not os.path.exists(stations_dir_filtered):
                os.makedirs(stations_dir_filtered, exist_ok=True)
                print(f"Created directory for filtered station files: {stations_dir_filtered}")
            print("Saving individual station files (FILTERED; True!=0)...")
            for station, g in df_filtered.groupby('Station'):
                safe_station_name = str(station).replace('/', '_').replace('\\', '_')
                file_path = os.path.join(stations_dir_filtered, f"station_{safe_station_name}.csv")
                g.to_csv(file_path, index=False)
            print(f"Successfully saved {df_filtered['Station'].nunique()} filtered station files in {stations_dir_filtered}")
        else:
            print("Warning: After filtering (True!=0), no samples remain. Skipping filtered metrics.")
        self.test_outputs.clear()
  
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(),
                          betas=self.hparams.betas,
                          eps=1e-8,
                          lr=self.hparams.lr,
                          weight_decay=self.hparams.weight_decay)
        return optimizer