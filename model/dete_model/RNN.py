import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
from pytorch_lightning.loggers import TensorBoardLogger
from dataload.dataset import SolarScaler

class SolarSeq2Seq(pl.LightningModule):
    def __init__(self, 
                 input_dim: int,         
                 decoder_input_dim: int, 
                 static_dim: int = 2,  
                lr=5e-4,
                weight_decay=1e-3,
                betas=(0.9, 0.95),
                 hidden_dim: int = 64, 
                 output_dim: int = 1,
                 test_results_path=None,
                 scaler: SolarScaler = None,
                *args,
                **kwargs
                ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['scaler'])
        self.scaler = scaler 
        self.test_results_path = test_results_path
        self.encoder = nn.LSTM(
            input_size=input_dim + static_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=0.1
        )
        self.decoder = nn.LSTM(
            input_size=decoder_input_dim + static_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=0.1
        )
        
        self.regressor = nn.Linear(hidden_dim, output_dim)
        self.loss_fn = nn.MSELoss()
        self.test_outputs = []

    def forward(self, x_hist, x_future, x_static):
        """
        x_hist: (Batch, Seq_in, Feat_in)
        x_future: (Batch, Seq_out, Feat_out)
        x_static: (Batch, 2)
        """
        batch_size, seq_in, _ = x_hist.shape
        seq_out = x_future.shape[1]
        static_enc = x_static.unsqueeze(1).repeat(1, seq_in, 1)
        static_dec = x_static.unsqueeze(1).repeat(1, seq_out, 1)
        enc_input = torch.cat([x_hist, static_enc], dim=-1)
        dec_input = torch.cat([x_future, static_dec], dim=-1)
        _, (h_n, c_n) = self.encoder(enc_input)
        dec_out, _ = self.decoder(dec_input, (h_n, c_n)) 
        
        return self.regressor(dec_out)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch['x_nsrdb'], batch['x_hrrr'], batch['x_static'])
        loss = self.loss_fn(y_hat, batch['y'])
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch['x_nsrdb'], batch['x_hrrr'], batch['x_static'])
        loss = self.loss_fn(y_hat, batch['y'])
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def test_step(self, batch, batch_idx):
        if self.scaler is None:
            if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                self.scaler = self.trainer.datamodule.scaler
            else:
                raise ValueError("Scaler is missing! Ensure DataModule is attached to Trainer.")

        y_hat_norm = self(batch['x_nsrdb'], batch['x_hrrr'], batch['x_static'])
        y_pred = self.scaler.inverse_transform(y_hat_norm, 'surfrad')
        y_true = batch['y_raw'] 
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
        return None 
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
        optimizer = torch.optim.Adam(self.parameters(),
                          eps=1e-8,
                          lr=self.hparams.lr)
        return optimizer
