import torch
from torch import nn
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
from model.dete_model.layers.loss import SoftDTWLossPyTorch
from model.dete_model.layers.TAU import Encoder, CrossDeformAttn
from model.dete_model.layers.Embed import Deform_Temporal_Embedding, Local_Temporal_Embedding, PositionalEmbedding
from math import ceil
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
class Layernorm(nn.Module):
    def __init__(self, dim):
        super(Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        x_hat = self.layernorm(x)
        return x_hat

class Model(pl.LightningModule):
    def __init__(self, 
                 input_len: int = 288,
                 pred_len: int = 96,
                 e_layers: int = 1,
                 d_layers: int = 1,
                 d_model: int = 64,
                 enc_in: int = 14,
                 dec_in: int = 9,
                lr=1e-3,
                weight_decay=1e-3,
                betas=(0.9, 0.95),
                c_out: int = 1,         
                kernel: int=3,
                dropout: float= 0.0,
                n_heads: int = 4,
                n_reshape: int = 6,
                patch_len: int = 6,
                stride: int = 3,
                scaler: SolarScaler = None,
                test_results_path=None,
                *args,
                **kwargs
                ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['scaler'])
        self.input_len = input_len
        self.scaler = scaler 
        self.pred_len = pred_len
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_model = d_model
        self.f_dim = c_out
        self.dropout = dropout
        self.n_heads = n_heads
        self.kernel_size = kernel
        self.loss_fn = SoftDTWLossPyTorch()
        self.test_outputs = []
        self.test_results_path = test_results_path

        # Embedding
        if enc_in == 1:
            self.enc_value_embedding = Deform_Temporal_Embedding(self.f_dim, self.d_model, freq='d')
        else:
            self.s_group = 4
            assert self.d_model % self.s_group == 0
            # Embedding local patches
            self.pad_in_len = ceil(1.0 * enc_in / self.s_group) * self.s_group
            self.enc_value_embedding = Local_Temporal_Embedding(self.pad_in_len//self.s_group, self.d_model, self.pad_in_len-enc_in, self.s_group)

        self.pre_norm = nn.LayerNorm(d_model)
        # Encoder
        n_days = [1,n_reshape,n_reshape]
        assert len(n_days) > self.e_layers-1
        drop_path_rate=dropout
        dpr = [x.item() for x in torch.linspace(drop_path_rate, drop_path_rate, self.e_layers)]
        self.encoder = Encoder(
            [
                CrossDeformAttn(seq_len=self.input_len, 
                                d_model=self.d_model, 
                                n_heads=self.n_heads, 
                                dropout=dropout, 
                                droprate=dpr[l], 
                                n_days=n_days[l], 
                                window_size=kernel, 
                                patch_len=patch_len, 
                                stride=stride) for l in range(e_layers)
            ],
            norm_layer=Layernorm(d_model)
        )

        self.dec_embedding = Deform_Temporal_Embedding(
            d_inp=dec_in, 
            d_model=d_model, 
            freq='h',
            dropout=dropout
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 8, 
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=d_layers)
        # MLP layer
        self.fc = nn.Sequential(
            nn.Linear(self.input_len, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.pred_len)
        )

        # Projection layer
        self.projection = nn.Linear(self.d_model, self.f_dim)

    def forward(self, x_enc, x_future):
        x_enc = self.enc_value_embedding(x_enc)
        x_enc = self.pre_norm(x_enc)
        enc_out, _ = self.encoder(x_enc) 
        tgt = self.dec_embedding(x_future)
        dec_out = self.decoder(tgt, enc_out) 
        out = self.projection(dec_out)
        return out

    def training_step(self, batch, batch_idx):
        y_hat = self(batch['x_nsrdb'], batch['x_hrrr'])
        loss = self.loss_fn(y_hat, batch['y'])
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch['x_nsrdb'], batch['x_hrrr'])
        loss = self.loss_fn(y_hat, batch['y'])
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def test_step(self, batch, batch_idx):
        if self.scaler is None:
            if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                self.scaler = self.trainer.datamodule.scaler
            else:
                raise ValueError("Scaler is missing! Ensure DataModule is attached to Trainer.")

        y_hat_norm = self(batch['x_nsrdb'], batch['x_hrrr'])
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

