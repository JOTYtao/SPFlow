import os
import yaml
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional

# =========================
# 1) TimeEncoder
# =========================
class TimeEncoder:
    def transform(self, time_strs: np.ndarray) -> np.ndarray:
        dt = pd.to_datetime(time_strs)
        month = dt.month.values
        day = dt.day.values
        hour = dt.hour.values
        minute = dt.minute.values

        feats = []
        # Month (1-12)
        feats.append(np.sin(2 * np.pi * (month - 1) / 11))
        feats.append(np.cos(2 * np.pi * (month - 1) / 11))
        # Day (1-31)
        feats.append(np.sin(2 * np.pi * (day - 1) / 30))
        feats.append(np.cos(2 * np.pi * (day - 1) / 30))
        # Hour (0-23)
        feats.append(np.sin(2 * np.pi * hour / 23))
        feats.append(np.cos(2 * np.pi * hour / 23))
        # Minute (0-59)
        feats.append(np.sin(2 * np.pi * minute / 59))
        feats.append(np.cos(2 * np.pi * minute / 59))

        out = np.stack(feats, axis=1).astype(np.float32)
        if not out.flags["C_CONTIGUOUS"]:
            out = np.ascontiguousarray(out)
        return out

# =========================
# 2) SolarScaler
# =========================
class SolarScaler:
    def __init__(self, feature_cols: List[str], target_col: str, station_coords: Dict[str, List[float]]):
        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.station_coords = station_coords

        coords = np.array(list(station_coords.values()))
        self.coord_mean = torch.from_numpy(coords.mean(axis=0)).float()
        self.coord_std = torch.from_numpy(coords.std(axis=0)).float()
        self.coord_std[self.coord_std < 1e-6] = 1.0

    def fit(self, root_paths: Dict, stations: List[str], years: List[int]):
        print("Fitting scaler on training data...")
        data_cache = {'nsrdb': [], 'hrrr': [], 'surfrad': []}

        for station in stations:
            for year in years:
                # NSRDB
                path_n = os.path.join(root_paths['nsrdb'], station, f"{year}.csv")
                if os.path.exists(path_n):
                    df = pd.read_csv(path_n, usecols=self.feature_cols)
                    df = df[self.feature_cols]
                    arr = df.values.astype(np.float32)
                    if not arr.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arr)
                    data_cache['nsrdb'].append(arr)

                # HRRR
                path_h = os.path.join(root_paths['hrrr'], station, f"{year}.csv")
                if os.path.exists(path_h):
                    df = pd.read_csv(path_h, usecols=[self.target_col])
                    arr = df.values.astype(np.float32)
                    if not arr.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arr)
                    data_cache['hrrr'].append(arr)

                # SURFRAD
                path_s = os.path.join(root_paths['surfrad'], station, f"{year}.csv")
                if os.path.exists(path_s):
                    df = pd.read_csv(path_s, usecols=[self.target_col])
                    arr = df.values.astype(np.float32)
                    if not arr.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arr)
                    data_cache['surfrad'].append(arr)

        for source in ['nsrdb', 'hrrr', 'surfrad']:
            if not data_cache[source]:
                print(f"Warning: No data found for {source}, skipping fit.")
                continue
            all_data = np.concatenate(data_cache[source], axis=0)
            mean = np.nanmean(all_data, axis=0)
            std = np.nanstd(all_data, axis=0)
            std = np.where(std < 1e-6, 1.0, std)
            self.stats[source] = {
                'mean': torch.from_numpy(mean).float(),
                'std': torch.from_numpy(std).float()
            }
            print(f"[{source}] fitted: mean[:3]={mean[:3]}")

    def save(self, path: str):
        save_dict = dict(self.stats)
        save_dict['coord_mean'] = self.coord_mean
        save_dict['coord_std'] = self.coord_std
        torch.save(save_dict, path)

    def load(self, path: str):
        loaded = torch.load(path, map_location="cpu")
        self.coord_mean = loaded.pop('coord_mean')
        self.coord_std = loaded.pop('coord_std')
        self.stats = loaded

    @torch.no_grad()
    def transform(self, data: torch.Tensor, source: str) -> torch.Tensor:
        mean = self.stats[source]['mean'].to(data.device)
        std = self.stats[source]['std'].to(data.device)
        return (data - mean) / std

    @torch.no_grad()
    def inverse_transform(self, data: torch.Tensor, source: str) -> torch.Tensor:
        mean = self.stats[source]['mean'].to(data.device)
        std = self.stats[source]['std'].to(data.device)
        return data * std + mean

    @torch.no_grad()
    def transform_coords(self, coords: torch.Tensor) -> torch.Tensor:
        return (coords - self.coord_mean.to(coords.device)) / self.coord_std.to(coords.device)

# =========================
# 3) SolarDataset（基于时间戳内连接对齐）
# =========================
class SolarDataset(Dataset):
    def __init__(
        self,
        data_index: List[Dict],
        root_paths: Dict[str, str],
        scaler: SolarScaler,
        feature_cols: List[str],
        target_col: str,
        input_len: int,
        pred_len: int,
        station_coords: Dict[str, List[float]],
        steps_per_day: int = 96,
    ):
        self.data_index = data_index
        self.root_paths = root_paths
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.input_len = input_len
        self.pred_len = pred_len
        self.station_coords = station_coords
        self.steps_per_day = steps_per_day

        self.time_encoder = TimeEncoder()
        self.cache: Dict[str, Dict[str, np.ndarray]] = {}

    def _load_data(self, station: str, year: int) -> Dict[str, np.ndarray]:
        key = f"{station}_{year}"
        if key in self.cache:
            return self.cache[key]

        path_n = os.path.join(self.root_paths['nsrdb'], station, f"{year}.csv")
        path_h = os.path.join(self.root_paths['hrrr'], station, f"{year}.csv")
        path_s = os.path.join(self.root_paths['surfrad'], station, f"{year}.csv")
        if not (os.path.exists(path_n) and os.path.exists(path_h) and os.path.exists(path_s)):
            raise FileNotFoundError(f"Missing CSV for {station}-{year}")

        # 读取并标准化 Time 列
        df_n = pd.read_csv(path_n)
        df_h = pd.read_csv(path_h)
        df_s = pd.read_csv(path_s)

        for name, df in [('NSRDB', df_n), ('HRRR', df_h), ('SURFRAD', df_s)]:
            if 'Time' not in df.columns:
                raise KeyError(f"{name} file missing Time column for {station}-{year}")

        for df in (df_n, df_h, df_s):
            df['Time'] = pd.to_datetime(df['Time'])

        # 仅保留需要的列
        df_n = df_n[['Time'] + self.feature_cols].copy()
        df_h = df_h[['Time', self.target_col]].copy()
        df_s = df_s[['Time', self.target_col]].copy()

        # 对齐：NSRDB ⨝ HRRR ⨝ SURFRAD（内连接确保三源都存在）
        df = pd.merge(df_n, df_h, on='Time', how='inner', suffixes=('', '_hrrr'))
        # 第二次 merge 会把 SURFRAD 的同名 target 列重命名为 target_col_surfrad
        df = pd.merge(df, df_s, on='Time', how='inner', suffixes=('', '_surfrad'))

        # 排序
        df = df.sort_values('Time').reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No overlapping timestamps after aligning for {station}-{year}")

        # 提取数组
        raw_time_strs = df['Time'].astype(str).values

        nsrdb_phys = df[self.feature_cols].values.astype(np.float32)
        if not nsrdb_phys.flags['C_CONTIGUOUS']:
            nsrdb_phys = np.ascontiguousarray(nsrdb_phys)

        time_embeds = self.time_encoder.transform(raw_time_strs)  # contiguous

        # HRRR 列名仍为 target_col
        hrrr = df[self.target_col].values.astype(np.float32).reshape(-1, 1)
        if not hrrr.flags['C_CONTIGUOUS']:
            hrrr = np.ascontiguousarray(hrrr)

        # SURFRAD 列名为 target_col_surfrad
        surf_col = self.target_col + '_surfrad'
        if surf_col not in df.columns:
            # 如果 pandas 的后缀命名不一致，退而求其次：找到第二个同名列
            # 但通常 suffixes 会生成 surf_col
            raise KeyError(f"Expected column '{surf_col}' not found for {station}-{year}")
        surfrad = df[surf_col].values.astype(np.float32).reshape(-1, 1)
        if not surfrad.flags['C_CONTIGUOUS']:
            surfrad = np.ascontiguousarray(surfrad)

        # 一致性检查
        n = len(df)
        assert nsrdb_phys.shape[0] == n
        assert time_embeds.shape[0] == n
        assert hrrr.shape[0] == n
        assert surfrad.shape[0] == n

        self.cache[key] = {
            'nsrdb_phys': nsrdb_phys,      # [N, Fp]
            'time_embeds': time_embeds,     # [N, Ft]
            'raw_time_strs': raw_time_strs,
            'hrrr': hrrr,                  # [N, 1]
            'surfrad': surfrad             # [N, 1]
        }
        return self.cache[key]

    def __len__(self) -> int:
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data_index[idx]
        station, year, day_idx = item['station'], item['year'], item['day_idx']

        data = self._load_data(station, year)

        steps_per_day = self.steps_per_day
        in_start = day_idx * steps_per_day
        in_end = in_start + self.input_len
        out_start = (day_idx + 1) * steps_per_day
        out_end = out_start + self.pred_len

        total_len = data['nsrdb_phys'].shape[0]
        if in_end > total_len or out_end > total_len:
            raise IndexError(
                f"Window out of bounds for {station} {year} day_idx={day_idx} "
                f"(total_len={total_len}, in=[{in_start},{in_end}), out=[{out_start},{out_end}))"
            )

        # NSRDB features + time
        x_phys_np = data['nsrdb_phys'][in_start:in_end]    # [input_len, Fp]
        x_time_np = data['time_embeds'][in_start:in_end]    # [input_len, Ft]
        x_phys = torch.from_numpy(np.ascontiguousarray(x_phys_np)).to(torch.float32)
        x_time = torch.from_numpy(np.ascontiguousarray(x_time_np)).to(torch.float32)
        x_phys = self.scaler.transform(x_phys, 'nsrdb')
        x_nsrdb_combined = torch.cat([x_phys, x_time], dim=-1).contiguous()  # [input_len, Fp+Ft]

        # HRRR future
        x_time_out_np = data['time_embeds'][out_start:out_end] # [pred_len, Ft]
        x_time_out = torch.from_numpy(np.ascontiguousarray(x_time_out_np)).to(torch.float32)

        x_hrrr_np = data['hrrr'][out_start:out_end]        # [pred_len, 1]
        x_hrrr = torch.from_numpy(np.ascontiguousarray(x_hrrr_np)).to(torch.float32)
        x_hrrr_val = self.scaler.transform(x_hrrr, 'hrrr').contiguous()
        x_hrrr_combined = torch.cat([x_hrrr_val, x_time_out], dim=-1).contiguous()
        # SURFRAD target
        y_np = data['surfrad'][out_start:out_end]          # [pred_len, 1]
        y = torch.from_numpy(np.ascontiguousarray(y_np)).to(torch.float32)
        y_norm = self.scaler.transform(y, 'surfrad').contiguous()

        # static coords
        coords = torch.tensor(self.station_coords[station], dtype=torch.float32)
        x_static = self.scaler.transform_coords(coords).to(torch.float32).contiguous()  # [2]

        # meta 用简单字符串，避免嵌套 dict 的 collate 风险
        meta = f"{station}-{year}-{day_idx}"

        return {
            'x_nsrdb': x_nsrdb_combined,  # [input_len, F]
            'x_hrrr': x_hrrr_combined,             # [pred_len, 1]
            'x': x_hrrr_val,
            'x_static': x_static,         # [2]
            'y': y_norm,                  # [pred_len, 1]
            'y_raw': y,                   # [pred_len, 1]
            'x_raw': x_hrrr,
            'meta': meta                  # str；默认 collate 保持为 list
        }

# =========================
# 4) 可选：安全 collate（调试时启用）
# =========================
def safe_collate(batch: List[Dict]):
    from collections import defaultdict
    out = defaultdict(list)
    for sample in batch:
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                if not v.flags['C_CONTIGUOUS']:
                    v = np.ascontiguousarray(v)
                v = torch.from_numpy(v.copy())
                out[k].append(v)
            elif isinstance(v, torch.Tensor):
                if not v.is_contiguous():
                    v = v.contiguous()
                out[k].append(v)
            else:
                out[k].append(v)

    stacked = {}
    for k, vs in out.items():
        try:
            if all(isinstance(x, torch.Tensor) for x in vs):
                shapes = [tuple(x.shape) for x in vs]
                if len(set(shapes)) == 1:
                    stacked[k] = torch.stack(vs, dim=0)
                else:
                    print(f"[safe_collate] key={k} shapes not equal: {shapes}; keep as list.")
                    stacked[k] = vs
            else:
                stacked[k] = vs
        except Exception as e:
            print(f"[safe_collate] key={k} stack error: {e}; keep as list.")
            stacked[k] = vs
    return stacked

# =========================
# 5) SolarDataModule
# =========================
class SolarDataModule(pl.LightningDataModule):
    def __init__(self, config_path: str = "config.yaml", use_safe_collate: bool = False):
        super().__init__()
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)['data']

        self.root_paths = self.cfg['root_paths']
        self.feature_cols = self.cfg['nsrdb_features']
        self.target_col = self.cfg['target_col']
        self.station_coords = self.cfg['station_metadata']

        self.scaler = SolarScaler(self.feature_cols, self.target_col, self.station_coords)
        self.scaler_path = self.cfg.get('scaler_path', 'scaler.pt')

        self.train_stations = self.cfg['train_stations']
        self.test_stations = self.cfg['test_stations']
        self.all_stations = list(set(self.train_stations + self.test_stations))

        self.train_years = self.cfg['train_years']
        self.val_years = self.cfg['val_years']
        self.test_years = self.cfg['test_years']

        self.steps_per_day = self.cfg.get('steps_per_day', 96)
        self.input_len = self.cfg['input_len']
        self.pred_len = self.cfg['pred_len']
        self.batch_size = self.cfg['batch_size']
        # 调试时建议 0，确认没问题再升高到 2/4
        self.num_workers = self.cfg.get('num_workers', 0)
        self.use_safe_collate = use_safe_collate

        self.train_ds: Optional[SolarDataset] = None
        self.val_ds: Optional[SolarDataset] = None
        self.test_ds: Optional[SolarDataset] = None

    def prepare_data(self):
        if not os.path.exists(self.scaler_path):
            self.scaler.fit(self.root_paths, self.train_stations, self.train_years)
            self.scaler.save(self.scaler_path)

    def setup(self, stage: Optional[str] = None):
        self.scaler.load(self.scaler_path)

        def compute_max_day(station: str, year: int) -> int:
            # 用对齐后的 N 动态计算可用天数
            path_n = os.path.join(self.root_paths['nsrdb'], station, f"{year}.csv")
            path_h = os.path.join(self.root_paths['hrrr'], station, f"{year}.csv")
            path_s = os.path.join(self.root_paths['surfrad'], station, f"{year}.csv")
            if not (os.path.exists(path_n) and os.path.exists(path_h) and os.path.exists(path_s)):
                return 0

            # 读取时间列，并求公共交集长度
            df_n = pd.read_csv(path_n, usecols=['Time'])
            df_h = pd.read_csv(path_h, usecols=['Time'])
            df_s = pd.read_csv(path_s, usecols=['Time'])
            for df in (df_n, df_h, df_s):
                df['Time'] = pd.to_datetime(df['Time'])

            df = pd.merge(df_n, df_h, on='Time', how='inner')
            df = pd.merge(df, df_s, on='Time', how='inner')
            N = len(df)
            if N <= 0:
                return 0

            # 需要保证 in 窗口和 out 窗口都能取到
            max_day = (N - (self.input_len + self.pred_len)) // self.steps_per_day
            return max(0, max_day)

        def build_index(stations, years):
            indices = []
            for station in stations:
                for year in years:
                    max_day = compute_max_day(station, year)
                    if max_day <= 0:
                        continue
                    for d in range(max_day):
                        indices.append({'station': station, 'year': year, 'day_idx': d})
            return indices

        if stage == 'fit' or stage is None:
            train_idx = build_index(self.train_stations, self.train_years)
            val_idx = build_index(self.train_stations, self.val_years)
            self.train_ds = SolarDataset(
                train_idx, self.root_paths, self.scaler,
                self.feature_cols, self.target_col,
                self.input_len, self.pred_len,
                self.station_coords, steps_per_day=self.steps_per_day
            )
            self.val_ds = SolarDataset(
                val_idx, self.root_paths, self.scaler,
                self.feature_cols, self.target_col,
                self.input_len, self.pred_len,
                self.station_coords, steps_per_day=self.steps_per_day
            )

        if stage == 'test' or stage is None:
            test_idx = build_index(self.all_stations, self.test_years)
            self.test_ds = SolarDataset(
                test_idx, self.root_paths, self.scaler,
                self.feature_cols, self.target_col,
                self.input_len, self.pred_len,
                self.station_coords, steps_per_day=self.steps_per_day
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=(safe_collate if self.use_safe_collate else None),
            pin_memory=False,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=(safe_collate if self.use_safe_collate else None),
            pin_memory=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=(safe_collate if self.use_safe_collate else None),
            pin_memory=False,
            drop_last=False,
        )