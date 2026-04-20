# backend.py
# -*- coding: utf-8 -*-
import os, math, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
from torch import nn

# Graceful handling of PyTorch Geometric imports
HAS_TORCH_GEOMETRIC = True
try:
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.nn.models import GNNExplainer
except Exception:
    HAS_TORCH_GEOMETRIC = False
    GCNConv = None
    GATConv = None
    GNNExplainer = None
    print("Warning: torch_geometric not found. Using fallback neural network models.")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

from haversine import haversine
import requests, joblib
import streamlit as st

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------
# Constants
# -----------------------------
REGION_COORDS = {
    'chennai': (13.0827, 80.2707),
    'gopalpur': (19.2582, 84.9051),
    'machilipatnam': (16.1875, 81.1389),
    'nellore': (14.4426, 79.9865),
    'rameswaram': (9.2882, 79.3127),
    'bangalore': (12.9716, 77.5946),
    'hyderabad': (17.3850, 78.4867),
    'nagpur': (21.1458, 79.0882),
    'bhopal': (23.2599, 77.4126),
    'jaisalmer': (26.9157, 70.9083),
    'delhi': (28.7041, 77.1025),
    'ahmedabad': (23.0225, 72.5714),
    'ennore': (13.2124, 80.3249),
    'mahabalipuram': (12.6229, 80.1958)
}

IMD_COLORS = {
    0: ("No Rain", "#B0BEC5"),
    1: ("Light Rain (Green)", "#8BC34A"),
    2: ("Moderate Rain (Yellow)", "#FFEB3B"),
    3: ("Heavy Rain (Blue)", "#2196F3"),
    4: ("Very Heavy Rain (Purple)", "#7E57C2"),
    5: ("Extremely Heavy (Orange)", "#FF9800"),
    6: ("Exceptionally Heavy / Red", "#F44336")
}

def imd_bin(mm: float) -> int:
    if mm <= 0.0: return 0
    if mm <= 7.5: return 1
    if mm <= 35.5: return 2
    if mm <= 64.4: return 3
    if mm <= 115.5: return 4
    if mm <= 204.4: return 5
    return 6

def month_dummies(df: pd.DataFrame, prefix='m') -> pd.DataFrame:
    return pd.get_dummies(df['date'].dt.month, prefix=prefix, dtype=int)

# -----------------------------
# Data pipeline
# -----------------------------
def load_and_preprocess(data_dir='data', rainfall_threshold=200.0):
    region_dfs, common_dates = [], None
    for region, coords in REGION_COORDS.items():
        fp = os.path.join(data_dir, f"weather_{region}.tsv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, sep='\t')
        if 'date' not in df.columns or 'rainfall' not in df.columns:
            continue
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['region'] = region
        df['latitude'], df['longitude'] = coords
        df['distance_from_chennai'] = haversine(coords, REGION_COORDS['chennai'])
        df['month'] = df['date'].dt.month
        df['dayofyear'] = df['date'].dt.dayofyear
        df['is_nem'] = df['month'].between(10, 12).astype(int)
        df['is_swm'] = df['month'].between(6, 9).astype(int)
        df['imd_alert'] = df['rainfall'].apply(imd_bin).astype(int)
        df['flood_binary'] = (df['rainfall'] > rainfall_threshold).astype(int)
        df['rainfall_lag1'] = df['rainfall'].shift(1).bfill()
        df['rainfall_lag3'] = df['rainfall'].rolling(3, min_periods=1).mean()
        df['rainfall_lag7'] = df['rainfall'].rolling(7, min_periods=1).mean()
        df = df.ffill().fillna(0)
        region_dfs.append(df)
        common_dates = set(df['date']) if common_dates is None else (common_dates & set(df['date']))

    if not region_dfs:
        raise RuntimeError("No data files found in data_dir.")

    common_dates = sorted(list(common_dates))
    filtered = [df[df['date'].isin(common_dates)].copy() for df in region_dfs]
    combined = pd.concat(filtered, ignore_index=True).sort_values(['region','date']).reset_index(drop=True)

    m_dums = month_dummies(combined, prefix='m')
    combined = pd.concat([combined, m_dums], axis=1)

    exclude = {'date','region','latitude','longitude','distance_from_chennai','rainfall','dew_point','flood_binary','imd_alert'}
    common_cols = set(combined.columns)
    for df in filtered:
        common_cols &= set(df.columns)
    common_cols -= exclude
    temporal_cols = {'month','dayofyear','is_nem','is_swm','rainfall_lag1','rainfall_lag3','rainfall_lag7'}
    common_cols |= temporal_cols

    feature_cols = [c for c in sorted(common_cols) if str(combined[c].dtype).startswith(('float','int'))]
    for c in m_dums.columns:
        if c not in feature_cols:
            feature_cols.append(c)

    scaler_map = {}
    for region in combined['region'].unique():
        mask = combined['region'] == region
        scaler = StandardScaler()
        combined.loc[mask, feature_cols] = scaler.fit_transform(combined.loc[mask, feature_cols])
        scaler_map[region] = scaler

    return combined.reset_index(drop=True), feature_cols, scaler_map

# -----------------------------
# OpenWeather (simple cached via st.cache_data in frontend)
# -----------------------------
def fetch_openweather_all(api_key, regions=REGION_COORDS):
    results = {}
    for region, (lat, lon) in regions.items():
        try:
            url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&appid={d8eb51ec2bc5fdc841979d0e600b658e
}&units=metric"
            j = requests.get(url, timeout=10).json()
            daily = []
            for d in j.get('daily', [])[:7]:
                temp = float(d['temp'].get('day', 0.0)) if isinstance(d.get('temp'), dict) else float(d.get('temp', 0.0))
                daily.append({
                    'temp': temp,
                    'humidity': float(d.get('humidity', 0.0)),
                    'wind_speed': float(d.get('wind_speed', 0.0)),
                    'wind_deg': float(d.get('wind_deg', 0.0)),
                    'pressure': float(d.get('pressure', 1013.0)),
                    'rain': float(d.get('rain', 0.0)) if d.get('rain') is not None else 0.0
                })
            current = {
                'wind_speed': float(j.get('current', {}).get('wind_speed', 0.0)),
                'wind_deg': float(j.get('current', {}).get('wind_deg', 0.0)),
                'pressure': float(j.get('current', {}).get('pressure', 1013.0)),
                'rain': float(j.get('current', {}).get('rain', {}).get('1h', 0.0) if j.get('current', {}).get('rain') else 0.0)
            }
            results[region] = {'daily': daily, 'current': current}
        except Exception:
            results[region] = {'daily': [{'temp':0,'humidity':0,'wind_speed':0,'wind_deg':0,'pressure':1013,'rain':0}]*7,
                               'current': {'wind_speed':0,'wind_deg':0,'pressure':1013,'rain':0}}
    return results

# -----------------------------
# Graph utils
# -----------------------------
def build_region_graph_knn(k=4, device='cpu'):
    regions = list(REGION_COORDS.keys())
    coords = np.array([REGION_COORDS[r] for r in regions])
    n = len(regions)
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,j] = 1e6 if i==j else (haversine(coords[i], coords[j]) + 1e-6)
    edges=set()
    for i in range(n):
        nbrs = np.argsort(D[i])[:k+1]
        for j in nbrs:
            if i!=j:
                edges.add((i,j)); edges.add((j,i))
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    weights = [1.0 / (D[i,j]) for (i,j) in edges]
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index.to(device), edge_weight.to(device), regions

def build_dynamic_edge_weights_for_day(node_weather, alpha=0.6, beta=0.3, gamma=0.1, device='cpu'):
    regions = list(REGION_COORDS.keys())
    coords = [REGION_COORDS[r] for r in regions]
    n = len(regions)
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,j] = 1e6 if i==j else (haversine(coords[i], coords[j]) + 1e-6)
    wind_vecs, pressures = [], []
    for r in regions:
        w = node_weather.get(r, {})
        ws = float(w.get('wind_speed', 0.0))
        wd = float(w.get('wind_deg', 0.0))
        wx = ws * math.cos(math.radians(wd))
        wy = ws * math.sin(math.radians(wd))
        wind_vecs.append((wx, wy))
        pressures.append(float(w.get('pressure', 1013.0)))
    pressures = np.array(pressures)
    max_pd = max(1.0, np.max(np.abs(pressures - pressures.mean())))
    edges, weights = [], []
    for i in range(n):
        for j in range(n):
            if i == j: continue
            base = 1.0 / D[i,j]
            lat_i, lon_i = coords[i]; lat_j, lon_j = coords[j]
            dx, dy = lat_j - lat_i, lon_j - lon_i
            norm = np.hypot(dx, dy) + 1e-9
            disp = (dx/norm, dy/norm)
            wx, wy = wind_vecs[i]
            wnorm = np.hypot(wx, wy) + 1e-9
            wind_unit = (wx/wnorm, wy/wnorm) if wnorm>0 else (0.0,0.0)
            align = max(0.0, wind_unit[0]*disp[0] + wind_unit[1]*disp[1])
            press_score = abs(pressures[i] - pressures[j]) / max_pd
            w = alpha*base + beta*(base*align) + gamma*(base*press_score)
            edges.append((i,j)); weights.append(w)
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    ew = torch.tensor(weights, dtype=torch.float).to(device)
    return ei, ew

# -----------------------------
# Fallback Models (when PyTorch Geometric is not available)
# -----------------------------
class FallbackGCNConv(nn.Module):
    """Simple fallback for GCNConv when torch_geometric is not available"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        # Simple message passing approximation
        x = self.linear(x)
        return x

class FallbackGATConv(nn.Module):
    """Simple fallback for GATConv when torch_geometric is not available"""
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0, concat=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels * heads)
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        
    def forward(self, x, edge_index):
        # Simple linear transformation as fallback
        x = self.linear(x)
        if not self.concat and self.heads > 1:
            x = x.view(-1, self.heads, self.out_channels).mean(dim=1)
        return x

# -----------------------------
# Models
# -----------------------------
class RegionalEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, emb_dim=64, use_attention=True, heads=4):
        super().__init__()
        self.use_attention = use_attention and HAS_TORCH_GEOMETRIC and (GATConv is not None)
        
        if self.use_attention:
            self.conv1 = GATConv(in_dim, hidden_dim//heads, heads=heads, dropout=0.1)
            self.conv2 = GATConv(hidden_dim, emb_dim//heads, heads=heads, dropout=0.1, concat=True)
            self.out_proj = nn.Linear(emb_dim, emb_dim)
        elif HAS_TORCH_GEOMETRIC and GCNConv is not None:
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, emb_dim)
        else:
            # Fallback to simple neural networks
            print("Using fallback neural network (no graph convolution)")
            self.conv1 = FallbackGCNConv(in_dim, hidden_dim)
            self.conv2 = FallbackGCNConv(hidden_dim, emb_dim)

    def forward(self, x, edge_index, edge_weight=None):
        if self.use_attention:
            h = F.elu(self.conv1(x, edge_index))
            h = self.conv2(h, edge_index)
            return self.out_proj(F.elu(h))
        elif HAS_TORCH_GEOMETRIC:
            h = F.relu(self.conv1(x, edge_index, edge_weight))
            return self.conv2(h, edge_index, edge_weight)
        else:
            # Fallback path
            h = F.relu(self.conv1(x, edge_index, edge_weight))
            return self.conv2(h, edge_index, edge_weight)

class TransformerFusion(nn.Module):
    def __init__(self, feat_in, model_dim=64, nhead=4, nlayers=2):
        super().__init__()
        self.input_proj = nn.Linear(feat_in, model_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x_seq):  # [B,T,F]
        z = self.input_proj(x_seq)
        z = self.enc(z)
        z = z.transpose(1,2)
        return self.pool(z).squeeze(-1)

class MultiTaskHeads(nn.Module):
    def __init__(self, emb_dim, feat_dim):
        super().__init__()
        combined = emb_dim + feat_dim
        self.flood = nn.Sequential(nn.Linear(combined, 96), nn.ReLU(), nn.Linear(96, 2))
        self.rain  = nn.Sequential(nn.Linear(combined, 96), nn.ReLU(), nn.Linear(96, 1))
        self.alert = nn.Sequential(nn.Linear(combined, 128), nn.ReLU(), nn.Linear(128, 7))
    def forward(self, emb, feats):
        x = torch.cat([emb, feats], dim=1)
        return self.flood(x), self.rain(x).squeeze(1), self.alert(x)

# -----------------------------
# Cached Model Hub (persists across reruns)
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_model_hub():
    """
    Returns an object holding the full system state (data, scalers, models).
    This persists across reruns, so you won't retrain unless you call .retrain(...)
    """
    return ModelHub()

class ModelHub:
    def __init__(self):
        self.sys = None  # will hold ChennaiFloodSystem

    def ensure_loaded(self, **kwargs):
        if self.sys is None:
            self.sys = ChennaiFloodSystem(**kwargs)
            # Load data & graph
            self.sys.load_data()
            ei, ew, _ = build_region_graph_knn(k=self.sys.k_knn, device=self.sys.device)
            self.sys.edge_index, self.sys.edge_weight = ei, ew
            # Try load models; if missing, train once
            if self.sys.try_load_models():
                pass
            else:
                self.sys.train_all()
                self.sys.save_models()

    def retrain(self):
        self.sys.train_all()
        self.sys.save_models()

    def predict(self, input_features, node_weather, alpha, beta, gamma, ablate_dynamic_edges=False):
        return self.sys.predict_7day(input_features, node_weather, alpha, beta, gamma, ablate_dynamic_edges)

# -----------------------------
# System Orchestrator
# -----------------------------
class ChennaiFloodSystem:
    def __init__(self,
                 data_dir='data', rainfall_threshold=200.0,
                 use_attention=True, seq_len=7, hidden_dim=64, emb_dim=64,
                 lstm_hidden=64, use_transformer_fusion=True, k_knn=4):
        self.data_dir = data_dir
        self.threshold = rainfall_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.use_attention = use_attention and HAS_TORCH_GEOMETRIC
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.lstm_hidden = lstm_hidden
        self.use_transformer_fusion = use_transformer_fusion
        self.k_knn = k_knn

        self.regions = list(REGION_COORDS.keys())
        self.reg2id = {r:i for i,r in enumerate(self.regions)}

        self.data = None
        self.feature_cols = None
        self.region_scalers = {}
        self.scaler_node = StandardScaler()

        self.edge_index = None
        self.edge_weight = None

        self.gnn = None
        self.multi = None
        self.all_embeddings = None

        self.flood_stack = None
        self.rain_q10 = self.rain_q50 = self.rain_q90 = None
        self.alert_xgb = None
        self.feature_importance_ = {}

        self.use_shap = False
        try:
            import shap  # noqa
            self.use_shap = True
        except Exception:
            self.use_shap = False

    # ------------- data -------------
    def load_data(self):
        self.data, self.feature_cols, self.region_scalers = load_and_preprocess(self.data_dir, self.threshold)
        self.node_id_for_row = self.data['region'].map(self.reg2id).values

    # ------------- train all -------------
    def train_all(self, epochs=80, lr=1e-3, alpha_smooth=1.0, alpha_flood=1.0, alpha_rain=1.0, alpha_alert=1.0):
        self.train_encoder_and_heads(epochs, lr, alpha_smooth, alpha_flood, alpha_rain, alpha_alert)
        self.train_ensembles()

    # ------------- encoder -------------
    def train_encoder_and_heads(self, epochs=80, lr=1e-3,
                                alpha_smooth=1.0, alpha_flood=1.0, alpha_rain=1.0, alpha_alert=1.0):
        # region-level mean features
        region_feats = []
        for r in self.regions:
            rf = self.data[self.data['region']==r][self.feature_cols].mean(axis=0).values
            region_feats.append(rf)
        X_nodes = np.vstack(region_feats)
        X_nodes = self.scaler_node.fit_transform(X_nodes)
        X_nodes_t = torch.tensor(X_nodes, dtype=torch.float, device=self.device)

        in_dim = X_nodes.shape[1]
        self.gnn = RegionalEncoder(in_dim=in_dim, hidden_dim=self.hidden_dim, emb_dim=self.emb_dim,
                                   use_attention=self.use_attention).to(self.device)
        self.multi = MultiTaskHeads(emb_dim=self.emb_dim, feat_dim=in_dim).to(self.device)

        opt = torch.optim.Adam(list(self.gnn.parameters()) + list(self.multi.parameters()), lr=lr)

        ch_rows = self.data[self.data['region']=='chennai'].reset_index(drop=True)
        y_flood = ch_rows['flood_binary'].values
        y_rain = ch_rows['rainfall'].values
        y_alert = ch_rows['imd_alert'].values
        num_dates = len(ch_rows)
        ch_id = self.reg2id['chennai']

        ch_feat_raw = ch_rows[self.feature_cols].values
        ch_feat_scaled = self.scaler_node.transform(np.vstack([X_nodes.mean(0), ch_feat_raw]))[1:]
        feats_t = torch.tensor(ch_feat_scaled, dtype=torch.float, device=self.device)

        for ep in range(epochs):
            self.gnn.train(); self.multi.train()
            opt.zero_grad()
            emb_nodes = self.gnn(X_nodes_t, self.edge_index, self.edge_weight)
            emb_ch = emb_nodes[ch_id].unsqueeze(0).repeat(num_dates,1)

            fl_logits, rain_pred, al_logits = self.multi(emb_ch, feats_t)
            loss_flood = F.cross_entropy(fl_logits, torch.tensor(y_flood, dtype=torch.long, device=self.device))
            loss_rain  = F.mse_loss(rain_pred, torch.tensor(y_rain, dtype=torch.float, device=self.device))
            loss_alert = F.cross_entropy(al_logits, torch.tensor(y_alert, dtype=torch.long, device=self.device))

            # Only compute smoothness if we have actual graph structure
            if HAS_TORCH_GEOMETRIC and self.edge_index.numel() > 0:
                src, dst = self.edge_index
                smooth = ((emb_nodes[src] - emb_nodes[dst])**2).mean()
            else:
                smooth = torch.tensor(0.0, device=self.device)

            loss = alpha_flood*loss_flood + alpha_rain*loss_rain + alpha_alert*loss_alert + alpha_smooth*smooth
            loss.backward(); opt.step()

        self.gnn.eval()
        with torch.no_grad():
            region_emb = self.gnn(X_nodes_t, self.edge_index, self.edge_weight).detach().cpu().numpy()
            self.all_embeddings = region_emb[self.node_id_for_row]

    # ------------- ensembles -------------
    def train_ensembles(self):
        df_ch = self.data[self.data['region']=='chennai'].reset_index(drop=True)
        X_base = df_ch[self.feature_cols].values if len(self.feature_cols)>0 else np.zeros((len(df_ch),1))
        ch_idx = self.data[self.data['region']=='chennai'].index.values
        ch_emb = self.all_embeddings[ch_idx]
        X_fused = np.hstack([X_base, ch_emb])
        y_flood = df_ch['flood_binary'].values
        y_rain = df_ch['rainfall'].values
        y_alert = df_ch['imd_alert'].values

        Xtr, Xte, yf_tr, yf_te, yr_tr, yr_te, ya_tr, ya_te = train_test_split(
            X_fused, y_flood, y_rain, y_alert, test_size=0.2, random_state=SEED
        )

        clf_estimators = [
            ('xgb', XGBClassifier(n_estimators=300, max_depth=6, eval_metric='logloss', verbosity=0)),
            ('rf', RandomForestClassifier(n_estimators=250, max_depth=10, random_state=SEED, n_jobs=-1))
        ]
        stk = StackingClassifier(estimators=clf_estimators, final_estimator=LogisticRegression(max_iter=600), cv=3, n_jobs=-1)
        stk.fit(Xtr, yf_tr)
        self.flood_stack = stk

        q10 = GradientBoostingRegressor(loss='quantile', alpha=0.10, n_estimators=500, max_depth=3)
        q50 = GradientBoostingRegressor(loss='quantile', alpha=0.50, n_estimators=500, max_depth=3)
        q90 = GradientBoostingRegressor(loss='quantile', alpha=0.90, n_estimators=500, max_depth=3)
        q10.fit(Xtr, yr_tr); q50.fit(Xtr, yr_tr); q90.fit(Xtr, yr_tr)
        self.rain_q10, self.rain_q50, self.rain_q90 = q10, q50, q90

        xgb_mc = XGBClassifier(n_estimators=350, max_depth=7, objective='multi:softprob', num_class=7, verbosity=0)
        xgb_mc.fit(Xtr, ya_tr)
        self.alert_xgb = xgb_mc

        # store permutation importance for explain fallback
        try:
            pi = permutation_importance(self.alert_xgb, Xte, ya_te, n_repeats=8, random_state=SEED)
            emb_cols = [f'emb_{i}' for i in range(ch_emb.shape[1])]
            self.feature_importance_ = dict(zip(list(self.feature_cols) + emb_cols, pi.importances_mean))
        except Exception:
            self.feature_importance_ = {}

        # quick metrics (not returned; UI handles when needed)
        _ = {
            'flood_acc': accuracy_score(yf_te, stk.predict(Xte)),
            'rain_r2': r2_score(yr_te, q50.predict(Xte)),
            'alert_acc': accuracy_score(ya_te, xgb_mc.predict(Xte))
        }

    # ------------- save/load -------------
    def save_models(self, path="models/"):
        os.makedirs(path, exist_ok=True)
        if self.gnn: torch.save(self.gnn.state_dict(), os.path.join(path, "gnn.pt"))
        if self.multi: torch.save(self.multi.state_dict(), os.path.join(path, "multi.pt"))
        if self.flood_stack: joblib.dump(self.flood_stack, os.path.join(path, "flood.pkl"))
        if self.rain_q10 and self.rain_q50 and self.rain_q90:
            joblib.dump((self.rain_q10, self.rain_q50, self.rain_q90), os.path.join(path, "rain.pkl"))
        if self.alert_xgb: joblib.dump(self.alert_xgb, os.path.join(path, "alert.pkl"))
        joblib.dump(self.scaler_node, os.path.join(path, "scaler_node.pkl"))
        joblib.dump(self.feature_cols, os.path.join(path, "feature_cols.pkl"))

    def try_load_models(self, path="models/"):
        try:
            self.feature_cols = joblib.load(os.path.join(path, "feature_cols.pkl"))
            self.scaler_node = joblib.load(os.path.join(path, "scaler_node.pkl"))
            # build models with correct dims
            in_dim = len(self.feature_cols)
            self.gnn = RegionalEncoder(in_dim=in_dim, hidden_dim=self.hidden_dim, emb_dim=self.emb_dim,
                                       use_attention=self.use_attention).to(self.device)
            self.multi = MultiTaskHeads(emb_dim=self.emb_dim, feat_dim=in_dim).to(self.device)
            self.gnn.load_state_dict(torch.load(os.path.join(path, "gnn.pt"), map_location=self.device))
            self.multi.load_state_dict(torch.load(os.path.join(path, "multi.pt"), map_location=self.device))
            self.flood_stack = joblib.load(os.path.join(path, "flood.pkl"))
            self.rain_q10, self.rain_q50, self.rain_q90 = joblib.load(os.path.join(path, "rain.pkl"))
            self.alert_xgb = joblib.load(os.path.join(path, "alert.pkl"))

            # also rebuild cached all_embeddings for current data
            region_feats = []
            for r in self.regions:
                rf = self.data[self.data['region']==r][self.feature_cols].mean(axis=0).values
                region_feats.append(rf)
            X_nodes = np.vstack(region_feats)
            X_nodes = self.scaler_node.transform(X_nodes)
            X_nodes_t = torch.tensor(X_nodes, dtype=torch.float, device=self.device)
            with torch.no_grad():
                self.all_embeddings = self.gnn(X_nodes_t, self.edge_index, self.edge_weight).detach().cpu().numpy()
                self.all_embeddings = self.all_embeddings[self.node_id_for_row]
            return True
        except Exception:
            return False

    # ------------- inference -------------
    def predict_7day(self, input_dict, forecast_node_weather, alpha=0.6, beta=0.3, gamma=0.1,
                     ablate_dynamic_edges=False):
        days = 7
        preds = {'median':[], 'q10':[], 'q90':[], 'flood_prob':[], 'flood_pred':[], 'alert_class':[], 'alert_prob':[]}
        influence_mats = []

        feat_vec = np.zeros((1, len(self.feature_cols)))
        c2i = {c:i for i,c in enumerate(self.feature_cols)}
        for c,v in input_dict.items():
            if c in c2i: feat_vec[0, c2i[c]] = float(v)
        try:
            feat_scaled = self.scaler_node.transform(feat_vec)
        except Exception:
            feat_scaled = feat_vec

        region_feats = []
        for r in self.regions:
            rf = self.data[self.data['region']==r][self.feature_cols].mean(axis=0).values
            region_feats.append(rf)
        X_nodes = np.vstack(region_feats)
        try:
            X_nodes_scaled = self.scaler_node.transform(X_nodes)
        except Exception:
            X_nodes_scaled = X_nodes
        X_nodes_t_base = torch.tensor(X_nodes_scaled, dtype=torch.float, device=self.device)

        ch_idx = self.reg2id['chennai']

        for d in range(days):
            if ablate_dynamic_edges:
                dyn_ei, dyn_ew = self.edge_index, self.edge_weight
            else:
                node_weather_day = {}
                for r,v in forecast_node_weather.items():
                    dd = v['daily'][d] if len(v['daily'])>d else v['daily'][0]
                    node_weather_day[r] = {'wind_speed': dd.get('wind_speed',0.0),
                                           'wind_deg': dd.get('wind_deg',0.0),
                                           'pressure': dd.get('pressure',1013.0)}
                dyn_ei, dyn_ew = build_dynamic_edge_weights_for_day(node_weather_day, alpha, beta, gamma, device=self.device)

            self.gnn.eval()
            with torch.no_grad():
                emb_nodes = self.gnn(X_nodes_t_base, dyn_ei, dyn_ew)
                emb_ch = emb_nodes[ch_idx].unsqueeze(0).cpu().numpy()

            X_fused = np.hstack([feat_scaled, emb_ch])

            try:
                q10 = float(self.rain_q10.predict(X_fused)[0])
                q50 = float(self.rain_q50.predict(X_fused)[0])
                q90 = float(self.rain_q90.predict(X_fused)[0])
            except Exception:
                q10=q50=q90=0.0
            try:
                flood_prob = float(self.flood_stack.predict_proba(X_fused)[0][1])
                flood_pred = int(self.flood_stack.predict(X_fused)[0])
            except Exception:
                flood_prob=0.0; flood_pred=0
            try:
                ap = self.alert_xgb.predict_proba(X_fused)[0]
                alert_class = int(np.argmax(ap)); alert_prob = float(ap[alert_class])
            except Exception:
                alert_class=0; alert_prob=1.0

            preds['median'].append(q50); preds['q10'].append(q10); preds['q90'].append(q90)
            preds['flood_prob'].append(flood_prob); preds['flood_pred'].append(flood_pred)
            preds['alert_class'].append(alert_class); preds['alert_prob'].append(alert_prob)

            # dense influence matrix
            n = len(self.regions)
            mat = np.zeros((n,n))
            ei = dyn_ei.detach().cpu().numpy().T
            ew = dyn_ew.detach().cpu().numpy()
            for (a,b),w in zip(ei, ew):
                mat[a,b] = w
            influence_mats.append(mat)

        return preds, influence_mats