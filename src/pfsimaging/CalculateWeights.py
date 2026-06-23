from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import optuna
import copy

import os, json, copy
from datetime import datetime

from sklearn.metrics import r2_score, mean_squared_error

import healpy as hp
import numpy as np

from astropy.table import Table

import yaml

import argparse

args = None
config=None

def mask_rdeep():
    nside = 256

    ra_deg = 163.8
    dec_deg = 0.2
    radius_deg = 0.85

    # 中心方向ベクトル
    vec = hp.ang2vec(ra_deg, dec_deg, lonlat=True)

    # 円内の healpix index
    pix = hp.query_disc(
        nside,
        vec,
        np.radians(radius_deg),
        inclusive=False,
        nest=False
    )

    return pix

def build_model(input_dim, width, depth):
    layers = [nn.Linear(input_dim, width), nn.BatchNorm1d(width), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU()]
    layers += [nn.Linear(width, 1)]
    return nn.Sequential(*layers)

def main():
    """function to calculate weights using quadratic regression

    Parameters
    ----------------------------------------------------
    property:pd dataframe
    dataframe including imaging systematics and target density of each healpixels

    pixels: list([int])
    list of healpixel number considered in the weighting

    keys: list([string])
    list of imaging systematics to consider

    Output
    -----------------------------------------------------
    weights: pd dataframe
    dataframe including imaging systematics and target density "and weights" of each healpixels
    """
    #params={'lr': 7.394509408202555e-06, 'depth': 2, 'width': 1024}
    #width = params['width']               # Optunaで選ばれたbestのwidth
    #depth = params['depth']             # Optunaで選ばれたbestのdepth

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", default=None,
                        help="YAML config file containing paths and SQL settings")

    global args, config
    args = parser.parse_args()
    
    config = {}
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        parser.error("--config are required")

    keys = ['gseeing', 'rseeing', 'iseeing', 'zseeing', 'yseeing', 
           'g_depth', 'r_depth', 'i_depth', 'z_depth', 'y_depth',
           'desi-csfd_extinction', 'star']

    path = os.path.join(config['Imaging']['imaging_dir'] ,'all_property.fits')
    properties = Table.read(path)
    _properties = properties[properties['area']>0.0].to_pandas()
    
    pix = mask_rdeep()
    mask = ~np.isin(_properties['healpix'], pix)
    
    df = _properties[mask]
    df = df.replace([np.inf, -np.inf], np.nan)
    df_cleaned = df.dropna()

    
    # build_modelは既存の関数を利用（入力次元は len(keys) ）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load("/home/YukaYamada/repository/PFS/PFS_imaging/bin/output/optuna_runs/test/best_model.pt")
    model = build_model(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
        
    properties = df_cleaned[keys]
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(properties), columns= properties.columns)

    X = np.concatenate([np.array(df_standardized[key]).reshape(-1, 1) for key in keys], axis=1)

    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        pred = model(X_tensor).squeeze().detach().cpu().numpy()

    # 密度 → 重み（ゼロ割/負値はクリップ）
    #density = np.clip(pred, 1e-6, None)
    density = pred
    w_nn = np.where(density <= 0, np.nan, 1.0 / density)

    out_table = Table()
    out_table["healpix"] = np.array(df_cleaned["healpix"])
    out_table["nn_weights"] = np.array(w_nn)

    out_table.write("dat/nn_weights.fits", overwrite=True)

if __name__ == '__main__':
	main()