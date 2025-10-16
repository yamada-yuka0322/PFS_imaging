from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import optuna

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import copy

import healpy as hp

import numpy as np

import os, json, copy
from datetime import datetime

nside = 256
area = hp.nside2pixarea(nside,degrees=True)

SAVE_DIR = "/home/YukaYamada/repository/PFS/notebook/output/optuna_runs"  # 保存先
os.makedirs(SAVE_DIR, exist_ok=True)

def target_weights(object, selection, properties):
    targets = object[selection]
    target_id = targets['OBJID']
    target_ra = targets['RA']%360
    target_dec = targets['DEC']
    
    target_healpy = hp.ang2pixel(nside=nside, theta=target_ra, phi=target_dec, lonlat=True)
    
    target_data = pd.DataFrame({'id':target_id, 'healpix':target_healpy})
    
    target_weight = pd.merge(target_data, properties, on="healpix", how="left")
    
    return target_weight
    
    

def linear_weights(property, pixels, keys):
    """function to calculate weights using linear regression

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
    dataframe with weights of each healpixels
    """
    df = property [property['healpix'].isin(pixels)]
    df_cleaned = df.dropna(subset=['target'])
    properties = df_cleaned[keys]
    
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(properties), columns= properties.columns)
    
    mean = np.sum(df_cleaned["target"]*df_cleaned["area"])/np.sum(df_cleaned["area"])
    density = df_cleaned["target"]/mean

    X = np.concatenate([np.array(df_standardized[key]).reshape(-1, 1) for key in keys], axis=1)

    train_X, test_X, train_Y, test_Y = train_test_split(X, density)

    #learn using linear regression
    regr = LinearRegression()
    regr.fit(train_X, train_Y)

    predict_train = regr.predict(train_X)
    predicted_density = regr.predict(test_X)

    print("predicted density for test data:", predicted_density)

    # 訓練データに対するR²
    r2_train = regr.score(train_X, train_Y)
    print("Training R² linear:", r2_train)

    # テストデータに対するR²
    r2_test = regr.score(test_X, test_Y)
    print("Test R² linear:", r2_test)

    # 訓練データに対するMSE
    mse_train = mean_squared_error(train_Y, predict_train)
    print("Training MSE linear:", mse_train)

    # テストデータに対するMSE
    mse_test = mean_squared_error(test_Y, predicted_density)
    print("Test MSE linear:", mse_test)

    _density = regr.predict(X)
    _weight = 1/_density

    df_cleaned['lin_weights'] = _weight
    
    # property weights (= regression coefficients)
    coef_df = pd.DataFrame({
        "property": keys,
        "coefficient": regr.coef_
    })

    return df_cleaned, coef_df

def quadratic_weights(property, pixels, keys):
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
    df = property [property['healpix'].isin(pixels)]
    df_cleaned = df.dropna(subset=['target'])
    properties = df_cleaned[keys]
    
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(properties), columns= properties.columns)
    
    mean = np.sum(df_cleaned["target"]*df_cleaned["area"])/np.sum(df_cleaned["area"])
    density = df_cleaned["target"]/mean

    X = np.concatenate([np.array(df_standardized[key]).reshape(-1, 1) for key in keys], axis=1)

    train_X, test_X, train_Y, test_Y = train_test_split(X, density)

    # Transform features to polynomial (degree=2)
    poly = PolynomialFeatures(degree=2)
    train_X_poly = poly.fit_transform(train_X)
    test_X_poly = poly.transform(test_X)
    X_poly = poly.transform(X)

    # Train the regression model
    regr = LinearRegression()
    regr.fit(train_X_poly, train_Y)

    # Predict values
    predict_train = regr.predict(train_X_poly)
    predicted_density = regr.predict(test_X_poly)

    # Print results
    print("Predicted density for test data:", predicted_density)

    # R² Scores
    r2_train = regr.score(train_X_poly, train_Y)
    r2_test = regr.score(test_X_poly, test_Y)
    print("Training R² quadratic:", r2_train)
    print("Test R² quadratic:", r2_test)

    # Mean Squared Errors
    mse_train = mean_squared_error(train_Y, predict_train)
    mse_test = mean_squared_error(test_Y, predicted_density)
    print("Training MSE quadratic:", mse_train)
    print("Test MSE quadratic:", mse_test)

    _density = regr.predict(X_poly)
    _weight = 1/_density

    df_cleaned['quad_weights'] = _weight

    return df_cleaned

def nn_weights(property, pixels, keys):
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

    # build_modelは既存の関数を利用（入力次元は len(keys) ）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load("/home/YukaYamada/repository/PFS/notebook/output/optuna_runs/logstar_study_2025-10-09_19-15-16/best_model_logstar.pt")
    model = build_model(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
        
    df = property [property['healpix'].isin(pixels)]
    df_cleaned = df.dropna(subset=['target'])
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
    df_cleaned['nn_weights'] = w_nn
    
    return df_cleaned

class CustomLoss(nn.Module):
    def __init__(self, model, lambda_reg=1e-3):
        super().__init__()
        self.model = model
        self.lambda_reg = lambda_reg

    def forward(self, output, target, fpix):
        weighted_mse = torch.mean(fpix * (output - target) ** 2)

        # L2正則化項：全パラメータに対して
        #l2_reg = sum(torch.norm(param) ** 2 for param in self.model.parameters())

        return weighted_mse



def _nn_weights(property, pixels, keys):
    """function to calculate the weights using neural network

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
    df = property [property['healpix'].isin(pixels)]
    df_cleaned = df.dropna(subset=['target'])
    properties = df_cleaned[keys]
    
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(properties), columns= properties.columns)
    
    mean = np.sum(df_cleaned["target"]*df_cleaned["area"])/np.sum(df_cleaned["area"])
    density = df_cleaned["target"]/mean
    
    fpix = df_cleaned['area']/area

    X = np.concatenate([np.array(df_standardized[key]).reshape(-1, 1) for key in keys], axis=1)

    train_X, test_X, train_Y, test_Y, train_fpix, test_fpix = train_test_split(X, density, fpix)
    
    dtype = torch.float
    train_X = torch.from_numpy(train_X).type(dtype)
    train_Y = torch.from_numpy(train_Y.values).type(dtype)
    train_Y = train_Y.view(-1, 1)
    
    test_X = torch.from_numpy(test_X).type(dtype)
    test_Y = torch.from_numpy(test_Y.values).type(dtype).view(-1, 1)
    
    train_Fpix = torch.from_numpy(train_fpix.values).type(dtype)
    test_Fpix = torch.from_numpy(test_fpix.values).type(dtype)
    
    X_all = torch.from_numpy(X).type(dtype)
    # Create a data loader with batch size of 4.
    dataset = TensorDataset(train_X, train_Y, train_Fpix)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    loss_list = []

    # Two-layer NN: 2 -> 4 -> 1
    model = torch.nn.Sequential(
        torch.nn.Linear(11, 22, bias=True),
        torch.nn.BatchNorm1d(22),
        torch.nn.ReLU(),
        torch.nn.Linear(22, 1, bias=True),
    )
    
    patience = 20
    best_val_loss = float('inf')
    trigger_times = 0
    best_model = copy.deepcopy(model)

    
    loss_fn = CustomLoss(model, lambda_reg=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

    for t in range(10000):
        for _x, _y, _fpix in loader:
            y_hat = model(_x)
            loss = loss_fn(y_hat, _y, _fpix)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # validation loss
        model.eval()
        with torch.no_grad():
            val_pred = model(test_X)
            val_loss = loss_fn(val_pred, test_Y, test_Fpix).item()
            loss_list.append(val_loss)
            #y_hat_all = model(X)
            #loss_epoch = loss_fn(y_hat_all, Y).item()
            #loss_list.append(loss_epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
                best_model = copy.deepcopy(model)
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early stopping at epoch {t}")
                    break
    
    model = best_model
    with torch.no_grad():
        Y_pred = model(torch.from_numpy(X).type(dtype)).numpy()
        
    density = Y_pred.reshape(1, -1)
    _weight = 1/density
    df_cleaned['weights'] = _weight.flatten()
    return df_cleaned

def build_model(input_dim, width, depth):
    layers = [nn.Linear(input_dim, width), nn.BatchNorm1d(width), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU()]
    layers += [nn.Linear(width, 1)]
    return nn.Sequential(*layers)

def make_objective(test_X, test_Y, test_Fpix, val_X, val_Y, val_Fpix, loader):
    def objective(trial: optuna.trial.Trial):
        # ハイパーパラメータの提案
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        depth = trial.suggest_int("depth", 2, 6)
        width = trial.suggest_categorical("width", [64, 128, 256, 512, 1024])

        model = build_model(12, width, depth)
        loss_fn = CustomLoss(model, lambda_reg=1e-3)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float("inf")
        patience = 20
        trigger_times = 0
        best_model = copy.deepcopy(model)
        EPOCHS = 300
        
        epoch = []
        loss_list = []

        for t in range(EPOCHS):
            model.train()
            for _x, _y, _fpix in loader:
                y_hat = model(_x)
                loss = loss_fn(y_hat, _y, _fpix)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(val_X)
                val_loss = loss_fn(val_pred, val_Y, val_Fpix).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                test_pred = model(test_X)
                test_loss = loss_fn(test_pred, test_Y, test_Fpix).item()
                trigger_times = 0
                best_model = copy.deepcopy(model)
                best_sd = copy.deepcopy(model.state_dict())
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    break

        # trial に保存（あとで一括保存する）
        trial.set_user_attr("state_dict", best_sd)
        trial.set_user_attr("val_loss", float(best_val_loss))
        trial.set_user_attr("test_loss", float(test_loss))
        trial.set_user_attr("epochs", EPOCHS)


        return best_val_loss

    return objective

def run_optuna_nn(property_df, pixels, keys, n_trials=200, top_k=5, tag="logstar"):
    df = property_df[property_df['healpix'].isin(pixels)]
    df_cleaned = df.dropna(subset=['target'])
    properties = df_cleaned[keys]

    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(properties), columns=properties.columns)

    mean = np.sum(df_cleaned["target"] * df_cleaned["area"]) / np.sum(df_cleaned["area"])
    density = df_cleaned["target"] / mean

    nside = 256
    area = hp.nside2pixarea(nside, degrees=True)
    fpix = df_cleaned["area"] / area

    X = np.stack([df_standardized[key].values for key in keys], axis=1)

    train_X_np, temp_X_np, train_Y_np, temp_Y_np, train_fpix_np, temp_fpix_np = train_test_split(
        X, density, fpix, test_size=0.2, random_state=42)
    val_X_np, test_X_np, val_Y_np, test_Y_np, val_fpix_np, test_fpix_np = train_test_split(
        temp_X_np, temp_Y_np, temp_fpix_np, test_size=0.5, random_state=42)

    dtype = torch.float
    train_X = torch.from_numpy(train_X_np).type(dtype)
    train_Y = torch.from_numpy(train_Y_np.values).type(dtype).view(-1, 1)
    train_Fpix = torch.from_numpy(train_fpix_np.values).type(dtype)

    val_X = torch.from_numpy(val_X_np).type(dtype)
    val_Y = torch.from_numpy(val_Y_np.values).type(dtype).view(-1, 1)
    val_Fpix = torch.from_numpy(val_fpix_np.values).type(dtype)
    
    test_X = torch.from_numpy(test_X_np).type(dtype)
    test_Y = torch.from_numpy(test_Y_np.values).type(dtype).view(-1, 1)
    test_Fpix = torch.from_numpy(test_fpix_np.values).type(dtype)

    dataset = TensorDataset(train_X, train_Y, train_Fpix)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    objective = make_objective(test_X, test_Y, test_Fpix, val_X, val_Y, val_Fpix, loader)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    # --- ここから保存処理 ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(SAVE_DIR, f"{tag}_study_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 1) すべてのトライアルのメタデータを CSV に保存
    rows = []
    for t in study.trials:
        row = {
            "trial": t.number,
            "value(val_loss)": t.value,
            "val_loss": t.user_attrs.get("val_loss", np.nan),
            "test_loss": t.user_attrs.get("test_loss", np.nan),
            "params": json.dumps(t.params),
        }
        rows.append(row)
    df_log = pd.DataFrame(rows).sort_values("value(val_loss)")
    df_log.to_csv(os.path.join(run_dir, "trials_log.csv"), index=False)

    # 2) 全トライアル or 上位K件のモデルを保存（容量に応じて選んでください）
    #    ここでは上位K件のみ保存（全件保存したい場合は df_log の全行を回せばOK）
    save_trials = df_log.head(top_k)["trial"].tolist()
    for rank, tnum in enumerate(save_trials, start=1):
        t = study.trials[tnum]
        sd = t.user_attrs.get("state_dict", None)
        if sd is None:
            continue
        fname = os.path.join(run_dir, f"rank{rank:02d}_trial{tnum:03d}_val{t.value:.5g}.pt")

        # 同じ構造でモデル再構築（各 trial のハイパラを使用）
        width = t.params["width"]
        depth = t.params["depth"]
        model = build_model(input_dim=X.shape[1], width=width, depth=depth)
        model.load_state_dict(sd)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {"input_dim": X.shape[1], "width": width, "depth": depth},
                "lr": t.params["lr"],
                "val_loss": t.user_attrs.get("val_loss", np.nan),
                "test_loss": t.user_attrs.get("test_loss", np.nan),
                "trial": t.number,
                "params": t.params,
            },
            fname,
        )

    # 3) 最良モデルも別名で保存（従来互換）
    best_params = study.best_trial.params
    best_sd = study.best_trial.user_attrs["state_dict"]
    best_model = build_model(input_dim=X.shape[1], width=best_params["width"], depth=best_params["depth"])
    best_model.load_state_dict(best_sd)

    best_path = os.path.join(run_dir, "best_model_logstar.pt")
    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "config": {"input_dim": X.shape[1], "width": best_params["width"], "depth": best_params["depth"]},
            "lr": best_params["lr"],
            "val_loss": study.best_trial.user_attrs.get("val_loss", np.nan),
            "test_loss": study.best_trial.user_attrs.get("test_loss", np.nan),
            "trial": study.best_trial.number,
            "params": best_params,
        },
        best_path,
    )

    print(f"[saved] logs: {os.path.join(run_dir, 'trials_log.csv')}")
    print(f"[saved] best: {best_path}")
    print(f"[saved] top-{top_k} checkpoints in: {run_dir}")
    return study
