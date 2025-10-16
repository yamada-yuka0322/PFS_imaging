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

def build_model(input_dim, width, depth):
    layers = [nn.Linear(input_dim, width), nn.BatchNorm1d(width), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU()]
    layers += [nn.Linear(width, 1)]
    return nn.Sequential(*layers)

def make_objective(test_X, test_Y, test_Fpix, val_X, val_Y, val_Fpix, loader):
    def objective(trial):
        # ハイパーパラメータの提案
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        depth = trial.suggest_int("depth", 2, 6)
        width = trial.suggest_categorical("width", [64, 128, 256, 512, 1024])

        model = build_model(12, width, depth)
        loss_fn = _CustomLoss(model, lambda_reg=1e-3)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float("inf")
        patience = 20
        trigger_times = 0
        best_model = copy.deepcopy(model)
        
        epoch = []
        loss_list = []

        for t in range(300):
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
                test_pred = model(test_X)
                test_loss = loss_fn(test_pred, test_Y, test_Fpix).item()
                loss_list.append(test_loss)
                epoch.append(t)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
                best_model = copy.deepcopy(model)
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    break

        # モデル保存
        #model_path = f"/home/YukaYamada/repository/PFS/notebook//output/model_trial_{trial.number}.pt"
        #torch.save(best_model.state_dict(), model_path)
        trial.set_user_attr("best_state_dict", best_model.state_dict())


        return best_val_loss

    return objective

def run_optuna_nn(property_df, pixels, keys):
    df = property_df[property_df['healpix'].isin(pixels)].dropna(subset=['target']).copy()

    nside = 256
    area = hp.nside2pixarea(nside, degrees=True)
    fpix = df_cleaned["area"] / area
    
    df_cleaned = df.dropna(subset=['target'])
    properties = df_cleaned[keys]

    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(properties), columns=properties.columns)

    mean = np.sum(df_cleaned["target"] * df_cleaned["area"]) / np.sum(df_cleaned["area"])
    density = df_cleaned["target"] / mean

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
    
    test_X = torch.from_numpy(val_X_np).type(dtype)
    test_Y = torch.from_numpy(val_Y_np.values).type(dtype).view(-1, 1)
    test_Fpix = torch.from_numpy(val_fpix_np.values).type(dtype)

    dataset = TensorDataset(train_X, train_Y, train_Fpix)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    objective = make_objective(test_X, test_Y, test_Fpix, val_X, val_Y, val_Fpix, loader)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)
    
    best_params = study.best_trial.params
    best_state_dict = study.best_trial.user_attrs["best_state_dict"]

    # 同じ構造で再構築して保存
    model = build_model(input_dim=X.shape[1], width=best_params["width"], depth=best_params["depth"])
    model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), "/home/YukaYamada/repository/PFS/notebook/output/best_model_log.pt")
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {"in_dim": len(keys), "width": best_params["width"], "depth": best_params["depth"], "out_dim": 1, "lr":best_params["lr"]},
        "scaler": scaler  # joblib.dump したパス or そのまま入れてもOK（要件に応じて）
    }
    torch.save(checkpoint, "best_model.pt")  # or .pth


    return study

#scalingをどうするべきか聞きたい
