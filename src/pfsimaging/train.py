from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import copy
import optuna

import numpy as np

import healpy as hp

import os

import json

class CustomLoss(nn.Module):
    """
    Container of loss function. Completeness weighted mean square.
    
    Attributes
    ------------------------------------------------------
    model: NN model

    Methods
    ------------------------------------------------------
    forward(output, target, fpix):
        Calculate the mean square error given the
        1. output: predicted, normalized target density
        2. target: true, normalized target density
        3. fpix: completeness of each pixel
    
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target, fpix):
        
        squared_error = (output - target) ** 2
        
        return torch.sum(fpix * squared_error) / torch.sum(fpix)
    

def build_model(input_dim, width, depth):
    """
    Function to build the NN-model
    
    Parameter
    ------------------------------------------------------
    input_dim: int
    dimention of the input parameter (the number of imaging attributes considered)
    
    width: int
     Number of neurons in each hidden layer.
    
    depth: int
    number of the total layer (including the input and output layer)

    Output
    ------------------------------------------------------
    nn.Sequential(*layers):
    NN-model
    
    """
    layers = [nn.Linear(input_dim, width), nn.BatchNorm1d(width), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU()]
    layers += [nn.Linear(width, 1)]
    return nn.Sequential(*layers)

def make_objective(test_X, test_Y, test_Fpix, val_X, val_Y, val_Fpix, loader):
    """
    Create an Optuna objective function for neural network hyperparameter optimization.

    Parameters
    ----------
    test_X, test_Y, test_Fpix : torch.Tensor
        Test dataset used to evaluate the model corresponding to the
        best validation loss.

    val_X, val_Y, val_Fpix : torch.Tensor
        Validation dataset used for early stopping and hyperparameter
        optimization.

    loader : torch.utils.data.DataLoader
        DataLoader containing the training samples.

    Returns
    -------
    objective : callable
        Objective function passed to ``optuna.study.optimize()``.
        The objective optimizes the learning rate, network depth,
        and hidden layer width. Training stops early if the validation
        loss does not improve for a given number of epochs.
    """
    def objective(trial):
        """
        Train a neural network with hyperparameters proposed by Optuna
        and return the minimum validation loss.

        Hyperparameters
        ----------------
        lr : float
            Learning rate sampled logarithmically between 1e-6 and 1e-3.
        depth : int
            Number of hidden layers (2--6).
        width : int
            Number of neurons in each hidden layer.

        Returns
        -------
        float
            Minimum validation loss achieved during training.
        """
        # suggest hiperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        depth = trial.suggest_int("depth", 2, 4)
        width = trial.suggest_categorical("width", [16, 32, 64, 128])

        model = build_model(12, width, depth)
        loss_fn = CustomLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float("inf")
        patience = 20 #If loss doesn't get better after patience steps, optuna stops
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

def run_optuna_nn(run_dir, Property, keys, n_trials=200, top_k=5):
    """
    Function to train the neural network
    
    Parameter
    ------------------------------------------------------
    run_dir: string
    path of the directory to save the outputs
    
    Property:table
    astropy table including imaging systematics and target density of each healpixels. Must include all column in keys, 'target' and 'area'
    
    keys: list
     The name of imaging attributes to be considered
    
    n_trials: int
    number of the trials that would be run in optuna
    
    top_k: int
    number of trials that would be saved
    

    Output
    ------------------------------------------------------
    dictionary
    """
    data = prepare_nn_data(Property, keys)
    loader, val_data, test_data = make_tensors(data)

    val_X, val_Y, val_Fpix = val_data
    test_X, test_Y, test_Fpix = test_data

    objective = make_objective(
        test_X, test_Y, test_Fpix,
        val_X, val_Y, val_Fpix,
        loader
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    save_optuna_results(
        study,
        input_dim=data["input_dim"],
        run_dir=run_dir,
        top_k=top_k,
    )

    return study

def prepare_nn_data(Property, keys, nside=256, test_size=0.2):
    """
    Function to prepare the train, validation and test datasets
    
    Parameter
    ------------------------------------------------------
    property_df: pandas dataframe
    imaging properties of each healpix. Must include all column in keys, 'target' and 'area'
    
    keys: list
     The name of imaging attributes to be considered
    
    nside: int
    nside of each healpix
    
    test_size: float
    ratio of the validation and test datasize
    
    Output
    ------------------------------------------------------
    dictionary with columns
    1. train: train dataset (imaging attributes, target density, completeness)
    2. val: validation dataset (imaging attributes, target density, completeness)
    3. test: test dataset (imaging attributes, target density, completeness)
    4. input_dim: number of input imaging attributes
    5. scaler: scaler to normalize the imaging attributes
    """
    cleaned = Property[Property['target']>0.0]

    properties = cleaned[keys]
    
    if 'star' in keys:
        properties['star'] = np.log10(properties['star'])
        
    if 'csfd_desi_extinction' in keys:
        properties['csfd_desi_extinction'] = np.log10(properties['csfd_desi_extinction'])
    
    df = properties.to_pandas()

    #scaler = StandardScaler()
    #X_standardized = scaler.fit_transform(df)

    mean = np.sum(cleaned["target"] * cleaned["area"]) / np.sum(cleaned["area"])
    density = np.log10(cleaned["target"] / mean)

    pix_area = hp.nside2pixarea(nside, degrees=True)
    fpix = cleaned["area"] / pix_area

    X = df.to_numpy(dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    fpix = np.asarray(fpix, dtype=np.float64)

    (
        train_X_np,
        temp_X_np,
        train_Y_np,
        temp_Y_np,
        train_fpix_np,
        temp_fpix_np,
    ) = train_test_split(
        X,
        density,
        fpix,
        test_size=test_size,
        random_state=42,
    )

    (
        val_X_np,
        test_X_np,
        val_Y_np,
        test_Y_np,
        val_fpix_np,
        test_fpix_np,
    ) = train_test_split(
        temp_X_np,
        temp_Y_np,
        temp_fpix_np,
        test_size=0.5,
        random_state=42,
    )

    scaler = StandardScaler()
    train_X_np = scaler.fit_transform(train_X_np)
    val_X_np = scaler.transform(val_X_np)
    test_X_np = scaler.transform(test_X_np)

    return {
        "train": (train_X_np, train_Y_np, train_fpix_np),
        "val": (val_X_np, val_Y_np, val_fpix_np),
        "test": (test_X_np, test_Y_np, test_fpix_np),
        "input_dim": X.shape[1],
        "scaler": scaler,
        "mean": mean,
    }

def to_tensor_1d(x, dtype=torch.float):
    return torch.from_numpy(np.asarray(x, dtype=float)).type(dtype).view(-1, 1)

def to_tensor(x, dtype=torch.float):
    return torch.from_numpy(np.asarray(x, dtype=float)).type(dtype)

def make_tensors(data, batch_size=128):
    dtype = torch.float

    train_X_np, train_Y_np, train_fpix_np = data["train"]
    val_X_np, val_Y_np, val_fpix_np = data["val"]
    test_X_np, test_Y_np, test_fpix_np = data["test"]

    train_X = to_tensor(train_X_np, dtype)
    train_Y = to_tensor_1d(train_Y_np, dtype)
    train_Fpix = to_tensor_1d(train_fpix_np, dtype)

    val_X = to_tensor(val_X_np, dtype)
    val_Y = to_tensor_1d(val_Y_np, dtype)
    val_Fpix = to_tensor_1d(val_fpix_np, dtype)

    test_X = to_tensor(test_X_np, dtype)
    test_Y = to_tensor_1d(test_Y_np, dtype)
    test_Fpix = to_tensor_1d(test_fpix_np, dtype)

    loader = DataLoader(
        TensorDataset(train_X, train_Y, train_Fpix),
        batch_size=batch_size,
        shuffle=True,
    )

    return loader, (val_X, val_Y, val_Fpix), (test_X, test_Y, test_Fpix)

def save_optuna_results(study, input_dim, run_dir, top_k=5):
    os.makedirs(run_dir, exist_ok=True)

    rows = []
    for t in study.trials:
        rows.append({
            "trial": t.number,
            "value(val_loss)": t.value,
            "val_loss": t.user_attrs.get("val_loss", np.nan),
            "test_loss": t.user_attrs.get("test_loss", np.nan),
            "params": json.dumps(t.params),
        })

    df_log = pd.DataFrame(rows).sort_values("value(val_loss)")
    df_log.to_csv(os.path.join(run_dir, "trials_log.csv"), index=False)

    save_trials = df_log.head(top_k)["trial"].tolist()

    for rank, tnum in enumerate(save_trials, start=1):
        t = study.trials[tnum]
        sd = t.user_attrs.get("state_dict", None)
        if sd is None:
            continue

        width = t.params["width"]
        depth = t.params["depth"]

        model = build_model(input_dim=input_dim, width=width, depth=depth)
        model.load_state_dict(sd)

        fname = os.path.join(run_dir, f"rank{rank:02d}_trial{tnum:03d}_val{t.value:.5g}.pt")

        torch.save({
            "state_dict": model.state_dict(),
            "config": {"input_dim": input_dim, "width": width, "depth": depth},
            "lr": t.params["lr"],
            "val_loss": t.user_attrs.get("val_loss", np.nan),
            "test_loss": t.user_attrs.get("test_loss", np.nan),
            "trial": t.number,
            "params": t.params,
        }, fname)

    best = study.best_trial
    best_params = best.params
    best_model = build_model(
        input_dim=input_dim,
        width=best_params["width"],
        depth=best_params["depth"],
    )
    best_model.load_state_dict(best.user_attrs["state_dict"])

    best_path = os.path.join(run_dir, "best_model.pt")

    torch.save({
        "state_dict": best_model.state_dict(),
        "config": {
            "input_dim": input_dim,
            "width": best_params["width"],
            "depth": best_params["depth"],
        },
        "lr": best_params["lr"],
        "val_loss": best.user_attrs.get("val_loss", np.nan),
        "test_loss": best.user_attrs.get("test_loss", np.nan),
        "trial": best.number,
        "params": best_params,
    }, best_path)
