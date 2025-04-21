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

import healpy as hp

import numpy as np

nside = 256
area = hp.nside2pixarea(nside,degrees=True)

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

    df_cleaned['weights'] = _weight

    return df_cleaned

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

    df_cleaned['weights'] = _weight

    return df_cleaned

class CustomLoss(nn.Module):
    def __init__(self, model, lambda_reg=1e-3):
        super().__init__()
        self.model = model
        self.lambda_reg = lambda_reg

    def forward(self, output, target, fpix):
        weighted_mse = torch.mean(fpix * (output - target) ** 2)

        # L2正則化項：全パラメータに対して
        l2_reg = sum(torch.norm(param) ** 2 for param in self.model.parameters())

        return weighted_mse + self.lambda_reg * l2_reg



def nn_weights(property, pixels, keys):
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
    train_Y = torch.from_numpy(train_Y).type(dtype)
    train_Y = train_Y.view(-1, 1)
    
    train_Fpix = torch.from_numpy(train_fpix).type(dtype)
    
    X_all = torch.from_numpy(X).type(dtype)
    # Create a data loader with batch size of 4.
    dataset = TensorDataset(train_X, train_Y, train_Fpix)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    loss_list = []

    # Two-layer NN: 2 -> 4 -> 1
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 6, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(6, 1, bias=True),
    )
    
    loss_fn = CustomLoss(model, lambda_reg=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for t in range(5000):
        for _x, _y, _fpix in loader:
            y_hat = model(_x)
            loss = loss_fn(y_hat, _y, _fpix)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            y_hat_all = model(X)
            loss_epoch = loss_fn(y_hat_all, Y).item()
            loss_list.append(loss_epoch)
    
    with torch.no_grad():
        Y_pred = model(X).numpy()
        
    density = Y_pred.reshape(1, -1)
    _weight = 1/density
    df_cleaned['weights'] = _weight
    return df_cleaned
    
    
    