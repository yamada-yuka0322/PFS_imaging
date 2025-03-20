import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


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
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df), columns= df.columns)
    density = df_standardized["target"]

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
    _weight = 1/(_density*np.std(df['target'])/np.average(df['target'])+1)

    data = {
        'healpix' : df['healpix'],
        'weights' : _weight
    }

    return pd.DataFrame(data)

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
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df), columns= df.columns)
    density = df_standardized["target"]

    X = np.concatenate([np.array(df_standardized[key]).reshape(-1, 1) for key in keys], axis=1)

    # Train-test split
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
    _weight = 1/(_density*np.std(df['target'])/np.average(df['target'])+1)

    data = {
        'healpix' : df['healpix'],
        'weights' : _weight
    }

    return pd.DataFrame(data)    
