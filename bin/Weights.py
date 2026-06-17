import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

import torch

#Calculate systematic weights using
#1. Linear Regression
#2. Quadratic Regression
#3. Neural Network (from optuna best model)
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

SAVE_DIR = os.path.join(BASE_DIR, "output", "optuna_runs")
os.makedirs(SAVE_DIR, exist_ok=True)


def prepare_data(df, keys):
    df_cleaned = df.dropna()
    properties = df_cleaned[keys]
    
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(properties), columns= properties.columns)
    
    mean = np.sum(df_cleaned["target"]*df_cleaned["area"])/np.sum(df_cleaned["area"])
    density = df_cleaned["target"]/mean

    X = np.concatenate([np.array(df_standardized[key]).reshape(-1, 1) for key in keys], axis=1)

    train_X, test_X, train_Y, test_Y = train_test_split(X, density)
    return train_X, test_X, train_Y, test_Y
    

def linear_weights(Property, keys):
    """function to calculate weights using linear regression

    Parameters
    ----------------------------------------------------
    Property:table
    astropy table including imaging systematics and target density of each healpixels

    keys: list([string])
    list of imaging systematics to consider

    Output
    -----------------------------------------------------
    table: table
    astropy table including the imaging attributes, target density and imaging systematic weights named "lin_weight" for each pixel
    """
    df = Property.to_pandas()
    df_cleaned = df.dropna()
    train_X, test_X, train_Y, test_Y = prepare_data(df, keys)

    #learn using linear regression
    regr = LinearRegression()
    regr.fit(train_X, train_Y)

    predict_train = regr.predict(train_X)
    predicted_density = regr.predict(test_X)

    print("predicted density for test data:", predicted_density)

    # MSE for train data
    mse_train = mean_squared_error(train_Y, predict_train)
    print("Training MSE linear:", mse_train)

    # MSE for test data
    mse_test = mean_squared_error(test_Y, predicted_density)
    print("Test MSE linear:", mse_test)

    _density = regr.predict(X)
    _weight = 1/_density

    df_cleaned['lin_weight'] = _weight
    table = Table.from_pandas(df_cleaned)
    return table

def quadratic_weights(Property, keys):
    """function to calculate weights using quadratic regression

    Parameters
    ----------------------------------------------------
    Property:table
    astropy table including imaging systematics and target density of each healpixels

    keys: list([string])
    list of imaging systematics to consider

    Output
    -----------------------------------------------------
    table: table
    astropy table including the imaging attributes, target density and imaging systematic weights named "quad_weight" for each pixel
    """
    df = Property.to_pandas()
    df_cleaned = df.dropna()
    train_X, test_X, train_Y, test_Y = prepare_data(df, keys)
    
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

    # Mean Squared Errors
    mse_train = mean_squared_error(train_Y, predict_train)
    mse_test = mean_squared_error(test_Y, predicted_density)
    print("Training MSE quadratic:", mse_train)
    print("Test MSE quadratic:", mse_test)

    _density = regr.predict(X_poly)
    _weight = 1/_density
    
    df_cleaned['quad_weight'] = _weight
    table = Table.from_pandas(df_cleaned)
    return table


def nn_weights(Property, keys, config):
    """function to calculate weights using quadratic regression

    Parameters
    ----------------------------------------------------
    Property:table
    astropy table including imaging systematics and target density of each healpixels

    keys: list([string])
    list of imaging systematics to consider

    Output
    -----------------------------------------------------
    table: table
    astropy table including the imaging attributes, target density and imaging systematic weights named "nn_weight" for each pixel
    """
    
    # build_modelは既存の関数を利用（入力次元は len(keys) ）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filename = os.path.join(SAVE_DIR, 'optuna_runs/test/best_model.pt')
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
    else:
        print(f"{filename} does not exist")
        return None
    model = build_model(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    df = Property.to_pandas()
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
    _weight = np.where(density <= 0, np.nan, 1.0 / density)
    
    df_cleaned['nn_weight'] = _weight
    table = Table.from_pandas(df_cleaned)
    return table