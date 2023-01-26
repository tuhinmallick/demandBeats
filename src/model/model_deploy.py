import ast
from model import model

def tryeval(val):
  try:
    val = ast.literal_eval(val)
  except:
    pass
  return val

def update_nested_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = tryeval(v)
    return d

def General(df, input_params):

    # Initialize params dictionary. Anything in this dictionary can be replaced by values in input_params in the call
    # to the update_nested_dict() function later in this function. Look through the code in this function to find which
    # parameters these are.
    params = {}

    # Extract some essential input parameters.
    df.index.freq = input_params["frequency"]
    target_name = input_params["target"]
    horizon = input_params["horizon"]


    # This specifies the "warm up" period for the base forecast model, if used. Must be minimum 2 full periods if
    # seasonality is used. You might want to reduce this to increase the amount of data available for training.
    params["base_forecast_params"] = {"lead": 24}

    # Set the size of the backtest and validation sets.
    params["backtest_periods"] = 24


    # Same with the hyperopt metric function.
    # hyperopt_metric_function = metrics.mean_absolute_error

    # Now update the params dictionary with any kwargs passed in
    params = update_nested_dict(params, input_params["model"]["kwargs"])

    # Not parameters, just processing
    # fruther preprocess of dataset is on model.py 
    Nbeats_model = model.nbeats_model(df, target_name, horizon, backtesting_period=params["backtest_periods"])
    df_forecast, df_backtesting= Nbeats_model.forecast()
    df_feature_importance = Nbeats_model.feature_importance()
  
    # forecast = df_forecast.rename(columns={target_name:'forecast'})
    # forecast.set_index(['date'], inplace= True)
    # feature_importance.set_index(['method'], inplace=True)

    actual = df[target_name]
    # forecast = forecast.astype(float)  
    return actual, df_forecast, df_feature_importance, df_backtesting

