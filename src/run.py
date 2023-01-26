import argparse, os
# import subprocess
import sys

import pathlib
import pandas as pd


dir = pathlib.Path(__file__).absolute()
sys.path.append(str(dir.parent.parent.parent))
script_path = pathlib.Path(__file__).parent


from utils import _logger as logger
from utils._config import Config

# another py file
from model import model_deploy

# Import custom libraries
if os.path.realpath("libs") not in sys.path:
    sys.path.append(os.path.realpath("libs"))


_logger = logger.config(handler='stdout')


def main():
    '''
        This setting consider that we can run the model either passing some arguments on the command line or using environment
        variables
    '''

    # Reading a parameters file  that gathers all the information required to run a customized model
    model_settings = Config(params_dir=params["params_path"]).load()
    params.update(model_settings)


    # The input file is saved with the name of the target the date column
    # The date column is standardize with the name Date
    data_path = os.path.join(params['data_path'],params['target']+'.csv')
    # if params["frequency"] == 'B':
    #     df = pd.read_csv(data_path)
    #     df['Date'] = pd.to_datetime(df["Date"], format='%d/%m/%Y')
    #     df = df.set_index(['Date'])
    #     df = df.reindex(df.index, fill_value=0)
    # else:
    df = pd.read_csv(data_path, index_col=0)


    # --->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Generate the dataframes by calling the main() function.
    actual, forecast, feature_importance, backtesting = model_deploy.General(df, params)
    print('Forecasting Done')


if __name__ == '__main__':

    try:
        params = {}
        parser = argparse.ArgumentParser(description="DemandForecasting.")

        if "PARAM_PATH" in os.environ:
            interface = "environment variable"
            # Get environment variables
            params["params_path"] = os.environ["PARAM_PATH"]

        else:
            interface = "command-line"
            parser.add_argument("--params_path", required=True)

        args = parser.parse_args()

        if interface == "command-line":
            params = args.__dict__

        main()

    except Exception as e:
        logger.error(_logger, str(e), exc_info=True)



