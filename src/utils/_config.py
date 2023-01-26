import os
import pathlib
from ._reader import DataReader


class Config(object):
    """
    This class must gather all the parameters of a project received from the database settings defined for a
    specific commodity that are required to run a forecast.

    Important: Changes here affect the integration with the automation of commodity desk, be cautious.
    """
    params = {}

    def __init__(self, params_dir: str):

        self.params = {}
        self.params_dir = params_dir

    def load(self, multitarget=False):

        reader = DataReader(path=self.params_dir)
        params_all = reader.read_yml(file_name='_config.yml')  # reading the project parameters
        if params_all['paths']['driversPath'].lower().endswith('drivers.json'):
            self.params['drivers_path'] = os.path.split(params_all['paths']['driversPath'])[0]
        else:
            self.params['drivers_path'] = params_all['paths']['driversPath']

        if params_all['paths']['forecastResultPath'].lower().endswith('forecast.json'):
            self.params['forecast_path'] = os.path.split(params_all['paths']['forecastResultPath'])[0]
        else:
            self.params['forecast_path'] = params_all['paths']['forecastResultPath']

        self.params['model_path'] = params_all['paths']['modelPath']
        self.params['input_path'] = pathlib.Path(params_all['paths']['combinedDataPath'])
        
        self.params['data_path'] = self.params["input_path"] / f"{params_all['target']['abbr']}.csv"
        self.params['backtesting_path'] = params_all['paths']['backtestingPath']
        self.params['simulation_path'] = params_all['paths']['simulationPath']
        self.params['combined_data_path'] = params_all['paths']['combinedDataPath']
        self.params['horizon'] = int(params_all['model']['kwargs']['horizon']) if 'horizon' in params_all['model'][
            'kwargs'].keys() else 3

        # Fetch model object
        self.params['model'] = params_all['model']

        if self.params["horizon"] < 1:
            raise Exception(
                f"Inappropriate value for Horizon: {self.params['horizon']}"
            )

        #self.params['frequency'] = params_all['target']['frequency']
        self.params['filter_column'] = int(params_all['target']['filter_column'])
        # self.params['target'] = params_all['target']['abbr']
        self.params['target_name'] = params_all['target']['abbr']
        self.params['target_display_name'] = "Historical " + params_all['target']['displayName']
        self.params['features'] = params_all['features']
        self.params['unit'] = params_all['target']['unit']

        return self.params