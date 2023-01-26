import pandas as pd
import numpy as np
from darts.models import NBEATSModel
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

from sklearn import preprocessing
import lightgbm as lgb
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from darts.dataprocessing.transformers import (
    Scaler,
    # MissingValuesFiller,
)
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# from sklearn.metrics import mean_absolute_percentage_error

class nbeats_model():

    def __init__(self,  df, target_name, horizon, backtesting_period):
        self.df = df
        self.target_name = target_name
        self.horizon = horizon
        self.backtesting_period = backtesting_period


    
    ############################################################################################################################################
    ####
    #### Data Preprocess
    ####
    ############################################################################################################################################
    def data_preprocess(self, df, target_name):
        """ This function is to preprocess the dataset, including cleaning null, filling the missing data and transform the formt of dataset into Time Series type. 
        The function also splits the dataset into two datasets, df_target and df_covariates. If the dataset is univariate series, df_covariates will be an empty DataFrame. If the dataset is multivariate series, df_covariates will be TimeSeries and series of covariate series. 

        Args:
            df (DataFrame): the input dataset 
            target_name (str): the name of target series

        Returns:
            df_target(TimeSeries): the target series 
            df_covariates(TimeSeries or DataFrame): if the df is an Univariate series, df_covariate is an empty DataFrame; Else, it will be covariate series in Time Series form. 
        """
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']) 

        # check the columns with NaN and fill the missing values
        nan_names = df.columns[df.isnull().any()].tolist()
        # if there is a column that NaN values, then process the columns
        if len(nan_names) > 0:
            df[nan_names] = df[nan_names].interpolate()
            df[nan_names] = df[nan_names].bfill()

        # get the names of covariates
        covariate_names = list(df.columns)
        covariate_names.remove(target_name)
        covariate_names.remove('Date')

        # transform the data into TimeSeries, which is neccessary for Nbeats
        df_target = TimeSeries.from_dataframe(df, 'Date', [target_name])        # target series
        if len(covariate_names) > 0:
            df_covariates = TimeSeries.from_dataframe(df, 'Date', covariate_names)  # covariates series
        else:
            df_covariates = pd.DataFrame()
            print('It is univariate Time Series')

        return df_target, df_covariates



    ############################################################################################################################################
    ####
    #### Feature Importance 
    ####
    ############################################################################################################################################
    def feature_importance(self):
        """ This function applies LightGBM function to get the Feature Importance. For LightGBM, y is the target series and x are the covariate series. Additionally the target_series_lag_1 is added into x and therefore it can obtain the feature importance of target seires itself. 
        For Univariate, the feature importance is simply 100%. 

        Args:
            df (DataFrame): the input dataset
            target_name (str): the name of target series

        Returns:
            df_fi (DataFrame): feature importance of variates in df
        """

        # df contains at least 2 columns, 'date' and 'target_series'. So if the number of columns <= 2, meaning it is Univariate series. 
        if self.df.shape[1] <= 2:
            df_fi = pd.DataFrame(
            {
            'method': self.target_name,
            'agg': [100]
            }) 
        else: 
            self.df[str(self.target_name+'_lag_1')] = self.df[self.target_name].shift(-1)
            # detect the columns that have NaN values 
            nan_names = self.df.columns[self.df.isnull().any()].tolist()

            # missing values, NaN
            if len(nan_names) > 0:
                self.df[nan_names] = self.df[nan_names].interpolate()
                self.df[nan_names] = self.df[nan_names].bfill()
            
            df_scaled = pd.DataFrame(preprocessing.normalize(self.df, axis=0), columns=self.df.columns, index=self.df.index)
            train_y = df_scaled[self.target_name]
            train_x = df_scaled.drop([self.target_name], axis=1)

            # train the model to get the future importance
            model = lgb.LGBMRegressor()
            model.fit(train_x, train_y)

            feature_name = list(self.df.columns)
            feature_name.remove(self.target_name)
            df_fi = pd.DataFrame(
            {
                'method': feature_name,
                'agg': (model.feature_importances_/sum(model.feature_importances_))*100
            })  

        df_fi.set_index(['method'], inplace=True)
        
        return df_fi


    ############################################################################################################################################
    ####
    #### Confidence Interval
    ####
    ############################################################################################################################################
    def confident_interval(self, df_train, df_forecast, t=0.8): 
        """This function is to get the confidence interval of forecast. 

        Args:
            df_train (TimeSeries): _description_
            df_forecast (DataFrame): _description_
            t (float, optional): can be seen as a constant scaler. Defaults to 0.8.

        Returns:
            df_forecast (DataFrame): the forecast dataframe has forecast values and their accordingly upper/lower values
        """
        if isinstance(df_train, TimeSeries):
            df_train = df_train.pd_dataframe()         # because of Darts, the type of training dataset is TimeSeries, here change it into DataFrame
                
        mse = np.var(df_train.values)   
        y_h = df_train.values[-1].item()                # item(): get the value from array
        n = len(df_train)
        denominator = mse*n
        molecule = y_h - np.mean(df_train.values)
        term = np.sqrt(mse*(1/n + 1 + molecule/denominator))
        y_upper = df_forecast['forecast'].values + t*term 
        y_lower = df_forecast['forecast'].values  - t*term 

        df_forecast['upper'] = y_upper
        df_forecast['lower'] = y_lower

        return df_forecast
 

    ############################################################################################################################################
    ####
    #### Forecast
    ####
    ############################################################################################################################################
    # for Nbeats model, the input:dataframe
    def forecast(self):
        """ This function is to get the forecast values and backtesting forecast values. 

        Args:
            df (DataFrame): input dataset for model
            horizon (int): forecast horizon
            target_name (str): the name of target

        Returns:
            _type_: _description_
        """
        # parameter settings,
        # df: TimeSeries

        ###############################################
        #### Data Preprocess
        ###############################################
        df_target, df_covariates = self.data_preprocess(self.df, self.target_name)           # df would be transformed into TimeSeries type and split into Target/Covariate dataframe
        scaler_target = Scaler(MinMaxScaler(feature_range=(0.01, 1)))         # for MAPE, the values have to be strictly positive. Therefore, y_actual (denominator in MAPE) has to be positive and can not be 0
        scaler_target.fit_transform(df_target)
        df_target_scaled = scaler_target.transform(df_target)

        # NOTE: after data_preprocess(), if it is multivariate series, type(df_covariates) will be TimeSeries for Nbeats and non-empty. But if it is Univariate, df_covariate will be just an empyty dataframe
        if isinstance(df_covariates, TimeSeries):
            scaler_covariate = Scaler(MinMaxScaler(feature_range=(0.01, 1)))         # for MAPE, the values have to be strictly positive. Therefore, y_actual (denominator in MAPE) has to be positive and can not be 0
            scaler_covariate.fit_transform(df_covariates)
            df_covariates_scaled = scaler_covariate.transform(df_covariates)
        


        ###############################################
        #### Grid Search
        ###############################################
        # NOTE: Regarding the order of parameters, please follow its accordingly order for Nbeats model. Also, there is a specific requirement for the input/output_chunk_length. For more information, plrease refer to https://unit8co.github.io/darts/examples/01-multi-time-series-and-covariates.html#Training-Process-(behind-the-scenes)
        param_dict = {
            'input_chunk_length': [7, 7*2, 7*3, 7*4, 7*5],
            'output_chunk_length': [1, 2, 3, 4, 5, 6, 7],
            'generic_architecture': [True, False],
        }

        k = len(df_target) - max(param_dict['input_chunk_length']) - max(param_dict['output_chunk_length']) + 1     # k: number of samples used for training  
        if k <= 0:
            if len(param_dict['input_chunk_length']) <= 1:
                param_dict['output_chunk_length'].remove(max(param_dict['output_chunk_length']))
            else:
                param_dict['input_chunk_length'].remove(max(param_dict['input_chunk_length']))

        Input_chunk_length = 7
        Output_chunk_length = 1
        Generic_architecture = True

        
        # model = model.reset_model()        # Resets the model object and removes all stored data - model, checkpoints, loggers and training history. 


        # uses the tunned parameters from Grid Search 
        model = NBEATSModel(input_chunk_length = Input_chunk_length, output_chunk_length = Output_chunk_length, generic_architecture = Generic_architecture)  
        if isinstance(df_covariates, TimeSeries):
            # for covariates, output_chunk_lenghth has to be greater than forecast horizon
            if Output_chunk_length < self.horizon:
                Output_chunk_length = self.horizon
                model = model.reset_model()
                model = NBEATSModel(input_chunk_length = Input_chunk_length, output_chunk_length = Output_chunk_length, generic_architecture = Generic_architecture)  
            model.fit(series=df_target_scaled, past_covariates=df_covariates_scaled)   # multivariate      
            pred = model.predict(n=self.horizon, series=df_target_scaled, past_covariates=df_covariates_scaled)
        else: 
            model.fit(df_target_scaled)       # univariate
            pred = model.predict(n=self.horizon, series=df_target_scaled)
        

        pred = scaler_target.inverse_transform(pred)
        pred_df = pred.pd_dataframe()
        pred_df.columns = ['forecast']



        # ###############################################
        # #### Confidence Interval
        # ###############################################
        pred_df = self.confident_interval(self.df[self.target_name], pred_df)

        # getting the last values from training datasets, pd.DataFrame()
        last_row = df_target[-1:].pd_dataframe()
        last_row.columns =['forecast']
        last_row['lower'] = last_row['forecast'].values
        last_row['upper'] = last_row['forecast'].values
        

        forecast_df = pd.concat([last_row, pred_df])




        ###############################################
        #### Backtesting 
        ###############################################
        forecast_steps = range(1,self.horizon+1)
        backtesting_actual = df_target[-int(self.backtesting_period):].pd_dataframe().values    # y_actual on backtesting period
        backtesting_df = pd.DataFrame()              # df for storing forecast on bactesting period
        i = 0 
        for step in forecast_steps: 
            model = model.reset_model()         # Resets the model, remove the training history ect from forecast 
            model = NBEATSModel(input_chunk_length = Input_chunk_length, output_chunk_length = Output_chunk_length)
            backtesting_time_stampt = df_target_scaled[-int(self.backtesting_period)-i:].pd_dataframe().index[0] # Time stampt formt, such as '2020/01/01'

            if isinstance(df_covariates, TimeSeries):
                # Multivariate forecast
                pred_backtesting = model.historical_forecasts(df_target_scaled, past_covariates=df_covariates_scaled,start=backtesting_time_stampt, forecast_horizon=step, last_points_only=True)
            else:
                # Univariate forecast
                pred_backtesting = model.historical_forecasts(df_target_scaled, start=backtesting_time_stampt, forecast_horizon=step, last_points_only=True)
            
            pred_inverse = scaler_target.inverse_transform(pred_backtesting)  # inverse the scaled values
            tempt_df = pred_inverse.pd_dataframe()    # tempt_df: temporary df storing the forecast on each step 
            tempt_df.columns = ['forecast']
            tempt_df['step'] = step 
            tempt_df['actual'] = backtesting_actual
            
            backtesting_df = pd.concat([backtesting_df, tempt_df])  # merge the dataframe of different steps of forecast

            i += 1

        # # # get confidence interval for backtesting forecast 
        bt_df = self.confident_interval(df_target, backtesting_df)
        bt_df = bt_df.reset_index()
        bt_df = bt_df.rename(columns={'time':'date'})
        bt_df = bt_df[['date', 'step', 'actual', 'forecast', 'upper', 'lower']]
        bt_df[['actual','forecast', 'upper', 'lower']] = bt_df[['actual','forecast', 'upper', 'lower']].astype(float) # error
        bt_df['step'] = bt_df['step'].astype(int)


        return forecast_df, bt_df
