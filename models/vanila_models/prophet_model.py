import itertools
from typing import Union, Iterable

import numpy as np
import pandas as pd
from prophet.diagnostics import cross_validation, performance_metrics

from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

from pathlib import Path
from tqdm import tqdm
from loguru import logger

from utilities.utils import export_dataframe_to_csv


class ProphetModel:
    def __init__(
            self,
            features: Union[Iterable[str], str] = None,
            model_params: dict = None,
    ) -> None:

        """
        Initialize a Prophet model for time series forecasting.

        Args:
            features (optional): Names of the explanatory variables (features) in form of a list of strings or a list.
                Default is None
            model_params (dict, optional): Dictionary of hyperparameters
        """

        self.y_hat = None  # Predicted values from the model

        self.hyper_param_space = None  # Hyperparameters for tuning the Prophet model

        if features:
            self.features = features if isinstance(features, list) else [features]
        else:
            self.features = []

        self.model_was_fit = False  # Flag indicating if the model has been fitted

        self.prophet_params = {}  # Parameters of a Prophet model
        self.prophet_model = None  # Prophet model
        self.hyper_param_space = None

        # Set Prophet hyperparameters based on provided args
        if model_params:
            self.prophet_params = model_params

        # If model has not been specified, initialize it
        if self.prophet_model is None:
            self.prophet_model = Prophet(**self.prophet_params)

    def add_holidays(
            self,
            country_names: Union[list, str],
    ) -> None:

        """
        Add country-specific holidays to the Prophet model.

        Args:
            country_names (str or list): Names of countries for which to include holidays in th Prophet model.

        """

        # Make sure that the country names are in form of a list
        if not isinstance(country_names, list):
            country_names = [country_names]

        # Add the country holidays to the model
        for country in country_names:
            self.prophet_model.add_country_holidays(country_name=country)

    def add_regressors(
            self,
    ) -> None:

        """
        Add additional regressors to the Prophet model.
        """

        # Add the additional regressors to the model
        for feature in self.features:
            self.prophet_model.add_regressor(feature)

    def fit_prophet(
            self,
            df_train: pd.DataFrame,
            country_names: Union[list, str] = None,
    ) -> Prophet:

        """
         Fit the Prophet model to the training data.

         Args:
             df_train (pd.DataFrame): The training dataset containing 'ds' (date) and 'y' (target) columns.
             country_names (str or list, optional): Names of countries for which holidays should be added.
                Default is None.

         Returns:
             prophet.model: The fitted Prophet model.

         """

        df_train = df_train.copy(deep=True)

        self.add_regressors()
        self.add_holidays(country_names)

        self.prophet_model.fit(df_train[['ds', 'y'] + self.features])

        # Make predictions using the fitted model
        self.predict_prophet(df_train[['ds'] + self.features])

        return self.prophet_model

    def grid_search_prophet(
            self,
            df: pd.DataFrame,
            target: str,
            hyper_param_space: dict,
            horizon: str,
            period: str = None,
            initial: str = None,
            parallel: str = "processes",
            country_names: Union[list, str] = None,
            metric_best: str = 'rmse',
            **kwargs,
    ) -> Prophet:

        """
         Fit the Prophet model to the training data.

         Args:
            df (pd.DataFrame): DataFrame containing data to perform grid search on. Must contain 'ds' (date) and
                'y' (target) columns.
            target (str): Target variable.
            hyper_param_space (dict): Hyperparameter space for model tuning.
            horizon (str): Parameter for prophet.diagnostics.cross_validation to define forecasted horizon.
                For more info refer to prophet.diagnostics.cross_validation. Default is None.
            period (str, optional): Parameter for prophet.diagnostics.cross_validation to define simulated forecast
                range period. For more info refer to prophet.diagnostics.cross_validation. Default is None.
            initial (str, optional): Parameter for prophet.diagnostics.cross_validation to define the first training
                period. For more info refer to prophet.diagnostics.cross_validation. Default is None
            parallel (str, optional): Parameter for prophet.diagnostics.cross_validation to define if and how to
                parallelize cross-validation. For more info refer to prophet.diagnostics.cross_validation.
                Default is 'processes'.
            country_names (str or list, optional): Names of countries for which holidays should be added.
                Default is None.
            metric_best (str, optional): Metric used to choose the best model from the CV procedure. Possible valid
                values are: ['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage']. Default is 'rmse'.
            **kwargs: Additional keyword arguments for prophet.diagnostics.cross_validation and
                prophet.diagnostics.performance_metrics functions. cross_validation should be referenced in the dict as
                'cv' and performance_metrics as 'pm'.

         Returns:
             prophet.model: The fitted Prophet model.
             pd.DataFrame: Predicted values on the training data.

         """

        self.hyper_param_space = hyper_param_space
        df.rename(columns={target: 'y'}, inplace=True)
        cv_kwargs = kwargs.get("cv") if kwargs.get("cv") else {}
        pm_kwargs = kwargs.get("pm") if kwargs.get("pm") else {}

        # Generate all combinations of parameters
        hyperparameter_combinations = [
            dict(zip(self.hyper_param_space.keys(), params))
            for params in itertools.product(*self.hyper_param_space.values())
        ]

        df_rmse = []  # Store the RMSEs for each params here

        # Use cross validation to evaluate all parameters
        for params in tqdm(hyperparameter_combinations):
            self.prophet_model = Prophet(**params)
            self.add_regressors()
            self.prophet_model.fit(df)  # Fit model with given params
            df_cv = cross_validation(
                self.prophet_model,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel=parallel,
                **cv_kwargs,
            )

            df_p = performance_metrics(df_cv, **pm_kwargs)
            df_rmse.append(df_p[metric_best].values.mean())

        # Find the best parameters and refit the model
        self.prophet_params = hyperparameter_combinations[np.argmin(df_rmse)]

        self.prophet_model = Prophet(**self.prophet_params)
        self.add_holidays(country_names)
        self.add_regressors()

        self.prophet_model.fit(df[['ds', 'y'] + self.features])

        # Make predictions using the fitted model
        self.predict_prophet(df[['ds'] + self.features])

        return self.prophet_model

    def predict_prophet(
            self,
            df_predict: pd.DataFrame,
    ) -> Union[pd.DataFrame, pd.Series]:

        """
        Make predictions using the fitted Prophet model.

        Args:
            df_predict (pd.DataFrame): Features and dates for prediction.

        Returns:
            pd.DataFrame: Predicted values for the specified dates.

        """

        self.y_hat = self.prophet_model.predict(df_predict)

        return self.y_hat

    def save_model(
            self,
            filename: str,
            output_dir: Union[Path, str],
    ) -> None:

        """
        Save the current Prophet model to a JSON file.

        Args:
            filename (str): The name of the model file.
            output_dir (Path or str): The path where the model file will be saved.

        """
        # Specify the extension and create a Path object
        file_extenstion = ".json"
        filename_model = filename + file_extenstion

        # Sanity check
        if self.prophet_model is None:
            raise ValueError("No model has been estimated. Estimate (or load) a model to save it.")

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        file_path = (output_dir / filename_model)

        with open(file_path, 'w') as fout:
            fout.write(model_to_json(self.prophet_model))

        logger.info(f"Prophet model successfully saved to {str(file_path)}.")

        # Export parameters used to estimate the final model
        if self.prophet_params is not None:
            if self.prophet_params != {}:
                if isinstance(self.prophet_params, dict):
                    params_export = pd.DataFrame.from_dict(self.prophet_params, orient='index')
                    params_export.columns = ['prophet_params']
                else:
                    params_export = self.prophet_params.copy()

                logger.info(f"Exporting Prophet model parameters...")

                export_dataframe_to_csv(
                    df=params_export,
                    output_path=output_dir,
                    filename=f"{filename}_params.csv",
                    index=True,
                )

    def load_model(
            self,
            file_path: Union[Path, str],
    ) -> None:

        """
        Load a pre-trained XGBoost model from a file.

        Args:
            file_path (Path or str): The path to the XGBoost model file.

        """

        logger.info(f"{'-' * 50}")

        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        with open(file_path, 'r') as fin:
            self.prophet_model = model_from_json(fin.read())  # Load model

        self.features = list(self.prophet_model.extra_regressors.keys())
        self.model_was_fit = True
        self.prophet_params = self.prophet_model.params

        logger.info(f"Prophet model successfully loaded from {str(file_path)}.")
