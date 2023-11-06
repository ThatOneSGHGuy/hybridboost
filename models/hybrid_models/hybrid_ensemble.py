from typing import Union, Tuple, Iterable
from pathlib import Path
from loguru import logger

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from prophet import Prophet

from models.vanila_models.prophet_model import ProphetModel
from models.ml_models.ensemble_models import XGBoostModel, RandomForestModel, LightGradientBoostingModel, EnsembleModel


def extract_prophet_features(
        df: pd.DataFrame,
        y_hat_prophet: pd.DataFrame,
        features: Union[Iterable[str], str],
        prophet_features: list = ['trend', 'additive_terms', 'multiplicative_terms', 'holidays', 'weekly',
                                  'yearly'],
):

    """
      Extract relevant features from the Prophet forecast.

      Args:
          df (pd.DataFrame): Input data.
          y_hat_prophet (pd.DataFrame): Prophet forecast dataset containing 'ds' (date), target variable 'y'
            and at least all prophet features specified in the next prophet_features arg.
          features: Names of the explanatory variables (features) in form of a list of strings or a list.
          prophet_features (list, optional): List of features to be extracted from the Prophet forecast which are
            then to be used in the ensemble model fitting. Default is ['trend', 'additive_terms',
            'multiplicative_terms', 'holidays', 'weekly', 'yearly'].

      Returns:
          Tuple[pd.DataFrame, List[str]]:
              A tuple containing dataframe with selected Prophet features and all "basic " features specified as
              the class attribute, and a list of all column names.

      """

    df = df.copy(deep=True)
    features = features if isinstance(features, list) else [features]

    # Extract relevant df_features from the Prophet forecast
    df_with_prophet_features = y_hat_prophet[list(prophet_features) + features]

    # Mark the prophet columns and create train dataframe to pass into ensemble model
    df_with_prophet_features.columns = [name + "_prophet" for name in df_with_prophet_features.columns]
    df_with_prophet_features = pd.concat(
        [df[features + ['y']], df_with_prophet_features],
        axis=1,
    )
    df_with_prophet_features['y'] = df_with_prophet_features['y'] - y_hat_prophet.yhat
    features_with_prophet_features = [x for x in df_with_prophet_features.columns if x != 'y']

    return df_with_prophet_features, features_with_prophet_features


class BoostedProphetModel:

    """
    Mixed model consisting of Prophet model and a kind of ensemble model, both of which are subject
     to configuration of hyperparameters.

    Attributes
    ----------
    y_hat: Predicted target values.
    target: Target variable.
    ensemble_model: The ensemble model instance.
    prophet_model: Prophet model instance
    ensemble_model_type: Chosen ensemble model i.e. 'xgb', 'lgbm' or 'rf'.
    """

    def __init__(
            self,
            target: str,
            ensemble_model_type: str,
    ) -> None:

        """
        Initialize Boosted Prophet Model.

        Args:
            target (str): The name of the target variable.
            ensemble_model_type (str): The type of boosting model to use ('xgb', 'rf', or 'lgbm').

        Raises:
            ValueError: If `ensemble_model_type` is not one of 'xgb', 'rf', or 'lgbm'.

        """

        self.y_hat = None
        self.y_hat_prophet = None

        self.target = target
        self.features = None

        self.ensemble_model_type = None
        self._prophet_model = None  # Private attribute to store ProphetModel instance
        self._ensemble_model = None

        if ensemble_model_type not in ['xgb', 'rf', 'lgbm']:
            raise ValueError(
                f"ensemble_model_type needs to be one of 'xgb', 'rf', 'lgbm'. {ensemble_model_type} was supplied"
            )
        else:
            self.ensemble_model_type = ensemble_model_type

        self.ensemble_model_dict = {
            "xgb": XGBoostModel,
            "rf": RandomForestModel,
            "lgbm": LightGradientBoostingModel,
        }

    @property
    def prophet_model(self):
        return self._prophet_model

    @prophet_model.setter
    def prophet_model(self, prophet_model_instance):
        if isinstance(prophet_model_instance, ProphetModel):
            self._prophet_model = prophet_model_instance
            # Set the features of BoostedProphetModel from ProphetModel
            self.features = prophet_model_instance.features
        else:
            raise ValueError("prophet_model must be an instance of ProphetModel")

    @property
    def ensemble_model(self):
        return self._ensemble_model

    @ensemble_model.setter
    def ensemble_model(self, ensemble_model_instance):
        if isinstance(ensemble_model_instance, EnsembleModel):
            self._ensemble_model = ensemble_model_instance
            # TODO: If a model is loaded, check if feature names are the same for Prophet and ensemble
        else:
            raise ValueError("ensemble_model must be an instance of EnsembleModel")

    def fit_hybrid(
            self,
            df_train: pd.DataFrame,
            features: Union[Iterable[str], str],
            country_holiday_prophet: list = None,
            prophet_params: dict = {},
            ensemble_params: dict = {},
            fit_ensemble_dict: dict = {},
            which_to_fit: list = ["prophet", "ensemble"],
            **kwargs,
    ) -> Tuple[Prophet, Union[xgb.XGBRegressor, xgb.Booster, lgb.LGBMRegressor, lgb.Booster, RandomForestRegressor]]:

        """
          Fit combined Prophet and ensemble models to the training data.

          Args:
              df_train (pd.DataFrame): The training dataset containing 'ds' (date) and the target variable 'y'.
              features: Names of the explanatory variables (features) in form of a list of strings or a list.
              country_holiday_prophet (str or list, optional): Names of countries for which to include holidays in the
                Prophet model. Default is None.
              prophet_params (dict, optional): Dictionary of specified Prophet hyperparameters. Default is None.
              ensemble_params (dict, optional): Dictionary of specified hyperparameters for an ensemble model.
                Default is None.
              fit_ensemble_dict (dict, optional): Dictionary of keyword args for an ensemble model fit function call.
                Default is None.
              which_to_fit (list, optional): List of which models are supposed to be fit. Specyfing only one implies
                that the second one has already been fit. Default is ["prophet", "ensemble"].
              **kwargs: Dictionary of keyword arguments to pass into extract_prophet_features method.

          Returns:
              Tuple[Prophet, Union[xgb.XGBRegressor, xgb.Booster, lgb.LGBMRegressor, lgb.Booster,
                RandomForestRegressor]]:
                  A tuple containing the fitted Prophet model and the ensemble model.

          """

        logger.info(f"{'-' * 50}")

        # Deep copy the data to avoid modifying the original
        df_train = df_train.copy(deep=True)
        df_train.reset_index(drop=True, inplace=True)
        df_train.rename(columns={self.target: 'y'}, inplace=True)
        self.features = features if isinstance(features, list) else [features]

        # Estimate Prophet model
        if "prophet" in which_to_fit:
            self.prophet_model = ProphetModel(model_params=prophet_params, features=self.features)
            self.prophet_model.fit_prophet(
                df_train=df_train,
                country_names=country_holiday_prophet,
            )
            logger.info(f"Prophet model has been successfully fit.")
            self.y_hat_prophet = self.prophet_model.y_hat
        else:
            self.y_hat_prophet = self.prophet_model.predict_prophet(df_train[['ds'] + self.features])

        df_train_with_prophet_features, features_with_prophet_features = extract_prophet_features(
            df=df_train,
            y_hat_prophet=self.y_hat_prophet,
            features=self.features,
            **kwargs,
        )

        if "ensemble" in which_to_fit:
            # Initialize ensemble model
            self.ensemble_model = self.ensemble_model_dict.get(self.ensemble_model_type)(
                target='y',
                model_params=ensemble_params,
                features=features_with_prophet_features,
            )

            # Estimate ensemble model
            self.ensemble_model.fit_ensemble(
                df_train=df_train_with_prophet_features,
                **fit_ensemble_dict,
            )
            logger.info(f"Ensemble model ({self.ensemble_model_type}) has been successfully fit.")

        # Get residual predictions using ensemble model
        residuals_ensemble = self.ensemble_model.predict_ensemble(
            df_train_with_prophet_features[features_with_prophet_features],
        )

        # Get final y_hat predictions using combined models
        self.y_hat = self.y_hat_prophet.yhat + residuals_ensemble

        return self.prophet_model, self.ensemble_model

    def predict_combined(
            self,
            df_predict: pd.DataFrame = None,
            prophet_features: list = ['trend', 'additive_terms', 'multiplicative_terms', 'holidays', 'weekly',
                                      'yearly'],
    ) -> pd.Series:

        """
        Make combined predictions using trained models.

        Args:
            df_predict (pd.DataFrame, optional): The input DataFrame for prediction. Default is None.
            prophet_features (list): List of features to be extracted from the Prophet model which are then to be used
                in the ensemble model fitting.
        Returns:
            pd.Series: Combined predictions.

        """

        # Make predictions using trained models
        df_predict = df_predict.copy(deep=True)
        df_predict.reset_index(drop=True, inplace=True)
        self.y_hat_prophet = self.prophet_model.predict_prophet(df_predict=df_predict)

        # Extract relevant df_features from the Prophet forecast
        df_predict_with_prophet_features = self.y_hat_prophet[prophet_features + self.features]

        # Mark the prophet columns and create train dataframe to pass into ensemble model
        df_predict_with_prophet_features.columns = [name + "_prophet"
                                                    for name in df_predict_with_prophet_features.columns]
        df_predict_with_prophet_features = pd.concat(
            [df_predict[self.features], df_predict_with_prophet_features],
            axis=1,
        )

        # Get residual predictions using ensemble model
        residuals_ensemble = self.ensemble_model.predict_ensemble(df_predict=df_predict_with_prophet_features)

        # Combine the Prophet and LightGBM predictions for the test data
        self.y_hat = self.y_hat_prophet.yhat + residuals_ensemble
        self.y_hat.index = df_predict.index

        return self.y_hat

    def save_models(
            self,
            prophet_filename: str = None,
            ensemble_filename: str = None,
            output_path: Union[Path, str] = None,
            **kwargs,
    ) -> None:

        """
        Save ensemble and Prophet models to specified output paths.

        Args:
            prophet_filename (str, optional): Filename for the Prophet model.
            ensemble_filename (str, optional): Filename for the ensemble model.
            output_path (Path): The path where the models will be saved.

        Raises:
            ValueError: If either the ensemble or the Prophet model has not been estimated.

        """

        # Sanity check
        if self.ensemble_model and self.prophet_model:
            # Save all models which have been loaded/estimated
            if ensemble_filename and prophet_filename:
                self.ensemble_model.save_model(
                    filename=ensemble_filename,
                    output_dir=output_path,
                    **kwargs,
                )
                self.prophet_model.save_model(
                    filename=prophet_filename,
                    output_dir=output_path,
                )

            elif ensemble_filename is None and prophet_filename is not None:
                logger.info(
                    "Arguments to save ensemble model have not been specified. Saving only Prophet model."
                )
                self.prophet_model.save_model(
                    filename=prophet_filename,
                    output_dir=output_path,
                )

            elif ensemble_filename is not None and prophet_filename is None:
                logger.info(
                    "Arguments to save Prophet model have not been specified. Saving only ensemble model."
                )
                self.ensemble_model.save_model(
                    filename=ensemble_filename,
                    output_dir=output_path,
                    **kwargs,
                )
        else:
            raise ValueError(
                "Either ensemble or prophet model has not been estimated. Estimate (or load) the model to save it."
            )

    def load_models(
            self,
            prophet_file_path: Union[Path, str] = None,
            ensemble_file_path: Union[Path, str] = None,
            **kwargs,
    ):

        """
        Load pre-trained model(s) from a file.

        Args:
            prophet_file_path (Path or str): The path to the Prophet model file. Default is None.
            ensemble_file_path (Path or str): The path to the ensemble model file. Default is None.
            **kwargs: Additional keyword args used in loading the model.

        Raises:
            ValueError: If both of the paths were not specified.

        """

        if prophet_file_path or ensemble_file_path:
            if prophet_file_path:
                if not isinstance(prophet_file_path, Path):
                    prophet_file_path = Path(prophet_file_path)
                self.prophet_model = ProphetModel()
                self.prophet_model.load_model(prophet_file_path)
            if ensemble_file_path:
                if not isinstance(ensemble_file_path, Path):
                    ensemble_file_path = Path(ensemble_file_path)
                self.ensemble_model = self.ensemble_model_dict.get(self.ensemble_model_type)(
                    target='y',
                )
                self.ensemble_model.load_model(file_path=ensemble_file_path, **kwargs)
            self.features = self.prophet_model.features
        else:
            raise ReferenceError(
                "Neither prophet_file_path nor ensemble_file_path were specified."
            )
