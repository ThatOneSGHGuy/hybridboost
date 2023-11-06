import pandas as pd

import joblib
from pathlib import Path
from typing import Union, Iterable
from loguru import logger

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

from utilities.utils import export_dataframe_to_csv


class EnsembleModel:

    """
    Super class to initialize an ensemble model with configurable hyperparameters.

    Attributes
    ----------
    hyper_tuning_results: Hyperparameter tuning results.
    hyper_param_space: Hyperparameter space for model tuning. If they are None, the model won't be fine-tuned and the
        parameters will be either default or taken as user-input.
    y_hat: Predicted target values.
    ensemble_params: Parameters used to estimate the model. In case of hyperparameter tuning, only the parameters of
        the best model will be stored.
    ensemble_model: The ensemble model instance.
    target: Target variable.
    features: Names of the explanatory variables (features).
    """

    def __init__(
            self,
            target: str,
            features: Union[Iterable[str], str] = None,
    ) -> None:

        """
        Initialize an abstract EnsembleModel class.

        Args:
            target (str): Target variable.
            features (optional): Names of the explanatory variables (features) in form of a list of strings or a list.
                Default is None
        """

        self.hyper_tuning_results = None
        self.hyper_param_space = None

        if features:
            self.features = features if isinstance(features, list) else [features]
        else:
            self.features = []

        self.y_hat = None  # Predicted values from the model

        self.ensemble_params = {}  # Parameters of a chosen model
        self.ensemble_model = None  # Chosen ensemble model
        self.model_was_fit = False  # Flag indicating if the model has been fitted

        self.target = target  # Target variable

    def fit_ensemble(
            self,
            df_train: pd.DataFrame,
            **kwargs,
    ) -> Union[xgb.XGBRegressor, xgb.Booster, lgb.LGBMRegressor, lgb.Booster, RandomForestRegressor]:

        """
        Fit the ensemble model to the training data and tune hyperparameters if specified.

        Args:
            df_train (pd.DataFrame): DataFrame containing training data.
            **kwargs: Additional keyword arguments for fit function corresponding to a chosen model.

        Returns:
            object: Fitted ensemble model.

        """

        # Deep copy the data to avoid modifying the original
        df_train = df_train.copy(deep=True)
        y_train = df_train[self.target]
        df_features = df_train[self.features]

        # Fit the ensemble model on the training data (parameters are either chosen by user or default)
        self.ensemble_model.fit(df_features, y_train, **kwargs)
        self.ensemble_params = self.ensemble_model.get_params()
        self.model_was_fit = True

        return self.ensemble_model

    def grid_search_ensemble(
            self,
            df: pd.DataFrame,
            hyper_param_space: dict,
            n_jobs: int = -1,
            verbose: float = 2,
            scoring: str = 'neg_mean_squared_error',
            refit: bool = True,
            **kwargs,
    ) -> Union[xgb.XGBRegressor, xgb.Booster, lgb.LGBMRegressor, lgb.Booster, RandomForestRegressor]:

        """
        Fit the ensemble model to the training data and tune hyperparameters if specified.

        Args:
            df (pd.DataFrame): DataFrame containing data to perform grid search on.
            hyper_param_space (dict): Hyperparameter space for model tuning.
            n_jobs (int): Number of CPUs to perform grid search with.
            verbose (float): Level of verbosity of the grid search.
            scoring (str): Evaluation metric in the grid search.
            refit (bool): Flag to specify whether the model should be fit once more after grid search with
                the best parameters.
            **kwargs: Additional keyword arguments for sklearn.model_selection.GridSearchCV function.

        Returns:
            object: Fitted ensemble model.

        """

        self.hyper_param_space = hyper_param_space

        # Deep copy the data to avoid modifying the original
        df = df.copy(deep=True)
        df_target = df[self.target]
        df_features = df[self.features]

        # Hyperparameter tuning using Grid Search
        rsearch = GridSearchCV(
            estimator=self.ensemble_model,
            refit=refit,
            verbose=verbose,
            n_jobs=n_jobs,
            param_grid=self.hyper_param_space,
            scoring=scoring,
            **kwargs,
        )

        rsearch.fit(df_features, df_target)

        # Get the best model along with its parameters
        self.hyper_tuning_results = rsearch
        self.ensemble_params = rsearch.best_params_
        self.ensemble_model = rsearch.best_estimator_
        self.model_was_fit = True

        return self.ensemble_model

    def predict_ensemble(
            self,
            df_predict: pd.DataFrame,
    ) -> pd.Series:

        """
        Make predictions using the ensemble model.

        Args:
            df_predict (pd.DataFrame): Features for prediction.

        Returns:
            pd.Series: Predicted target values.

        """

        df_predict = df_predict.copy(deep=True)

        # If the ensemble model is a Booster class (XGBoost), the training data passed needs to be in form of DMatrix
        if type(self.ensemble_model) == xgb.sklearn.Booster:
            df_predict = xgb.DMatrix(df_predict)

        # Make predictions using estimated model
        self.y_hat = self.ensemble_model.predict(df_predict)

        return self.y_hat


class XGBoostModel(EnsembleModel):

    """
    XGBoost model wrapper.

    Attributes
    ----------
    ensemble_params: Parameters used to estimate the model. In case of hyperparameter tuning, only the parameters of
        the best model will be stored.
    model_was_fit: Flag indicating whether the model has been already fitted.
    hyper_param_space: Hyperparameter space for model tuning. If they are None, the model won't be fine-tuned and the
        parameters will be either default or taken as user-input.
    """

    def __init__(
            self,
            target: str,
            features: Union[Iterable[str], str] = None,
            model_params: dict = None,
    ) -> None:

        """
        Initialize an XGBoost model.

        Args:
            target (str): Target variable.
            features (optional): Names of the explanatory variables (features) in form of a list of strings or a list.
                Default is None
            model_params (dict, optional): Dictionary of hyperparameters
        """

        super().__init__(target=target, features=features)  # Call the constructor of the EnsembleModel superclass

        if model_params:
            self.ensemble_params = model_params

        # Initialize the ensemble_params attribute if it's still None
        if self.ensemble_params is None:
            self.ensemble_params = {}

        # Create an XGBoost model with specified or default hyperparameters
        if self.ensemble_model is None:
            if self.hyper_param_space is None:
                self.ensemble_model = xgb.XGBRegressor(**self.ensemble_params)
            else:
                self.ensemble_model = xgb.XGBRegressor()

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

        self.ensemble_model = xgb.Booster()
        self.ensemble_model.load_model(file_path)

        self.model_was_fit = True
        self.ensemble_params = self.ensemble_model.save_config()
        self.features = self.ensemble_model.feature_names

        logger.info(f"XGBoost model successfully loaded from {str(file_path)}.")

    def save_model(
            self,
            filename: str,
            output_dir: Union[Path, str],
    ) -> None:

        """
        Save the current XGBoost model to a file.

        Args:
            filename (str): The name of the model file.
            output_dir (Path or str): The path where the model file will be saved.

        Raises:
            ValueError: If no model has been estimated before saving.

        """

        logger.info(f"{'-' * 50}")

        # Sanity check
        if self.ensemble_model is None:
            raise ValueError("No model has been estimated. Estimate (or load) a model to save it.")

        # Specify the extension and create a Path object
        file_extenstion = ".json"
        filename_model = filename + file_extenstion

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        file_path = (output_dir / filename_model)

        self.ensemble_model.save_model(file_path)

        logger.info(f"XGBoost model successfully saved to {str(file_path)}.")

        # Export parameters used to estimate the model
        if self.ensemble_params is not None and self.ensemble_params != {}:
            if isinstance(self.ensemble_params, dict):
                params_export = pd.DataFrame.from_dict(self.ensemble_params, orient='index')
                params_export.columns = ['xgb_params']
            else:
                params_export = self.ensemble_params.copy()

            logger.info(f"Exporting XGBoost model parameters...")

            export_dataframe_to_csv(
                df=params_export,
                output_path=output_dir,
                filename=f"{filename}_params.csv",
                index=True,
            )


class RandomForestModel(EnsembleModel):

    """
    RandomForest model wrapper.

    Attributes
    ----------
    ensemble_params: Parameters used to estimate the model. In case of hyperparameter tuning, only the parameters of
        the best model will be stored.
    model_was_fit: Flag indicating whether the model has been already fitted.
    hyper_param_space: Hyperparameter space for model tuning. If they are None, the model won't be fine-tuned and the
        parameters will be either default or taken as user-input.
    """

    def __init__(
            self,
            target: str,
            features: Union[Iterable[str], str] = None,
            model_params: dict = None,
    ) -> None:

        """
        Initialize Random Forest model.

        Args:
            target (str): Target variable.
            features (optional): Names of the explanatory variables (features) in form of a list of strings or a list.
                Default is None
            model_params (dict, optional): Dictionary of hyperparameters
        """

        super().__init__(target=target, features=features)  # Call the constructor of the EnsembleModel superclass

        if model_params:
            self.ensemble_params = model_params

        # Initialize the ensemble_params attribute if it's still None
        if self.ensemble_params is None:
            self.ensemble_params = {}

        # Create a Random Forest model with specified or default hyperparameters
        if self.ensemble_model is None:
            if self.hyper_param_space is None:
                self.ensemble_model = RandomForestRegressor(**self.ensemble_params)
            else:
                self.ensemble_model = RandomForestRegressor()

    def load_model(
            self,
            file_path: Union[Path, str],
            **kwargs,
    ) -> None:

        """
        Load a pre-trained Random Forest model from a file.

        Args:
            file_path (Path or str): The path to the model file.

        """

        logger.info(f"{'-' * 50}")

        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        self.ensemble_model = joblib.load(file_path, **kwargs)

        self.model_was_fit = True
        self.ensemble_params = self.ensemble_model.get_params()
        self.features = self.ensemble_model.feature_names_in_

        logger.info(f"Random Forest model successfully loaded from {str(file_path)}.")

    def save_model(
            self,
            filename: str,
            output_dir: Union[Path, str],
            **kwargs,
    ) -> None:

        """
        Save the current Random Forest model to a file.

        Args:
            filename (str): The name of the model file.
            output_dir (Path or str): The path where the model file will be saved.
            **kwargs: Keyword arguments to be passed into joblib.dump function.

        Raises:
            ValueError: If no model has been estimated before saving.

        """

        # Sanity check
        if self.ensemble_model is None:
            raise ValueError("No model has been estimated. Estimate (or load) a model to save it.")

        # Specify the extension and create a Path object
        file_extenstion = ".joblib"
        filename_model = filename + file_extenstion

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        file_path = (output_dir / filename_model)

        joblib.dump(self.ensemble_model, file_path, **kwargs)

        logger.info(f"Random Forest model successfully saved to {str(file_path)}.")

        # Export parameters used to estimate the model
        if self.ensemble_params is not None and self.ensemble_params != {}:
            if isinstance(self.ensemble_params, dict):
                params_export = pd.DataFrame.from_dict(self.ensemble_params, orient='index')
                params_export.columns = ['rf_params']
            else:
                params_export = self.ensemble_params.copy()

            logger.info(f"Exporting Random Forest model parameters...")

            export_dataframe_to_csv(
                df=params_export,
                output_path=output_dir,
                filename=f"{filename}_params.csv",
                index=True,
            )


class LightGradientBoostingModel(EnsembleModel):

    """
    LightGBM model wrapper.

    Attributes
    ----------
    ensemble_params: Parameters used to estimate the model. In case of hyperparameter tuning, only the parameters of
        the best model will be stored.
    model_was_fit: Flag indicating whether the model has been already fitted.
    hyper_param_space: Hyperparameter space for model tuning. If they are None, the model won't be fine-tuned and the
        parameters will be either default or taken as user-input.
    """

    def __init__(
            self,
            target: str,
            features: Union[Iterable[str], str] = None,
            model_params: dict = None,
    ) -> None:

        """
        Initialize Light Gradient Boost model.

        Args:
            target (str): Target variable.
            features (optional): Names of the explanatory variables (features) in form of a list of strings or a list.
                Default is None
            model_params (dict, optional): Dictionary of hyperparameters
        """

        super().__init__(target=target, features=features)  # Call the constructor of the EnsembleModel superclass

        if model_params:
            self.ensemble_params = model_params

        # Initialize the ensemble_params attribute if it's still None
        if self.ensemble_params is None:
            self.ensemble_params = {}

        # Create a LightGBM model with specified or default hyperparameters
        if self.ensemble_model is None:
            if self.hyper_param_space is None:
                self.ensemble_model = lgb.LGBMRegressor(**self.ensemble_params)
            else:
                self.ensemble_model = lgb.LGBMRegressor()

    def load_model(
            self,
            file_path: Union[Path, str],
    ) -> None:

        """
        Load a pre-trained LightGBM model from a file.

        Args:
            file_path (Path or str): The path to the model file.

        """

        logger.info(f"{'-' * 50}")

        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        self.ensemble_model = lgb.Booster(model_file=file_path)

        self.model_was_fit = True
        self.ensemble_params = self.ensemble_model.params
        self.features = self.ensemble_model.feature_name()

        logger.info(f"Light Gradient Booosting model successfully loaded from {str(file_path)}.")

    def save_model(
            self,
            filename: str,
            output_dir: Union[Path, str],
            **kwargs,
    ) -> None:

        """
        Save the current LightGBM model to a file.

        Args:
            filename (str): The name of the model file.
            output_dir (Path or str): The path where the model file will be saved.
            **kwargs: Keyword arguments to be passed into lightgbm.boosted_.save_model function.

        Raises:
            ValueError: If no model has been estimated before saving.

        """

        # Sanity check
        if self.ensemble_model is None:
            raise ValueError("No model has been estimated. Estimate (or load) a model to save it.")

        # Specify the extension and create a Path object
        file_extenstion = ".json"
        filename_model = filename + file_extenstion

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        file_path = (output_dir / filename_model)

        self.ensemble_model.booster_.save_model(file_path, **kwargs)

        logger.info(f"Light Gradient Boosting model successfully saved to {str(file_path)}.")

        # Export parameters used to estimate the model
        if self.ensemble_params is not None and self.ensemble_params != {}:
            if isinstance(self.ensemble_params, dict):
                params_export = pd.DataFrame.from_dict(self.ensemble_params, orient='index')
                params_export.columns = ['lgbm_params']
            else:
                params_export = self.ensemble_params.copy()

            logger.info(f"Exporting Light Gradient Boosting model parameters...")

            export_dataframe_to_csv(
                df=params_export,
                output_path=output_dir,
                filename=f"{filename}_params.csv",
                index=True,
            )
