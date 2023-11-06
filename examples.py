from models.hybrid_models.hybrid_ensemble import BoostedProphetModel, extract_prophet_features
from models.ml_models.ensemble_models import XGBoostModel
from models.vanila_models.prophet_model import ProphetModel

from sklearn.model_selection import train_test_split

import pandas as pd
import argparse
from loguru import logger

parser = argparse.ArgumentParser(description="Running hybrid boosting examples.")
parser.add_argument("--example_1", action="store_true", help="Run example no. 1.")
parser.add_argument("--example_2", action="store_true", help="Run example no. 2.")
parser.add_argument("--example_3", action="store_true", help="Run example no. 3.")
parser.add_argument("--example_4", action="store_true", help="Run example no. 4.")

if __name__ == "__main__":
    steps = parser.parse_args()

    if steps.example_1:
        #####################################################
        # Example 1 - estimating "empty" (hyperparams) models
        #####################################################

        logger.info("Running example No. 1")

        df = pd.read_csv(r"examples/data_example_eurpln.csv")
        df = df[['Date', 'Open', 'Adj Close']]
        df.columns = ['ds', 'open', 'price']
        df_train, df_test = train_test_split(df, train_size=0.8, shuffle=False)
        model = BoostedProphetModel(target="price", ensemble_model_type="xgb")

        model.fit_hybrid(df_train=df_train, features=['open'], country_holiday_prophet="PL")
        model.save_models(
            prophet_filename="prophet_03112023",
            ensemble_filename="xgb_03112023",
            output_path="examples/",
        )
        y_hat = model.predict_combined(df_predict=df_test[['ds', 'open']])
        y_hat.index = df_test.ds

    if steps.example_2:
        ############################
        # Example 2 - loading models
        ############################

        logger.info("Running example No. 2")

        df = pd.read_csv(r"examples/data_example_eurpln.csv")
        df = df[['Date', 'Open', 'Adj Close']]
        df.columns = ['ds', 'open', 'price']
        df_train, df_test = train_test_split(df, train_size=0.8, shuffle=False)
        model = BoostedProphetModel(target="price", ensemble_model_type="xgb")

        model.load_models(
            prophet_file_path=r"examples/prophet_03112023.json",
            ensemble_file_path=r"examples/xgb_03112023.json",
        )
        y_hat = model.predict_combined(df_predict=df_test[['ds', 'open']])
        y_hat.index = df_test.ds

    if steps.example_3:
        ###########################################################
        # Example 3 - estimating models with custom hyperparameters
        ###########################################################

        logger.info("Running example No. 3")

        df = pd.read_csv(r"examples/data_example_eurpln.csv")
        df = df[['Date', 'Open', 'Adj Close']]
        df.columns = ['ds', 'open', 'price']
        df_train, df_test = train_test_split(df, train_size=0.8, shuffle=False)
        model = BoostedProphetModel(target="price", ensemble_model_type="xgb")

        model.fit_hybrid(
            df_train=df_train,
            features=['open'],
            country_holiday_prophet="PL",
            prophet_params={
                "interval_width": 0.95,
                "changepoint_prior_scale": 0.3,
                "holidays_prior_scale": 7,
                "seasonality_prior_scale": 7,
                "seasonality_mode": "additive",
                "mcmc_samples": 10,
            },
            ensemble_params={
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": 30,
                "max_leaves": 0,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "gamma": 0.01,
                "eval_metric": "mphe",
            },
        )
        model.save_models(
            prophet_filename="prophet_03112023",
            ensemble_filename="xgb_03112023",
            output_path="examples/",
        )
        y_hat = model.predict_combined(df_predict=df_test[['ds', 'open']])
        y_hat.index = df_test.ds

    if steps.example_4:
        ##########################################################
        # Example 4 - estimating models with hyperparameter tuning
        ##########################################################

        logger.info("Running example No. 4")

        df = pd.read_csv(r"examples/data_example_eurpln.csv")
        df = df[['Date', 'Open', 'Adj Close']]
        df.columns = ['ds', 'open', 'price']
        df_train, df_test = train_test_split(df, train_size=0.8, shuffle=False)
        model = BoostedProphetModel(target="price", ensemble_model_type="xgb")
        model.prophet_model = ProphetModel(features="open")
        model.prophet_model.grid_search_prophet(
            df=df,
            target="price",
            hyper_param_space={
                "interval_width": [0.95],
                "changepoint_prior_scale": [0.3],
                "holidays_prior_scale": [7, 10],
                "seasonality_prior_scale": [7],
                "seasonality_mode": ["additive"],
                "mcmc_samples": [10],
            },
            horizon="90D",
            features=["open"],
            country_names="PL",
            metric_best="rmse",
        )

        df_train_with_prophet_features, features_with_prophet_features = extract_prophet_features(
            df=df,
            features="open",
            y_hat_prophet=model.prophet_model.y_hat,
        )

        model.ensemble_model = XGBoostModel(target="y", features=features_with_prophet_features)
        model.ensemble_model.grid_search_ensemble(
            df=df_train_with_prophet_features,
            hyper_param_space={
                "n_estimators": [50],
                "learning_rate": [0.05],
                "max_depth": [30, 13],
                "max_leaves": [0, 10],
                "subsample": [0.9, 0.8],
                "colsample_bytree": [0.9, 0.3],
                "gamma": [0.01],
                "eval_metric": ["mphe"],
            },
        )

        model.save_models(
            prophet_filename="prophet_03112023",
            ensemble_filename="xgb_03112023",
            output_path="examples/",
        )
        y_hat = model.predict_combined(df_predict=df_test[['ds', 'open']])
        y_hat.index = df_test.ds
