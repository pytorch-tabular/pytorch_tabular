from unittest.mock import patch

import optuna
import pytest
from scipy.stats import uniform
from sklearn.metrics import r2_score

from pytorch_tabular import TabularModelTuner
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.models import CategoryEmbeddingModelConfig


@pytest.fixture(scope="module")
def tuner_setup(regression_data):
    train, test, target = regression_data
    continuous_cols = [
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    categorical_cols = ["HouseAgeBin"]
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=True,
        handle_unknown_categories=True,
    )
    trainer_config = TrainerConfig(
        max_epochs=1,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=True,
    )
    optimizer_config = OptimizerConfig()
    model_config = CategoryEmbeddingModelConfig(
        task="regression",
        layers="8-4",
    )
    return {
        "train": train,
        "test": test,
        "data_config": data_config,
        "trainer_config": trainer_config,
        "optimizer_config": optimizer_config,
        "model_config": model_config,
    }


def test_optuna_basic_minimize(tuner_setup):
    tuner = TabularModelTuner(
        data_config=tuner_setup["data_config"],
        model_config=tuner_setup["model_config"],
        optimizer_config=tuner_setup["optimizer_config"],
        trainer_config=tuner_setup["trainer_config"],
    )
    search_space = {
        "model_config__layers": ["8-4", "16-8"],
        "model_config.head_config__dropout": uniform(0, 0.5),
        "optimizer_config__optimizer": ["RAdam", "AdamW"],
    }
    result = tuner.tune(
        train=tuner_setup["train"],
        validation=tuner_setup["test"],
        search_space=search_space,
        strategy="optuna",
        n_trials=3,
        metric="loss",
        mode="min",
        progress_bar=False,
    )
    # Check backwards compatible 4-tuple unpacking
    trials_df, best_params, best_score, best_model = result
    assert len(trials_df) == 3
    assert result.study is not None
    assert tuner.study_ is result.study
    assert len(result.study.trials) == 3
    assert best_score in trials_df["loss"].values.tolist()
    assert best_model is not None


def test_optuna_basic_maximize_with_callable_metric(tuner_setup):
    tuner = TabularModelTuner(
        data_config=tuner_setup["data_config"],
        model_config=tuner_setup["model_config"],
        optimizer_config=tuner_setup["optimizer_config"],
        trainer_config=tuner_setup["trainer_config"],
    )
    search_space = {
        "model_config__layers": ["8-4", "16-8"],
        "model_config.head_config__dropout": (0.0, 0.3),
    }

    def custom_r2(y_true, y_pred):
        return r2_score(y_true, y_pred["MedHouseVal_prediction"].values)

    result = tuner.tune(
        train=tuner_setup["train"],
        validation=tuner_setup["test"],
        search_space=search_space,
        strategy="optuna",
        n_trials=2,
        metric=custom_r2,
        mode="max",
        progress_bar=False,
    )
    assert len(result.trials_df) == 2
    assert result.best_score in result.trials_df["custom_r2"].values.tolist()
    assert result.best_score == result.trials_df["custom_r2"].max()


def test_optuna_search_space_specifications(tuner_setup):
    tuner = TabularModelTuner(
        data_config=tuner_setup["data_config"],
        model_config=tuner_setup["model_config"],
        optimizer_config=tuner_setup["optimizer_config"],
        trainer_config=tuner_setup["trainer_config"],
    )
    search_space = {
        # Native optuna distributions
        "model_config.head_config__dropout": optuna.distributions.FloatDistribution(
            0.0, 0.4
        ),
        # Scipy distributions
        "optimizer_config__lr_scheduler_monitor_metric": ["val_loss"],
        # Dict specification
        "optimizer_config__optimizer": {
            "type": "categorical",
            "choices": ["RAdam", "AdamW"],
        },
        # Tuple specification (float with log)
        "model_config__learning_rate": ("float", 1e-4, 1e-1, True),
    }
    result = tuner.tune(
        train=tuner_setup["train"],
        validation=tuner_setup["test"],
        search_space=search_space,
        strategy="optuna",
        n_trials=2,
        metric="loss",
        mode="min",
        progress_bar=False,
    )
    assert len(result.trials_df) == 2
    for trial in result.study.trials:
        assert 0.0 <= trial.params["model_config.head_config__dropout"] <= 0.4
        assert trial.params["optimizer_config__optimizer"] in ["RAdam", "AdamW"]
        assert 1e-4 <= trial.params["model_config__learning_rate"] <= 1e-1


def test_optuna_custom_sampler_and_study(tuner_setup):
    custom_study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=123),
        pruner=optuna.pruners.MedianPruner(),
    )
    tuner = TabularModelTuner(
        data_config=tuner_setup["data_config"],
        model_config=tuner_setup["model_config"],
        optimizer_config=tuner_setup["optimizer_config"],
        trainer_config=tuner_setup["trainer_config"],
    )
    search_space = {
        "model_config__layers": ["8-4", "16-8"],
    }
    result = tuner.tune(
        train=tuner_setup["train"],
        validation=tuner_setup["test"],
        search_space=search_space,
        strategy="optuna",
        n_trials=2,
        metric="loss",
        mode="min",
        optuna_study=custom_study,
        progress_bar=False,
    )
    assert result.study is custom_study
    assert len(custom_study.trials) == 2


def test_optuna_cross_validation(tuner_setup):
    tuner = TabularModelTuner(
        data_config=tuner_setup["data_config"],
        model_config=tuner_setup["model_config"],
        optimizer_config=tuner_setup["optimizer_config"],
        trainer_config=tuner_setup["trainer_config"],
    )
    search_space = {
        "model_config__layers": ["8-4", "16-8"],
    }
    result = tuner.tune(
        train=tuner_setup["train"],
        search_space=search_space,
        strategy="optuna",
        n_trials=2,
        cv=3,
        metric="loss",
        mode="min",
        progress_bar=False,
    )
    assert len(result.trials_df) == 2
    assert result.best_score in result.trials_df["loss"].values.tolist()


def test_optuna_multi_model_tuning(tuner_setup):
    model_config_1 = CategoryEmbeddingModelConfig(task="regression", layers="8-4")
    model_config_2 = CategoryEmbeddingModelConfig(task="regression", layers="16-8")
    tuner = TabularModelTuner(
        data_config=tuner_setup["data_config"],
        model_config=[model_config_1, model_config_2],
        optimizer_config=tuner_setup["optimizer_config"],
        trainer_config=tuner_setup["trainer_config"],
    )
    search_space = [
        {"model_config.head_config__dropout": [0.1, 0.2]},
        {"model_config.head_config__dropout": [0.3, 0.4]},
    ]
    result = tuner.tune(
        train=tuner_setup["train"],
        validation=tuner_setup["test"],
        search_space=search_space,
        strategy="optuna",
        n_trials=2,
        metric="loss",
        mode="min",
        progress_bar=False,
    )
    # 2 trials per model config = 4 total
    assert len(result.trials_df) == 4
    assert len(tuner.studies_) == 2
    assert result.best_model is not None


def test_optuna_not_installed_error(tuner_setup):
    tuner = TabularModelTuner(
        data_config=tuner_setup["data_config"],
        model_config=tuner_setup["model_config"],
        optimizer_config=tuner_setup["optimizer_config"],
        trainer_config=tuner_setup["trainer_config"],
    )
    with patch("pytorch_tabular.tabular_model_tuner._optuna_available", False):
        with pytest.raises(ImportError, match="Optuna is not installed"):
            tuner.tune(
                train=tuner_setup["train"],
                validation=tuner_setup["test"],
                search_space={"model_config__layers": ["8-4"]},
                strategy="optuna",
                n_trials=1,
                metric="loss",
                mode="min",
            )
