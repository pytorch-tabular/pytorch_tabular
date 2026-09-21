# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabular Model."""

import warnings
from collections import namedtuple
from collections.abc import Callable, Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf.dictconfig import DictConfig
from pandas import DataFrame
from rich.progress import Progress
from sklearn.model_selection import BaseCrossValidator, ParameterGrid, ParameterSampler

try:
    import optuna

    _optuna_available = True
except ImportError:
    _optuna_available = False

from pytorch_tabular.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.tabular_model import TabularModel
from pytorch_tabular.utils import (
    OOMException,
    OutOfMemoryHandler,
    get_logger,
    suppress_lightning_logs,
)

logger = get_logger(__name__)


class TunerOutput(
    namedtuple("OUTPUT", ["trials_df", "best_params", "best_score", "best_model"])
):
    """Output of TabularModelTuner."""

    def __new__(cls, trials_df, best_params, best_score, best_model, study=None):
        obj = super().__new__(cls, trials_df, best_params, best_score, best_model)
        obj.study = study
        return obj


OUTPUT = TunerOutput


class TabularModelTuner:
    """Tabular Model Tuner.

    This class is used to tune the hyperparameters of a TabularModel, given the search space,  strategy and metric to
    optimize.

    """

    ALLOWABLE_STRATEGIES = ["grid_search", "random_search", "optuna"]
    OUTPUT = TunerOutput

    def __init__(
        self,
        data_config: DataConfig | str | None = None,
        model_config: ModelConfig | str | None = None,
        optimizer_config: OptimizerConfig | str | None = None,
        trainer_config: TrainerConfig | list[TrainerConfig] | None = None,
        model_callable: Callable | None = None,
        model_state_dict_path: str | Path | None = None,
        suppress_lightning_logger: bool = True,
        **kwargs,
    ):
        """Tabular Model Tuner helps you tune the hyperparameters of a TabularModel.

        Args:
            data_config (Optional[Union[DataConfig, str]], optional): The DataConfig for the TabularModel.
                If str is passed, will initialize the DataConfig using the yaml file in that path.
                Defaults to None.

            model_config (Optional[Union[ModelConfig, List[TrainerConfig]], optional): The ModelConfig for the
                TabularModel. If str is passed, will initialize the ModelConfig using the yaml file in that path.
                Defaults to None.

            optimizer_config (Optional[Union[OptimizerConfig, str]], optional): The OptimizerConfig for the
                TabularModel. If str is passed, will initialize the OptimizerConfig using the yaml file in
                that path. Defaults to None.

            trainer_config (Optional[Union[TrainerConfig, str]], optional): The TrainerConfig for the TabularModel.
                If str is passed, will initialize the TrainerConfig using the yaml file in that path.
                Defaults to None.

            model_callable (Optional[Callable], optional): A callable that returns a PyTorch Tabular Model.
                If provided, will ignore the model_config and use this callable to initialize the model.
                Defaults to None.

            model_state_dict_path (Optional[Union[str, Path]], optional): Path to the state dict of the model.

                If provided, will ignore the model_config and use this state dict to initialize the model.
                Defaults to None.

            suppress_lightning_logger (bool, optional): Whether to suppress the lightning logger. Defaults to True.

            **kwargs: Additional keyword arguments to be passed to the TabularModel init.

        """
        if not isinstance(model_config, list):
            model_config = [model_config]

        if trainer_config.profiler is not None:
            warnings.warn(
                "Profiler is not supported in tuner. Set profiler=None in TrainerConfig to disable this warning."
            )
            trainer_config.profiler = None
        if trainer_config.fast_dev_run:
            warnings.warn(
                "fast_dev_run is turned on. Tuning results won't be accurate."
            )
        if trainer_config.progress_bar != "none":
            # If config and tuner have progress bar enabled, it will result in a bug within the library (rich.progress)
            trainer_config.progress_bar = "none"
            warnings.warn(
                "Turning off progress bar. Set progress_bar='none' in TrainerConfig to disable this warning."
            )
        trainer_config.trainer_kwargs.update({"enable_model_summary": False})
        self.data_config = data_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.trainer_config = trainer_config
        self.suppress_lightning_logger = suppress_lightning_logger
        self.study_ = None
        self.studies_ = []
        self.tabular_model_init_kwargs = {
            "model_callable": model_callable,
            "model_state_dict_path": model_state_dict_path,
            **kwargs,
        }

    def _check_assign_config(self, config, param, value):
        if isinstance(config, DictConfig):
            if param in config:
                config[param] = value
            else:
                raise ValueError(f"{param} is not a valid parameter for {str(config)}")
        elif isinstance(config, ModelConfig | OptimizerConfig):
            if hasattr(config, param):
                setattr(config, param, value)
            else:
                raise ValueError(f"{param} is not a valid parameter for {str(config)}")

    def _update_configs(
        self,
        optimizer_config: OptimizerConfig,
        model_config: ModelConfig,
        params: dict,
    ):
        """Update the configs with the new parameters."""
        # update configs with the new parameters
        for k, v in params.items():
            if k == "model":
                continue

            root, param = k.split("__")
            if root.startswith("trainer_config"):
                raise ValueError(
                    "The trainer_config is not supported by tuner. Please remove it from tuner parameters!"
                )
            elif root.startswith("optimizer_config"):
                self._check_assign_config(optimizer_config, param, v)
            elif root.startswith("model_config.head_config"):
                param = param.replace("model_config.head_config.", "")
                self._check_assign_config(model_config.head_config, param, v)
            elif root.startswith("model_config") and "head_config" not in root:
                self._check_assign_config(model_config, param, v)
            else:
                raise ValueError(
                    f"{k} is not in the proper format. Use __ to separate the "
                    "root and param. for eg. `optimizer_config__optimizer` should be "
                    "used to update the optimizer parameter in the optimizer_config"
                )
        return optimizer_config, model_config

    def _sample_optuna_param(self, trial, name: str, spec: Any) -> Any:
        """Sample a hyperparameter using Optuna Trial suggest methods based on spec."""
        if _optuna_available and isinstance(
            spec, optuna.distributions.BaseDistribution
        ):
            return trial._suggest(name, spec)
        if hasattr(spec, "dist"):
            dist_name = getattr(spec.dist, "name", None)
            if dist_name == "uniform":
                loc, scale = spec.args[0], spec.args[1]
                return trial.suggest_float(name, loc, loc + scale)
            elif dist_name in ("loguniform", "reciprocal"):
                a, b = spec.args[0], spec.args[1]
                return trial.suggest_float(name, a, b, log=True)
            elif dist_name == "randint":
                low, high = spec.args[0], spec.args[1]
                return trial.suggest_int(name, low, high - 1)
            elif hasattr(spec, "rvs"):
                return spec.rvs()
        if isinstance(spec, dict):
            t = spec.get("type", "float")
            if t == "float":
                return trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                    step=spec.get("step", None),
                )
            elif t == "int":
                return trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                    step=spec.get("step", 1),
                    log=spec.get("log", False),
                )
            elif t == "categorical":
                return trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(
                    f"Unsupported distribution dict type: {t} for parameter {name}"
                )
        if isinstance(spec, tuple):
            if len(spec) >= 3 and spec[0] in ("float", "int", "categorical"):
                t = spec[0]
                if t == "float":
                    log = spec[3] if len(spec) > 3 else False
                    return trial.suggest_float(name, spec[1], spec[2], log=log)
                elif t == "int":
                    step = spec[3] if len(spec) > 3 else 1
                    return trial.suggest_int(name, spec[1], spec[2], step=step)
                elif t == "categorical":
                    return trial.suggest_categorical(name, spec[1])
            elif (
                len(spec) == 2
                and isinstance(spec[0], int | float)
                and isinstance(spec[1], int | float)
            ):
                if isinstance(spec[0], int) and isinstance(spec[1], int):
                    return trial.suggest_int(name, spec[0], spec[1])
                return trial.suggest_float(name, spec[0], spec[1])
            elif (
                len(spec) == 3
                and isinstance(spec[0], int | float)
                and isinstance(spec[1], int | float)
                and (spec[2] in (True, False, "log"))
            ):
                log = spec[2] == "log" or spec[2] is True
                return trial.suggest_float(name, spec[0], spec[1], log=log)
            else:
                return trial.suggest_categorical(name, list(spec))
        if isinstance(spec, list):
            return trial.suggest_categorical(name, spec)
        if callable(spec):
            return spec(trial)
        return spec

    def tune(
        self,
        train: DataFrame,
        search_space: dict | list[dict],
        metric: str | Callable,
        mode: str,
        strategy: str,
        validation: DataFrame | None = None,
        n_trials: int | None = None,
        cv: int | Iterable | BaseCrossValidator | None = None,
        cv_agg_func: Callable | None = np.mean,
        cv_kwargs: dict | None = None,
        return_best_model: bool = True,
        verbose: bool = False,
        progress_bar: bool = True,
        random_state: int | None = 42,
        ignore_oom: bool = True,
        timeout: int | None = None,
        optuna_sampler: Any | None = None,
        optuna_pruner: Any | None = None,
        optuna_study: Any | None = None,
        **kwargs,
    ):
        """Tune the hyperparameters of the TabularModel.

        Args:
            train (DataFrame): Training data

            validation (DataFrame, optional): Validation data. Defaults to None.

            search_space (Dict): A dictionary of the form {param_name: [values to try]}
                for grid search or {param_name: distribution} for random search / optuna.

            metric (Union[str, Callable]): The metric to be used for evaluation.
                If str is provided, will use that metric from the defined ones.
                If callable is provided, will use that function as the metric.
                We expect callable to be of the form `metric(y_true, y_pred)`. For classification
                problems, The `y_pred` is a dataframe with the probabilities for each class
                (<class>_probability) and a final prediction(prediction). And for Regression, it is a
                dataframe with a final prediction (<target>_prediction).
                Defaults to None.

            mode (str): One of ['max', 'min']. Whether to maximize or minimize the metric.

            strategy (str): One of ['grid_search', 'random_search', 'optuna']. The strategy to use for tuning.

            n_trials (int, optional): Number of trials to run. Required for random search and optuna
                (unless timeout is specified for optuna). Defaults to None.

            cv (Optional[Union[int, Iterable, BaseCrossValidator]]): Determines the cross-validation splitting strategy.
                Possible inputs for cv are:

                - None, to not use any cross validation. We will just use the validation data
                - integer, to specify the number of folds in a (Stratified)KFold,
                - An iterable yielding (train, test) splits as arrays of indices.
                - A scikit-learn CV splitter.
                Defaults to None.

            cv_agg_func (Optional[Callable], optional): Function to aggregate the cross validation scores.
                Defaults to np.mean.

            cv_kwargs (Optional[Dict], optional): Additional keyword arguments to be passed to the cross validation
                method. Defaults to None.

            return_best_model (bool, optional): If True, will return the best model. Defaults to True.

            verbose (bool, optional): Whether to print the results of each trial. Defaults to False.

            progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.

            random_state (Optional[int], optional): Random state to be used for random search and optuna sampler.
                Defaults to 42.

            ignore_oom (bool, optional): Whether to ignore out of memory errors. Defaults to True.

            timeout (Optional[int], optional): Maximum time in seconds to run trials for optuna strategy.
                Defaults to None.

            optuna_sampler (Optional[Any], optional): Custom Optuna sampler instance (e.g. TPESampler,
                RandomSampler). If None and strategy is 'optuna', defaults to TPESampler(seed=random_state).
                Defaults to None.

            optuna_pruner (Optional[Any], optional): Custom Optuna pruner instance (e.g. MedianPruner).
                Defaults to None.

            optuna_study (Optional[Any], optional): Pre-existing Optuna study to optimize into.
                Defaults to None.

            **kwargs: Additional keyword arguments to be passed to the TabularModel fit.

        Returns:
            OUTPUT: A named tuple with the following attributes:
                trials_df (DataFrame): A dataframe with the results of each trial
                best_params (Dict): The best parameters found
                best_score (float): The best score found
                best_model (TabularModel or None): If return_best_model is True, return best_model otherwise return None
                study (optuna.Study or None): The Optuna study object if strategy='optuna', otherwise None. Accessible
                    via `result.study`.

        """
        assert (
            strategy in self.ALLOWABLE_STRATEGIES
        ), f"tuner must be one of {self.ALLOWABLE_STRATEGIES}"
        assert mode in ["max", "min"], "mode must be one of ['max', 'min']"
        assert metric is not None, "metric must be specified"
        assert (
            isinstance(search_space, dict) or (isinstance(search_space, list))
        ) and len(search_space) > 0, "search_space must be a non-empty dict"
        if strategy == "optuna":
            if not _optuna_available:
                raise ImportError(
                    "Optuna is not installed. Please install optuna to use the optuna strategy: "
                    "`pip install optuna` or `pip install 'pytorch_tabular[extra]'`"
                )
            assert (
                n_trials is not None or timeout is not None
            ), "n_trials or timeout must be specified for optuna"
        if self.suppress_lightning_logger:
            suppress_lightning_logs()
        if cv is not None and validation is not None:
            warnings.warn(
                "Both validation and cv are provided. Ignoring validation and using cv. Use "
                "`validation=None` to turn off this warning."
            )
            validation = None

        if not isinstance(search_space, list):
            search_space = [search_space]

        assert len(self.model_config) == len(
            search_space
        ), "model_config and search_space must have the same length"

        verbose_tabular_model = self.tabular_model_init_kwargs.pop("verbose", False)
        if cv_kwargs is None:
            cv_kwargs = {}

        if strategy == "optuna":
            if not verbose:
                optuna.logging.set_verbosity(optuna.logging.WARNING)
            else:
                optuna.logging.set_verbosity(optuna.logging.INFO)

        with Progress() as progress:
            model_config_iterator = range(len(self.model_config))
            if progress_bar:
                model_config_iterator = progress.track(
                    model_config_iterator, description="[green]Running models config..."
                )

            datamodule = None
            trials = []
            best_model = None
            best_score = 0.0
            studies = []

            if isinstance(metric, str):
                is_callable_metric = False
                metric_str = metric
            elif callable(metric):
                is_callable_metric = True
                metric_str = metric.__name__

            def _evaluate_trial(params: dict, trial_id: int, idx: int):
                nonlocal datamodule, best_model, best_score, validation

                trainer_config_t = deepcopy(self.trainer_config)
                optimizer_config_t = deepcopy(self.optimizer_config)
                model_config_t = deepcopy(self.model_config[idx])

                optimizer_config_t, model_config_t = self._update_configs(
                    optimizer_config_t, model_config_t, params
                )
                tabular_model_t = TabularModel(
                    data_config=self.data_config,
                    model_config=model_config_t,
                    optimizer_config=optimizer_config_t,
                    trainer_config=trainer_config_t,
                    verbose=verbose_tabular_model,
                    **self.tabular_model_init_kwargs,
                )

                # Create datamodule
                if not datamodule:
                    prep_dl_kwargs, prep_model_kwargs, train_kwargs = (
                        tabular_model_t._split_kwargs(kwargs)
                    )
                    if "seed" not in prep_dl_kwargs:
                        prep_dl_kwargs["seed"] = random_state
                    datamodule = tabular_model_t.prepare_dataloader(
                        train=train, validation=validation, **prep_dl_kwargs
                    )
                    validation = (
                        validation
                        if validation is not None
                        else datamodule.validation_dataset.data
                    )
                else:
                    prep_dl_kwargs, prep_model_kwargs, train_kwargs = (
                        tabular_model_t._split_kwargs(kwargs)
                    )

                if cv is not None:
                    cv_kwargs_t = deepcopy(cv_kwargs)
                    cv_verbose = cv_kwargs_t.pop("verbose", False)
                    cv_kwargs_t.pop("handle_oom", None)
                    with OutOfMemoryHandler(handle_oom=True) as handler:
                        cv_scores, _ = tabular_model_t.cross_validate(
                            cv=cv,
                            train=train,
                            metric=metric,
                            verbose=cv_verbose,
                            handle_oom=False,
                            **cv_kwargs_t,
                        )
                    if handler.oom_triggered:
                        if not ignore_oom:
                            raise OOMException(
                                "Out of memory error occurred during cross validation. "
                                "Set ignore_oom=True to ignore this error."
                            )
                        else:
                            score = float(np.inf if mode == "min" else -np.inf)
                            params.update({metric_str: score})
                            params.update({"model": f"{params['model']} (OOM)"})
                    else:
                        score = float(cv_agg_func(cv_scores))
                        params.update({metric_str: score})
                else:
                    model = tabular_model_t.prepare_model(
                        datamodule=datamodule,
                        **prep_model_kwargs,
                    )
                    train_kwargs_t = deepcopy(train_kwargs)
                    train_kwargs_t.pop("handle_oom", None)
                    with OutOfMemoryHandler(handle_oom=True) as handler:
                        tabular_model_t.train(
                            model=model,
                            datamodule=datamodule,
                            handle_oom=False,
                            **train_kwargs_t,
                        )
                    if handler.oom_triggered:
                        if not ignore_oom:
                            raise OOMException(
                                "Out of memory error occurred during training. "
                                "Set ignore_oom=True to ignore this error."
                            )
                        else:
                            score = float(np.inf if mode == "min" else -np.inf)
                            params.update({metric_str: score})
                            params.update({"model": f"{params['model']} (OOM)"})
                    else:
                        if is_callable_metric:
                            preds = tabular_model_t.predict(
                                validation, include_input_features=False
                            )
                            score = float(
                                metric(validation[tabular_model_t.config.target], preds)
                            )
                            params.update({metric_str: score})
                        else:
                            result = tabular_model_t.evaluate(validation, verbose=False)
                            params.update(
                                {
                                    k.replace("test_", ""): v
                                    for k, v in result[0].items()
                                }
                            )
                            score = float(params[metric_str])

                        if return_best_model:
                            tabular_model_t.datamodule = None
                            if best_model is None:
                                best_model = deepcopy(tabular_model_t)
                                best_score = score
                            else:
                                if mode == "min":
                                    if score < best_score:
                                        best_model = deepcopy(tabular_model_t)
                                        best_score = score
                                elif mode == "max":
                                    if score > best_score:
                                        best_model = deepcopy(tabular_model_t)
                                        best_score = score

                params.update({"trial_id": trial_id})
                trials.append(params)
                if verbose:
                    logger.info(
                        f"Trial {trial_id + 1}: {params} | Score: {params[metric_str]}"
                    )
                return score

            for idx in model_config_iterator:
                search_space_temp = {
                    **{"model": [f"{idx}-{self.model_config[idx].__class__.__name__}"]},
                    **search_space[idx],
                }

                if strategy == "grid_search":
                    assert all(
                        isinstance(v, list) for v in search_space_temp.values()
                    ), "For grid search, all values in search_space must be a list of values to try"
                    search_space_iterator = list(ParameterGrid(search_space_temp))
                    if n_trials is not None:
                        warnings.warn(
                            "n_trials is ignored for grid search to do a complete sweep of"
                            " the grid. Set n_trials=None to turn off this warning."
                        )
                    n_trials_curr = sum(1 for _ in search_space_iterator)
                elif strategy == "random_search":
                    assert (
                        n_trials is not None
                    ), "n_trials must be specified for random search"
                    search_space_iterator = list(
                        ParameterSampler(
                            search_space_temp,
                            n_iter=n_trials,
                            random_state=random_state,
                        )
                    )
                    n_trials_curr = n_trials
                elif strategy == "optuna":
                    search_space_iterator = None
                    n_trials_curr = n_trials
                else:
                    raise NotImplementedError(f"{strategy} is not implemented yet.")

                if strategy in ["grid_search", "random_search"]:
                    # Sort by trainer_config to recreate the datamodule when necessary
                    trainer_configs = [
                        key for key in search_space_iterator if "trainer_config" in key
                    ]
                    for key in trainer_configs:
                        search_space_iterator = sorted(
                            search_space_iterator, key=lambda s: s[key]
                        )

                    if progress_bar:
                        search_space_iterator = progress.track(
                            search_space_iterator,
                            description=f"[blue]Training {idx}-{self.model_config[idx].__class__.__name__}...",
                        )

                    for i, params in enumerate(search_space_iterator):
                        _evaluate_trial(params, i, idx)
                elif strategy == "optuna":
                    direction = "maximize" if mode == "max" else "minimize"
                    if optuna_study is not None:
                        study = optuna_study
                    else:
                        if optuna_sampler is not None:
                            sampler = optuna_sampler
                        elif random_state is not None:
                            sampler = optuna.samplers.TPESampler(seed=random_state)
                        else:
                            sampler = optuna.samplers.TPESampler()
                        pruner = optuna_pruner
                        study = optuna.create_study(
                            direction=direction,
                            sampler=sampler,
                            pruner=pruner,
                        )
                    studies.append(study)

                    callbacks = []
                    if progress_bar:
                        optuna_task = progress.add_task(
                            f"[blue]Training {idx}-{self.model_config[idx].__class__.__name__}...",
                            total=n_trials_curr,
                        )

                        def _progress_cb(s, t):
                            progress.update(optuna_task, advance=1)

                        callbacks.append(_progress_cb)

                    def _objective(trial):
                        params = {
                            "model": f"{idx}-{self.model_config[idx].__class__.__name__}",
                        }
                        for k, v in search_space[idx].items():
                            params[k] = self._sample_optuna_param(trial, k, v)
                        score = _evaluate_trial(params, trial.number, idx)
                        return score

                    study.optimize(
                        _objective,
                        n_trials=n_trials_curr,
                        timeout=timeout,
                        callbacks=callbacks if callbacks else None,
                    )

        trials_df = pd.DataFrame(trials)
        trials_col = trials_df.pop("trial_id")
        if mode == "max":
            best_idx = trials_df[metric_str].idxmax()
        elif mode == "min":
            best_idx = trials_df[metric_str].idxmin()
        else:
            raise NotImplementedError(f"{mode} is not implemented yet.")
        best_params = trials_df.iloc[best_idx].to_dict()
        best_score = best_params.pop(metric_str)
        trials_df.insert(0, "trial_id", trials_col)

        if verbose:
            logger.info("Model Tuner Finished")
            logger.info(
                f"Best Model: {best_params['model']} - Best Score ({metric_str}): {best_score}"
            )

        if strategy == "optuna":
            self.studies_ = studies
            self.study_ = studies[0] if len(studies) == 1 else studies
        else:
            self.studies_ = []
            self.study_ = None

        if return_best_model and best_model is not None:
            best_model.datamodule = datamodule
            return self.OUTPUT(
                trials_df, best_params, best_score, best_model, study=self.study_
            )
        else:
            return self.OUTPUT(
                trials_df, best_params, best_score, None, study=self.study_
            )
