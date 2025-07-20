import errno
import logging
import logging.config
import os
from abc import abstractmethod, ABC
from argparse import Namespace
from dataclasses import asdict, is_dataclass
from datetime import datetime
from time import time
from pathlib import Path
from typing import Optional, Union, Dict, Any, Mapping, MutableMapping, Literal, List, Callable

import re
import mlflow
import pandas as pd
from mlflow.entities import Param
from mlflow.entities.metric import Metric
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

from torch import Tensor
from torch.nn import Module

from core.utils.general import root_path
from core.configs.values import TrainingState


def setup_logging(timestamp=None, base_log_dir=None):
    """
    Setup logging configuration with timestamp-based directory structure.

    @param timestamp: Custom timestamp string. If None, uses current time.
    @param base_log_dir: Base directory for logs. If None, uses 'logs' in project root.
    @return: The log directory path that was created
    """
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set base log directory
    if base_log_dir is None:
        try:
            base_log_dir = Path(root_path())
        except ImportError:
            base_log_dir = Path.cwd().parents[0]
            print(f"Could not find project root. Using current working directory as base log directory: {base_log_dir}")
    else:
        base_log_dir = Path(base_log_dir)

    # Create timestamped log directory
    log_dir = Path(base_log_dir, "logs", timestamp)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get environment variable to determine the working environment
    env = os.getenv('ENVIRONMENT', 'development')

    # Set log levels based on environment
    if env == 'production':
        log_level = 'INFO'
        console_level = 'WARNING'
    else:
        log_level = 'DEBUG'
        console_level = 'INFO'

    # Define log file paths
    app_log_file = Path(log_dir, "application.log")
    error_log_file = Path(log_dir, "error.log")

    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '[%(asctime)s][%(name)s line %(lineno)s][%(levelname)s] - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(asctime)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': console_level,
                'formatter': 'simple'
            },
            'file_all': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': str(app_log_file),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf-8'
            },
            'file_error': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': str(error_log_file),
                'maxBytes': 10485760,
                'backupCount': 3,
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            # Root logger
            '': {
                'handlers': ['console', 'file_all', 'file_error'],
                'level': log_level,
                'propagate': False
            }
        }
    }

    logging.config.dictConfig(LOGGING_CONFIG)

    # Create a symlink to the latest log directory for easy access
    latest_link = Path(base_log_dir, "logs", "latest")
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(timestamp, target_is_directory=True)

    print(f"Logger initialized for environment: {env}")
    print(f"Log directory: {log_dir}")
    print(f"Latest logs symlink: {latest_link}")

    return str(log_dir)


def get_logger(name=None):
    """
    Get a logger with the specified name.

    Args:
        name (str, optional): Logger name. If None, uses the caller's __name__

    Returns:
        logging.Logger: Configured logger
    """
    if name is None:
        # Get the caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')

    return logging.getLogger(name)


# Usage example for deep learning framework
def setup_ml_logging_and_mlflow(experiment_name, run_name=None, tracking_uri=None, disable_mlflow=False):
    """
    Setup both logging and MLflow for a deep learning experiment.

    @param experiment_name: Name of the MLflow experiment
    @param run_name: Name of the MLflow run (optional)
    @param tracking_uri: URI for the MLflow tracking server (optional, defaults to environment variable)
    @param disable_mlflow: If True, disables MLflow logging. Useful for local testing/developing without MLflow.
    @return: Tuple containing the log directory path and the MLFlowLogger instance

    Usage Example:
    >>> log_dir, mlflow_logger = setup_ml_logging_and_mlflow(experiment_name="my_trial", run_name="demo_v1")
    >>> runtime_logger = get_logger(__name__) # or logging.getLogger(__name__)

    mlflow_logger is used to log model, hyperparameters, and metrics to MLflow.
    >>> mlflow_logger.log_hyperparams({"learning_rate": 0.001, "batch_size": 32})
    >>> mlflow_logger.log_metrics({"train_loss": 0.5, "val_accuracy": 0.8}, step=1)
    >>> mlflow_logger.log_model(model, model_name="my_model")

    runtime_logger is used for general logging during the experiment.
    >>> runtime_logger.info("Experiment started")
    >>> runtime_logger.error("An error occurred during training")

    """
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = setup_logging(timestamp=timestamp)

    if disable_mlflow:
        print("MLflow logging is disabled. Returning log directory only.")
        mlflow_logger = None
    else:
        # Setup MLflow logger
        mlflow_logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=run_name or f"run_{timestamp}",
            tracking_uri=tracking_uri,
            tags={
                "timestamp": timestamp,
                "log_dir": log_dir
            }
        )
        print(f"MLFlow directory: {mlflow_logger.save_dir}")
        # Log the logging directory to MLflow
        mlflow_logger.log_hyperparams({"logging_dir": log_dir})

    return log_dir, mlflow_logger


class Logger(ABC):
    """

    Base class for experiment loggers.

    """

    @property
    @abstractmethod
    def name(self) -> Optional[str]:
        """Return the experiment name."""

    @property
    @abstractmethod
    def version(self) -> Optional[Union[int, str]]:
        """Return the experiment version."""

    @property
    def root_dir(self) -> Optional[str]:
        """Return the root directory where all versions of an experiment get saved, or `None` if the logger does not
        save data locally."""
        return None

    @property
    def log_dir(self) -> Optional[str]:
        """Return directory the current version of the experiment gets saved, or `None` if the logger does not save
        data locally."""
        return None

    @property
    def group_separator(self) -> str:
        """Return the default separator used by the logger to group the data into subfolders."""
        return "/"

    @abstractmethod
    def log_metrics(
            self, metrics: Dict[str, float], step: Optional[int] = None, **kwargs: Any
    ) -> None:
        """Records metrics. This method logs metrics as soon as it received them.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded

        """
        pass

    @abstractmethod
    def log_hyperparams(
            self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any
    ) -> None:
        """Record hyperparameters.

        Args:
            params: :class:`~argparse.Namespace` or `Dict` containing the hyperparameters
            args: Optional positional arguments, depends on the specific logger being used
            kwargs: Optional keyword arguments, depends on the specific logger being used

        """

    def log_graph(self, model: Module, input_array: Optional[Tensor] = None) -> None:
        """Record model graph.

        Args:
            model: the model with an implementation of ``forward``.
            input_array: input passes to `model.forward`

        """
        pass

    def save(self) -> None:
        """Save log data."""

    def finalize(self, status: str) -> None:
        """Do any processing that is necessary to finalize an experiment.

        Args:
            status: Status that the experiment finished with (e.g. success, failed, aborted)

        """
        self.save()

    def log_model(self, model, model_name: str):
        pass


class MLFlowLogger(Logger):
    """
    Logger for MLFlow. More details can be found in the [official documentation](https://mlflow.org/docs/latest/python_api/mlflow.html).
    and [mlflow logger](https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/loggers/mlflow.py)
    """

    LOGGER_JOIN_CHAR = "_"

    @property
    def name(self) -> Optional[str]:
        return self._experiment_name

    @property
    def version(self) -> Optional[Union[int, str]]:
        return self._run_id

    def __init__(
            self,
            experiment_name: str = "lightning_logs",
            run_name: Optional[str] = None,
            tracking_uri: Optional[str] = os.getenv("MLFLOW_TRACKING_URI"),
            tags: Optional[Dict[str, Any]] = None,
            save_dir: Optional[str] = "./mlruns",
            log_model: Literal[True, False, "all"] = False,
            prefix: str = "",
            artifact_location: Optional[str] = None,
            run_id: Optional[str] = None,
            system_metrics_logging: bool = False,
    ):
        """

        :param experiment_name:
        :param run_name:
        :param tracking_uri:
        :param tags:
        :param save_dir:
        :param log_model:
            * if ``log_model == 'all'``, checkpoints are logged during training.
            * if ``log_model == True``, checkpoints are logged at the end of training, except when
              :paramref:`~lightning.pytorch.callbacks.Checkpoint.save_top_k` ``== -1``
              which also logs every checkpoint during training.
            * if ``log_model == False`` (default), no checkpoint is logged.
        :param prefix:
        :param artifact_location:
        :param run_id:
        """
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tracking_uri = tracking_uri
        self._tags = tags
        self._save_dir = save_dir
        self._log_model = log_model
        self._prefix = prefix
        self._artifact_location = artifact_location
        self._run_id = run_id
        self._mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
        self._initialized = False
        self._experiment_id: Optional[str] = None
        if system_metrics_logging:
            mlflow.enable_system_metrics_logging()

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]):
        params = _convert_params(params)
        params = _flatten_dict(params)

        # Truncate parameter values to 250 characters.
        params_list = [Param(key=k, value=str(v)[:500]) for k, v in params.items()]

        # Log in chunks of 100 parameters (the maximum allowed by MLflow).
        for idx in range(0, len(params_list), 100):
            self.experiment.log_batch(
                run_id=self.run_id, params=params_list[idx: idx + 100]
            )

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None, prefix: Optional[str] = None):
        # Handle pandas DataFrame input
        if isinstance(metrics, pd.DataFrame):
            if metrics.shape[0] != 1:
                raise ValueError("MLflow only supports logging single-row DataFrame as metrics.")
            new_metrics = {}
            for col in metrics.columns:
                key = col.replace("@", "_at_")
                new_metrics[key] = float(metrics[col].values[0])
            metrics = new_metrics

        valid_prefix = prefix if prefix is not None else self._prefix
        metrics = _add_prefix(metrics, valid_prefix, self.LOGGER_JOIN_CHAR)
        metrics_list: List[Metric] = []

        timestamp_ms = int(time() * 1000)
        for k, v in metrics.items():
            if isinstance(v, str):
                print(f"Discarding metric with string value {k}={v}.")
                continue

            new_k = re.sub("[^a-zA-Z0-9_/. -]+", "", k)
            if k != new_k:
                print(
                    "MLFlow only allows '_', '/', '.' and ' ' special characters in metric name."
                    f" Replacing {k} with {new_k}.",
                )
                k = new_k
            metrics_list.append(
                Metric(key=k, value=v, timestamp=timestamp_ms, step=step or 0)
            )

        self.experiment.log_batch(run_id=self.run_id, metrics=metrics_list)

    def log_model(self, model, model_name: str):
        mlflow.pytorch.log_model(model, model_name)

    def log_model_symlink(self, model_dir):
        from core.utils.general import check_and_mkdir, symlink_force
        if model_dir is None or not os.path.exists(model_dir):
            return
        mlflow_best_model_dir = os.path.join(self.artifacts_dir, "model", "best")
        check_and_mkdir(mlflow_best_model_dir)
        symlink_force(model_dir, mlflow_best_model_dir)

    @property
    def experiment(self) -> "MlflowClient":
        r"""Actual MLflow object. To use MLflow features in your :class:`~lightning.pytorch.core.LightningModule` do the
        following.

        Example::

            self.logger.experiment.some_mlflow_function()

        """
        if self._initialized:
            return self._mlflow_client

        mlflow.set_tracking_uri(self._tracking_uri)

        if self._run_id is not None and len(self._run_id) > 0:
            run = self._mlflow_client.get_run(self._run_id)
            self._experiment_id = run.info.experiment_id
            self._initialized = True
            return self._mlflow_client

        if self._experiment_id is None:
            expt = self._mlflow_client.get_experiment_by_name(self._experiment_name)
            if expt is not None:
                self._experiment_id = expt.experiment_id
            else:
                print(
                    f"Experiment with name {self._experiment_name} not found. Creating it."
                )
                self._experiment_id = self._mlflow_client.create_experiment(
                    name=self._experiment_name,
                    artifact_location=self._artifact_location,
                )

        if self._run_id is None or len(self._run_id) == 0:
            if self._run_name is not None:
                self._tags = self._tags or {}

                if MLFLOW_RUN_NAME in self._tags:
                    print(
                        f"The tag {MLFLOW_RUN_NAME} is found in tags. The value will be overridden by {self._run_name}."
                    )
                self._tags[MLFLOW_RUN_NAME] = self._run_name

            resolve_tags = _get_resolve_tags()
            run = self._mlflow_client.create_run(
                experiment_id=self._experiment_id, tags=resolve_tags(self.tags), run_name=self._run_name
            )
            self._run_id = run.info.run_id

        self._initialized = True
        return self._mlflow_client

    @property
    def run_id(self) -> Optional[str]:
        """Create the experiment if it does not exist to get the run id.

        Returns:
            The run id.

        """
        _ = self.experiment
        return self._run_id

    @property
    def experiment_id(self) -> Optional[str]:
        """Create the experiment if it does not exist to get the experiment id.

        Returns:
            The experiment id.

        """
        _ = self.experiment
        return self._experiment_id

    def finalize(self, status: int = TrainingState.COMPLETED) -> None:
        if not self._initialized:
            return
        if status == TrainingState.COMPLETED:
            status = "FINISHED"
        elif status == TrainingState.FAILED:
            status = "FAILED"

        if self.experiment.get_run(self._run_id):
            self.experiment.set_terminated(self._run_id, status)

    @property
    def save_dir(self) -> Optional[str]:
        """The root file directory in which MLflow experiments are saved.

        Return:
            Local path to the root experiment directory if the tracking uri is local.
            Otherwise returns `None`.

        """
        return self.artifacts_dir

    @property
    def tags(self):
        return self._tags or {}

    @property
    def artifacts_dir(self) -> str:
        if not self._initialized:
            _ = self.experiment

        run = self.experiment.get_run(self.run_id)
        artifact_uri = run.info.artifact_uri

        # remove "file://" for local system
        if artifact_uri.startswith('file://'):
            return artifact_uri[7:]

        return artifact_uri


def _get_resolve_tags() -> Callable:
    from mlflow.tracking import context

    # before v1.1.0
    if hasattr(context, "resolve_tags"):
        from mlflow.tracking.context import resolve_tags
    # since v1.1.0
    elif hasattr(context, "registry"):
        from mlflow.tracking.context.registry import resolve_tags
    else:
        resolve_tags = lambda tags: tags

    return resolve_tags


def _flatten_dict(
        params: MutableMapping[Any, Any], delimiter: str = "/", parent_key: str = ""
) -> Dict[str, Any]:
    """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.

    Returns:
        Flattened dict.

    Examples:
        >>> _flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> _flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> _flatten_dict({5: {'a': 123}})
        {'5/a': 123}

    """
    result: Dict[str, Any] = {}
    for k, v in params.items():
        new_key = parent_key + delimiter + str(k) if parent_key else str(k)
        if is_dataclass(v):
            v = asdict(v)
        elif isinstance(v, Namespace):
            v = vars(v)

        if isinstance(v, MutableMapping):
            result = {
                **result,
                **_flatten_dict(v, parent_key=new_key, delimiter=delimiter),
            }
        else:
            result[new_key] = v
    return result


def _convert_params(
        params: Optional[Union[Dict[str, Any], Namespace]]
) -> Dict[str, Any]:
    """Ensure parameters are a dict or convert to dict if necessary.

    @param params: Parameters to be converted to a dictionary. Can be a `Namespace` or a `Dict`.
    @return: Dictionary containing the parameters.

    """
    # in case converting from namespace
    if isinstance(params, Namespace):
        params = vars(params)

    if params is None:
        params = {}

    return params


def _add_prefix(
        metrics: Mapping[str, Union[Any, float]], prefix: str, separator: str
) -> Mapping[str, Union[Any, float]]:
    """Insert prefix before each key in a dict, separated by the separator.

    @param metrics: Dictionary with metric names as keys and measured quantities as values
    @param prefix: Prefix to insert before each key
    @param separator: Separates prefix and original key name

    @returns: Dictionary with prefix and separator inserted before each key

    """
    if not prefix:
        return metrics
    return {f"{prefix}{separator}{k}": v for k, v in metrics.items()}


# Usage example
if __name__ == "__main__":
    # Setup logging and MLflow
    log_dir, mlflow_logger = setup_ml_logging_and_mlflow(
        experiment_name="my_deep_learning_experiment",
        run_name="training_session_v1"
    )

    # Get logger using __name__ (standard approach)
    # logger = get_logger()  # or logging.getLogger(__name__)
    logger = logging.getLogger(__name__)

    # Example usage
    logger.info("Starting training...")
    logger.info("Model architecture: ResNet50")

    # MLflow logging
    mlflow_logger.log_hyperparams({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    })

    # During training loop
    for epoch in range(5):  # Example
        # Log training progress
        logger.info(f"Epoch {epoch + 1}/5")

        # Log metrics to MLflow
        mlflow_logger.log_metrics({
            "train_loss": 0.5 - epoch * 0.1,
            "val_accuracy": 0.8 + epoch * 0.02
        }, step=epoch)

        # Log evaluation results
        logger.info(f"Validation accuracy: {0.8 + epoch * 0.02:.3f}")

    logger.info("Training completed!")
    mlflow_logger.finalize()