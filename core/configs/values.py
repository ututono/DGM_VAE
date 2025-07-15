# This script is to store
from enum import Enum


class TypeEnum(Enum):
    """
    Base class for enums with string representation and fast lookup.
    """

    def __init__(self, value):
        self._id = value

    def __eq__(self, other):
        if isinstance(other, TypeEnum):
            return self._id == other._id
        elif isinstance(other, str):
            return self._id == other

    def __hash__(self):
        return hash(self._id)

    def __str__(self):
        """
        Return string representation of the enum.
        @return: String representation.
        """
        return self.value


class TrainingState(TypeEnum):
    """
    Enum for training states.
    """
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"  # completed state indicates successful completion of the training process
    FAILED = "failed"


class DataSplitType(TypeEnum):
    """
    Enum for data split types.
    """
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "val"

class VAEModelType(TypeEnum):
    """
    Enum for VAE model types.
    """
    VAE = "vae"
    CVAE = "cvae"


class DatasetLabelType(TypeEnum):
    """
    Enum for dataset label types.
    - single: the dataset has a single label for each sample
    - multi: the dataset has multiple labels for each sample
    """
    SINGLE = "single"
    MULTI = "multi"

class DatasetLabelInfoNames(TypeEnum):
    """
    Key names in dataset label info.
    - type: the type of the label (e.g. single, multi)
    - n_classes: the number of classes in the label
    """
    TYPE = "type"
    N_CLASSES = "n_classes"
