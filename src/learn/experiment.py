from collections import namedtuple

from abc import ABC, abstractmethod
from enum import Enum
from os import makedirs
from os.path import join, basename
from typing import List, Dict, Type

from .metrics import Metric
from .results import ResultSet

DataTuple = namedtuple('DataTuple', ['names', 'timestamps', 'features', 'labels', 'target_names'])

RESULT_EXT = 'json'


class Modes(Enum):
    CLASSIFICATION = 'Classification'
    #REGRESSION = 'Regression'

    def __str__(self):
        return self.value


class Experiment(ABC):

    def __init__(self):
        self.results: Dict[str, ResultSet] = dict()

    @abstractmethod
    def optimize(self, metric: Metric):
        pass

    @abstractmethod
    def evaluate(self, eval_file: str, metric: Type[Metric]):
        pass

    @abstractmethod
    def predict(self, predict_file: str) -> ResultSet:
        pass

    def save(self, path: str):
        makedirs(path, exist_ok=True)
        for result_key, resultset in self.results.items():
            resultset.save(join(path, basename(result_key)) + '.' + RESULT_EXT)

    @abstractmethod
    def _load_training_data(self):
        pass

    @abstractmethod
    def load_feature_file(self, path: str) -> DataTuple:
        pass


class TrainExperiment(Experiment):

    def __init__(self, train_file: str):
        super().__init__()
        self.train_file = train_file
        self.train: DataTuple = None

    def _load_training_data(self):
        self.train = self.load_feature_file(self.train_file)


class DevelExperiment(Experiment):

    def __init__(self, train_file: str, devel_file: str):
        super().__init__()
        self.train_file = train_file
        self.devel_file = devel_file
        self.train: DataTuple = None
        self.devel: DataTuple = None

    def _load_training_data(self):
        self.train = self.load_feature_file(self.train_file)
        self.devel = self.load_feature_file(self.devel_file)


class CrossValidationExperiment(Experiment):

    def __init__(self, fold_files: List[str]):
        super().__init__()
        self.fold_files = fold_files
        self.folds: List[DataTuple] = None

    def _load_training_data(self):
        self.folds = [self.load_feature_file(fold) for fold in self.fold_files]
