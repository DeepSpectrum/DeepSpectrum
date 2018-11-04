from abc import ABC, abstractmethod
from .results import ResultSet, Metrics
from typing import List, Dict
from collections import namedtuple
from os.path import join
from os import makedirs

DataTuple = namedtuple('DataTuple', ['names', 'timestamps', 'features', 'labels'])

RESULT_EXT = 'results'


class Experiment(ABC):

    def __init__(self):
        self.results: Dict[str, ResultSet] = dict()

    @abstractmethod
    def optimize(self, metric: Metrics):
        pass

    @abstractmethod
    def evaluate(self, eval_file: str, metric: Metrics):
        pass

    def save(self, path: str):
        makedirs(path, exist_ok=True)
        for result_key, resultset in self.results.items():
            resultset.save(join(path, result_key) + '.' + RESULT_EXT)

    def load(self, path: str):
        self.results = ResultSet.load(path)

    @classmethod
    @abstractmethod
    def load_feature_file(cls, path: str) -> DataTuple:
        pass


class DevelExperiment(Experiment):

    def __init__(self, train_file: str, devel_file: str):
        super().__init__()
        self.train = self.load_feature_file(train_file)
        self.devel = self.load_feature_file(devel_file)


class CrossValidationExperiment(Experiment):

    def __init__(self, fold_files: List[str]):
        super().__init__()
        self.folds = [self.load_feature_file(fold) for fold in fold_files]
