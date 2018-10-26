from abc import ABC, abstractmethod
from .results import ResultSet, Metrics
from typing import List
from collections import namedtuple

DataTuple = namedtuple('DataTuple', ['names', 'timestamps', 'features', 'labels'])

class Experiment(ABC):

    def __init__(self):
        self.results = None

    @abstractmethod
    def optimize(self, metric: Metrics):
        pass

    @abstractmethod
    def evaluate(self, eval_file: str, metric: Metrics):
        pass

    def save(self, path: str):
        self.results.save(path)

    def load(self, path: str):
        self.results = ResultSet.load(path)

    @classmethod
    @abstractmethod
    def load_file(cls, path: str) -> DataTuple:
        pass


class DevelExperiment(Experiment):

    def __init__(self, train_file: str, devel_file: str):
        super().__init__()
        self.train = self.load_file(train_file)
        self.devel = self.load_file(devel_file)


class CrossValidationExperiment(Experiment):

    def __init__(self, fold_files: List[str]):
        super().__init__()
        self.folds = [self.load_file(fold) for fold in fold_files]
