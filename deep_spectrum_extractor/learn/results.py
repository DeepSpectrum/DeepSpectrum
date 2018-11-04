from dataclasses import dataclass, field, asdict
from typing import List, Set, Dict, TypeVar
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score, log_loss, f1_score
from sklearn.preprocessing import LabelBinarizer
from enum import Enum
from abc import ABC, abstractmethod
from itertools import chain
from json import dump
from statistics import pstdev, mean
from decimal import Decimal
from scipy.stats import shapiro
import pickle



class ClassificationMetrics(Enum):
    UAR = 'Unweighted Average Recall'
    ACCURACY = 'Classification Accuracy'
    F1 = 'F1 Score'


Metrics = Enum(value='Metrics', names=[(entry.name, entry.value) for enum in [ClassificationMetrics] for entry in enum])

def convert_keys(obj, convert=str):
    if isinstance(obj, list):
        return [convert_keys(i, convert) for i in obj]
    if not isinstance(obj, dict):
        return obj
    return {convert(k): convert_keys(v, convert) for k, v in obj.items()}

def enum_names(key):
    if isinstance(key, ClassificationMetrics):
        return key.name
    return str(key)

def names_to_enum(key):
    try:
        return Metrics[key]
    except KeyError:
        return key


@dataclass
class Result(ABC):
    sort_index: float = field(init=False, repr=False)
    names: List[str]
    predictions: List
    true: List
    metrics: Dict[Metrics, float] = field(init=False)
    _comparison_metric: Metrics
    timestamps: List[Decimal]

    @property
    def comparison_metric(self) -> Metrics:
        return self._comparison_metric

    @comparison_metric.setter
    def comparison_metric(self, value: Metrics):
        self._comparison_metric = value

        # The comparison metric determines how results are compared
        self.sort_index = self.metrics[self._comparison_metric]

    def save(self, path):
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as fp:
            result = pickle.load(fp)
            assert isinstance(result, cls), f'{path} is not a {cls.__name__} object!'
            return


@dataclass(order=True)
class ClassificationResult(Result):
    predictions: List[str]
    decision_func: List[List[float]]
    probabilities: List[List[float]]
    true: List[str]
    labels: Set[str]

    _comparison_metric: ClassificationMetrics

    def __post_init__(self):
        self.metrics = {metric_name: metric(self) for metric_name, metric in METRICS_MAPPING.items()}
        self.sort_index = self.metrics[self._comparison_metric]
        self.labels = sorted(self.labels)


@dataclass(order=True)
class ResultSet(ABC):
    sort_index: float = field(init=False, repr=False)
    _comparison_metric: Metrics
    meta: Dict
    description: str

    def save(self, path):
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as fp:
            result_set = pickle.load(fp)
            assert isinstance(result_set, cls), f'{path} is not a {cls.__name__} object!'
            return result_set

@dataclass(order=True)
class EvalPartitionResultSet(ResultSet):
    eval: Result

    def __post_init__(self):
        self.sort_index = self.eval.metrics[self._comparison_metric]

    @property
    def comparison_metric(self) -> Metrics:
        return self._comparison_metric

    @comparison_metric.setter
    def comparison_metric(self, value: Metrics):
        self._comparison_metric = value
        self.sort_index = self.eval.metrics[self._comparison_metric]


@dataclass
class MetricStats():
    mean: float
    standard_deviation: float
    normality_tests: Dict[str, tuple]


def compute_metric_stats(metric_values: List[float]) -> MetricStats:
    return MetricStats(mean=mean(metric_values), standard_deviation=pstdev(metric_values),
                       normality_tests={'Shapiro-Wilk': shapiro(metric_values)})


@dataclass(order=True)
class CVResultSet(ResultSet):
    folds: List[Result]
    combined: Result = field(init=False)
    metric_stats: Dict[Metrics, MetricStats] = field(init=False)

    def __post_init__(self):

        combined_names = chain(*[fold.names for fold in self.folds])
        combined_predictions = chain(*[fold.predictions for fold in self.folds])
        combined_true = chain(*[fold.true for fold in self.folds])
        if isinstance(self.folds[0], ClassificationResult):
            self.combined = ClassificationResult(names=combined_names, predictions=combined_predictions,
                                                 true=combined_true, labels=self.folds[0].labels,
                                                 _comparison_metric=self._comparison_metric,
                                                 meta={'Description': 'Combined results of Crossvalidation.'})
        else:
            pass
        self.sort_index = self.combined.metrics[self._comparison_metric]
        metric_keys = self.folds[0].metrics.keys()
        self.metric_stats = {
            key: compute_metric_stats([fold.metrics[key] for fold in self.folds]) for key in metric_keys}

    @property
    def comparison_metric(self) -> Metrics:
        return self._comparison_metric

    @comparison_metric.setter
    def comparison_metric(self, value: Metrics):
        self._comparison_metric = value
        self.sort_index = self.combined.metrics[self._comparison_metric]


def uar(result: ClassificationResult) -> float:
    return recall_score(y_true=result.true, y_pred=result.predictions, labels=list(sorted(result.labels)),
                        average='macro')


def accuracy(result: ClassificationResult) -> float:
    return accuracy_score(y_true=result.true, y_pred=result.predictions)


def f1(result: ClassificationResult) -> float:
    return f1_score(y_true=result.true, y_pred=result.predictions, labels=list(sorted(result.labels)), average='macro')


METRICS_MAPPING = {ClassificationMetrics.UAR: uar,
                   ClassificationMetrics.ACCURACY: accuracy,
                   ClassificationMetrics.F1: f1}
