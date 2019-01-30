from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from scipy.stats import shapiro
from sklearn.metrics import recall_score, make_scorer, accuracy_score, f1_score, mean_squared_error
from sklearn.metrics.scorer import _BaseScorer
from statistics import pstdev, mean
from typing import Dict, List, Iterable, ClassVar, Set




@dataclass(order=True)
class Metric(ABC):
    sort_index: float = field(init=False, repr=False)
    description: ClassVar[str] = 'Metric'
    value: float
    scikit_scorer: ClassVar[_BaseScorer] = field(init=False, repr=False)


@dataclass(order=True)
class ClassificationMetric(Metric, ABC):

    @staticmethod
    @abstractmethod
    def compute(y_true: Iterable, y_pred: Iterable, labels: Set) -> Metric:
        pass


@dataclass(order=True)
class RegressionMetric(Metric, ABC):

    @staticmethod
    @abstractmethod
    def compute(y_true: Iterable, y_pred: Iterable) -> Metric:
        pass


@dataclass(order=True)
class UAR(ClassificationMetric):
    description: ClassVar[str] = 'Unweighted Average Recall'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(recall_score, average='macro')

    def __post_init__(self):
        self.sort_index = self.value

    @staticmethod
    def compute(y_true: Iterable, y_pred: Iterable, labels: Set) -> ClassificationMetric:
        score = recall_score(y_true=y_true, y_pred=y_pred, labels=labels,
                             average='macro')
        return UAR(value=score)


@dataclass(order=True)
class Accuracy(ClassificationMetric):
    description: ClassVar[str] = 'Classification Accuracy'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(accuracy_score)

    def __post_init__(self):
        self.sort_index = self.value

    @staticmethod
    def compute(y_true: Iterable, y_pred: Iterable, labels: Set) -> ClassificationMetric:
        score = accuracy_score(y_true=y_true, y_pred=y_pred)
        return Accuracy(value=score)


@dataclass(order=True)
class F1(ClassificationMetric):
    description: ClassVar[str] = 'F1 Score'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(f1_score, average='macro')

    def __post_init__(self):
        self.sort_index = self.value

    @staticmethod
    def compute(y_true: Iterable, y_pred: Iterable, labels: Set) -> ClassificationMetric:
        score = f1_score(y_true=y_true, y_pred=y_pred, labels=labels,
                         average='macro')
        return F1(value=score)

@dataclass(order=True)
class MSE(RegressionMetric):
    description: ClassVar[str] = 'Mean Squared Error'
    scikit_scorer: ClassVar[_BaseScorer] = make_scorer(mean_squared_error, greater_is_better=False)

    def __post_init__(self):
        self.sort_index = - self.value

    @staticmethod
    def compute(y_true: Iterable, y_pred: Iterable) -> RegressionMetric:
        score = mean_squared_error(y_true=y_true, y_pred=y_pred)
        return MSE(value=score)



@dataclass
class MetricStats():
    mean: float
    standard_deviation: float
    normality_tests: Dict[str, tuple]


def compute_metric_stats(metrics: List[Metric]) -> MetricStats:
    metric_values = [metric.value for metric in metrics]
    normality_tests = dict()
    if len(metric_values) > 2:
        normality_tests['Shapiro-Wilk'] = shapiro(metric_values)
    return MetricStats(mean=mean(metric_values), standard_deviation=pstdev(metric_values),
                       normality_tests=normality_tests)


# scorers for use in scikit-learn classifiers
SCIKIT_CLASSIFICATION_SCORERS = {UAR.__name__: UAR.scikit_scorer,
                                 Accuracy.__name__: Accuracy.scikit_scorer,
                                 F1.__name__: F1.scikit_scorer}

# scorers for use in scikit-learn regressors
SCIKIT_REGRESSION_SCORERS = {MSE.__name__: MSE.scikit_scorer}

CLASSIFICATION_METRICS = [UAR, Accuracy, F1]
REGRESSION_METRICS = [MSE]

KEY_TO_METRIC = {metric.__name__: metric for metric in CLASSIFICATION_METRICS + REGRESSION_METRICS}

