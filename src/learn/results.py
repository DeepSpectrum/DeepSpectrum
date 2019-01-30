import codecs
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from os.path import join
from scipy.stats import ttest_ind
from sklearn.metrics import confusion_matrix
from statsmodels.sandbox.stats.runs import mcnemar
from typing import List, Set, Dict, Type

from .metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS, ClassificationMetric, RegressionMetric, Metric, \
    compute_metric_stats, MetricStats, UAR, MSE


def remove_keys(obj, rubbish):
    if isinstance(obj, dict):
        obj = {
            key: remove_keys(value, rubbish)
            for key, value in obj.items()
            if key not in rubbish}
    elif isinstance(obj, list):
        obj = [remove_keys(item, rubbish)
               for item in obj
               if item not in rubbish]
    return obj


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def none_or_array(array_data_or_none):
    if array_data_or_none is None:
        return None
    else:
        return np.array(array_data_or_none)


@dataclass
class Result(ABC):
    sort_index: float = field(init=False, repr=False)
    names: np.ndarray
    predictions: np.ndarray
    true: np.ndarray
    timestamps: np.ndarray
    metrics: Dict[str, Metric] = field(init=False)
    _comparison_metric: Type[Metric]

    @property
    def comparison_metric(self) -> Type[Metric]:
        return self._comparison_metric

    @comparison_metric.setter
    def comparison_metric(self, value: Type[Metric]):
        self._comparison_metric = value

        # The comparison metric determines how results are compared
        self.sort_index = self.metrics[self.comparison_metric.__name__]

    @abstractmethod
    def export_predictions(self, path):
        pass

    @staticmethod
    def from_csv(csv: str, labels: Set[str] = None, comparison_metric: Type[Metric] = None) -> 'Result':
        frame = pd.read_csv(csv)
        columns = list(frame)
        names = frame['names'].values if 'names' in columns else None
        timestamps = frame['timestamps'].values if 'timestamps' in columns else None
        # check if classification
        if [column for column in columns if column.endswith('label')]:
            predictions = frame['pred_label'].values
            true = frame['true_label'].values if 'true_label' in columns else None
            decision_func_columns = [column for column in columns if column.startswith('decision_func')]
            decision_func_columns = decision_func_columns[0] if len(
                decision_func_columns) == 1 else decision_func_columns
            decision_func = frame[decision_func_columns].values if decision_func_columns else None
            probabilities_columns = [column for column in columns if column.startswith('probability')]
            probabilities_columns = probabilities_columns[0] if len(
                probabilities_columns) == 1 else probabilities_columns
            probabilities = frame[probabilities_columns].values if probabilities_columns else None
            if labels is None:
                if decision_func is not None and len(decision_func.shape) > 1 and decision_func.shape[1] > 1:
                    labels = sorted(set(column_name.split('_')[2] for column_name in decision_func_columns))
                elif probabilities is not None and len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
                    labels = sorted(set(column_name.split('_')[2] for column_name in probabilities_columns))
                elif true is not None:
                    labels = sorted(set(true))
                else:
                    labels = sorted(set(predictions))

            if comparison_metric is None:
                comparison_metric = UAR
            return ClassificationResult(names=names, predictions=predictions, timestamps=timestamps, true=true,
                                        _comparison_metric=comparison_metric, decision_func=decision_func,
                                        probabilities=probabilities, labels=labels)


        else:
            prediction_columns = [column for column in columns if column.startswith('pred')]
            predictions = frame[prediction_columns].values
            targets = [column_name.split('_')[1] for column_name in prediction_columns]
            true_columns = [column for column in columns if column.startswith('true')]
            true = frame[true_columns].values if true_columns else None
            if comparison_metric is None:
                comparison_metric = MSE
            return RegressionResult(names=names, predictions=predictions, timestamps=timestamps, true=true,
                                    targets=targets, _comparison_metric=comparison_metric)


@dataclass(order=True)
class ClassificationResult(Result):
    decision_func: np.ndarray
    probabilities: np.ndarray
    labels: Set[str]
    confusion_matrix: np.ndarray = field(init=False)

    _comparison_metric: Type[ClassificationMetric]

    def __post_init__(self):
        self.labels = sorted(self.labels)
        if self.true is not None:
            self.metrics = {metric.__name__: metric_wrapper(self, metric) for metric in CLASSIFICATION_METRICS}
            self.sort_index = self.metrics[self.comparison_metric.__name__]
            self.confusion_matrix = confusion_matrix(y_true=self.true, y_pred=self.predictions, labels=self.labels)
        else:
            self.metrics = None
            self.confusion_matrix = None
            self.sort_index = 0

    def export_predictions(self, csv_path: str):
        frame = pd.DataFrame()
        if self.names is not None:
            frame['names'] = self.names
        if self.timestamps is not None:
            print(self.timestamps)
            frame['timestamps'] = self.timestamps

        if self.decision_func is not None:
            # Multilabel classification
            if len(self.decision_func.shape) > 1:
                for i, label in enumerate(self.labels):
                    frame[f'decision_func_{label}'] = self.decision_func[:, i]
            # binary classification
            else:
                frame['decision_func'] = self.decision_func

        if self.probabilities is not None:
            # Multilabel classification
            if len(self.probabilities.shape) > 1:
                for i, label in enumerate(self.labels):
                    frame[f'probability_{label}'] = self.probabilities[:, i]
            # binary classification
            else:
                frame['probability'] = self.probabilities

        frame['pred_label'] = self.predictions
        if self.true is not None:
            frame['true_label'] = self.true
        frame.to_csv(csv_path, index=False)


@dataclass(order=True)
class RegressionResult(Result):
    targets: np.array

    _comparison_metric: Type[RegressionMetric]

    def __post_init__(self):
        if self.true is not None:
            self.metrics = {metric.__name__: metric_wrapper(self, metric) for metric in REGRESSION_METRICS}
            self.sort_index = self.metrics[self.comparison_metric.__name__]
        else:
            self.metrics = None
            self.sort_index = 0

    def export_predictions(self, csv_path: str):
        frame = pd.DataFrame()
        if self.names is not None:
            frame['names'] = self.names
        if self.timestamps is not None:
            frame['timestamps'] = self.timestamps

        for i, target in enumerate(self.targets):
            frame[f'pred_{target}'] = self.predictions[:, i]
            if self.true is not None:
                frame[f'true_{target}'] = self.true[:, i]

        frame.to_csv(csv_path, index=False)


@dataclass(order=True)
class ResultSet(ABC):
    sort_index: float = field(init=False, repr=False)
    _comparison_metric: Type[Metric]
    meta: Dict
    description: str

    def save(self, path):
        as_dict = asdict(self)
        clean = remove_keys(as_dict, ['_comparison_metric', 'sort_index'])
        with codecs.open(path, 'w', encoding='utf-8') as out:
            json.dump(clean, out, cls=NumpyEncoder, separators=(',', ':'), sort_keys=True, indent=4)

    @abstractmethod
    def export_predictions(self, directory: str, prefix: str):
        pass

    @classmethod
    def load(cls, path, comparison_metric=None):
        with codecs.open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'folds' in data.keys():
                # The results should be parsed into a CVResultSet object

                if 'labels' in data['combined'].keys():
                    # The results should be parsed into ClassificationResult objects
                    if comparison_metric is None:
                        comparison_metric = UAR

                    combined = ClassificationResult(names=none_or_array(data['combined']['names']),
                                                    timestamps=none_or_array(data['combined']['timestamps']),
                                                    predictions=none_or_array(data['combined']['predictions']),
                                                    true=none_or_array(data['combined']['true']),
                                                    decision_func=none_or_array(data['combined']['decision_func']),
                                                    probabilities=none_or_array(data['combined']['probabilities']),
                                                    labels=data['combined']['labels'],
                                                    _comparison_metric=comparison_metric)
                    folds = [ClassificationResult(names=none_or_array(fold['names']),
                                                  timestamps=none_or_array(fold['timestamps']),
                                                  predictions=none_or_array(fold['predictions']),
                                                  true=none_or_array(fold['true']),
                                                  decision_func=none_or_array(fold['decision_func']),
                                                  probabilities=none_or_array(fold['probabilities']),
                                                  labels=fold['labels'],
                                                  _comparison_metric=comparison_metric) for fold in data['folds']]
                else:
                    if comparison_metric is None:
                        comparison_metric = MSE

                    # RegressionResult
                    combined = RegressionResult(names=none_or_array(data['combined']['names']),
                                                timestamps=none_or_array(data['combined']['timestamps']),
                                                predictions=none_or_array(data['combined']['predictions']),
                                                true=none_or_array(data['combined']['true']),
                                                targets=data['combined']['targets'],
                                                _comparison_metric=comparison_metric)

                    folds = [RegressionResult(names=none_or_array(fold['names']),
                                              timestamps=none_or_array(fold['timestamps']),
                                              predictions=none_or_array(fold['predictions']),
                                              true=none_or_array(fold['true']), targets=fold['targets'],
                                              _comparison_metric=comparison_metric) for fold in data['folds']]

                return CVResultSet(folds=folds, combined=combined, _comparison_metric=comparison_metric,
                                   meta=data['meta'], description=data['description'])
            else:
                # The results should be parsed in to a EvalPartitionResultSet
                if 'labels' in data['eval'].keys():
                    # The results should be parsed into ClassificationResult objects
                    if comparison_metric is None:
                        comparison_metric = UAR

                    eval = ClassificationResult(names=none_or_array(data['eval']['names']),
                                                timestamps=none_or_array(data['eval']['timestamps']),
                                                predictions=none_or_array(data['eval']['predictions']),
                                                true=none_or_array(data['eval']['true']),
                                                decision_func=none_or_array(data['eval']['decision_func']),
                                                probabilities=none_or_array(data['eval']['probabilities']),
                                                labels=data['eval']['labels'],
                                                _comparison_metric=comparison_metric)

                else:
                    if comparison_metric is None:
                        comparison_metric = MSE

                    # RegressionResult
                    eval = RegressionResult(names=none_or_array(data['eval']['names']),
                                            timestamps=none_or_array(data['eval']['timestamps']),
                                            predictions=none_or_array(data['eval']['predictions']),
                                            true=none_or_array(data['eval']['true']), targets=data['eval']['targets'],
                                            _comparison_metric=comparison_metric)

                return EvalPartitionResultSet(eval=eval, _comparison_metric=comparison_metric,
                                              meta=data['meta'], description=data['description'])

    @classmethod
    def from_csv(cls, csvs: List[str], comparison_metric: Type[Metric] = None, labels=None):
        if len(csvs) > 1:
            return CVResultSet.from_csv(csvs, comparison_metric, labels)
        else:
            return EvalPartitionResultSet.from_csv(csvs, comparison_metric, labels)


@dataclass(order=True)
class EvalPartitionResultSet(ResultSet):
    eval: Result

    def __post_init__(self):
        if self.eval.true is not None:
            self.sort_index = self.eval.metrics[self.comparison_metric.__name__]
        else:
            self.sort_index = 0

    @property
    def comparison_metric(self) -> Type[Metric]:
        return self._comparison_metric

    @comparison_metric.setter
    def comparison_metric(self, value: Type[Metric]):
        self._comparison_metric = value
        if self.eval.true is not None:
            self.sort_index = self.eval.metrics[self.comparison_metric.__name__]
        else:
            self.sort_index = 0

    def __str__(self):
        if self.eval.metrics is not None:
            pretty_print = f"""{'*'*100}
    {self.description:80}\n
    Metrics:\n"""
            for metric_type, metric in self.eval.metrics.items():
                pretty_print += f"\n   {metric_type}: {metric.value:.2%}"
            if isinstance(self.eval, ClassificationResult):
                pretty_print += f"\n\nConfusion Matrix:\n{self.eval.confusion_matrix.T}"
            pretty_print += f"\n\nMeta information about the results:\n \n{self.meta}"
            pretty_print += f"\n\n{'*'*100}\n"
        else:
            pretty_print = f'Results only contain predictions! Use "ds-results export" to export them to a predictions csv.\n\nMeta information:\n\n{self.meta}\n'
        return pretty_print

    def export_predictions(self, directory: str, prefix: str):
        prediction_path = join(directory, f'{prefix}_predictions.csv')
        self.eval.export_predictions(prediction_path)

    @classmethod
    def from_csv(cls, csvs: List[str], comparison_metric: Metric = None, labels=None) -> 'EvalPartitionResultSet':
        assert len(
            csvs) == 1, f'{cls.__name__} only supports loading from a single prediction csv! ' \
                        f'If you want to load a CV experiment use {CVResultSet.__name__} instead.'
        result = Result.from_csv(csvs[0], labels=None)
        return EvalPartitionResultSet(eval=result, _comparison_metric=result._comparison_metric, meta=None,
                                      description=f'Results generated from "{csvs[0]}"')


@dataclass(order=True)
class CVResultSet(ResultSet):
    folds: List[Result]
    metric_stats: Dict[str, MetricStats] = field(init=False)
    combined: Result = None

    def __post_init__(self):
        if self.combined is None:

            combined_names = np.concatenate([fold.names for fold in self.folds]) if self.folds[
                                                                                        0].names is not None else None
            combined_predictions = np.concatenate([fold.predictions for fold in self.folds])
            combined_true = np.concatenate([fold.true for fold in self.folds]) if self.folds[
                                                                                      0].true is not None else None
            combined_timestamps = np.concatenate([fold.timestamps for fold in self.folds]) if self.folds[
                                                                                                  0].timestamps is not None else None

            if isinstance(self.folds[0], ClassificationResult):

                combined_decision_func = np.concatenate([fold.decision_func for fold in self.folds]) if self.folds[
                                                                                                            0].decision_func is not None else None
                combined_probabilities = np.concatenate([fold.probabilities for fold in self.folds]) if self.folds[
                                                                                                            0].probabilities is not None else None
                self.combined = ClassificationResult(names=combined_names, predictions=combined_predictions,
                                                     true=combined_true, decision_func=combined_decision_func,
                                                     probabilities=combined_probabilities, labels=self.folds[0].labels,
                                                     timestamps=combined_timestamps,
                                                     _comparison_metric=self._comparison_metric)
            else:
                self.combined = RegressionResult(names=combined_names, predictions=combined_predictions,
                                                 true=combined_true, targets=self.folds[0].targets,
                                                 timestamps=combined_timestamps,
                                                 _comparison_metric=self._comparison_metric)

        metric_keys = self.folds[0].metrics.keys()
        if self.folds[0].true is not None:
            self.sort_index = self.combined.metrics[self.comparison_metric.__name__]
            self.metric_stats = {
                key: compute_metric_stats([fold.metrics[key] for fold in self.folds]) for key in metric_keys}
        else:
            self.sort_index = None
            self.metric_stats = None

    @property
    def comparison_metric(self) -> Type[Metric]:
        return self._comparison_metric

    @comparison_metric.setter
    def comparison_metric(self, value: Type[Metric]):
        self._comparison_metric = value
        if self.combined.true is not None:
            self.sort_index = self.combined.metrics[self.comparison_metric.__name__]
        else:
            self.sort_index = 0

    def __str__(self):
        if self.folds[0].metrics is not None:
            pretty_print = f"""{'*'*100}
    {self.description:80}
    
    Individual folds:\n"""
            for i, fold in enumerate(self.folds):
                pretty_print += f"\n{'_'*50}\n\nFold {i}:\n"
                for metric_type, metric in fold.metrics.items():
                    pretty_print += f"\n  {metric_type}: {metric.value:.2%}"
                if isinstance(fold, ClassificationResult):
                    pretty_print += f"\n\nConfusion Matrix: \n\n{fold.confusion_matrix.T}"
            pretty_print += f"\n{'_'*50}\n\nMetric stats:\n"
            for metric, stats in self.metric_stats.items():
                pretty_print += f"\n  {metric}:\n    Mean: {stats.mean:.2%} (+- {stats.standard_deviation:.2%})"
                pretty_print += f"\n    Normality tests: {stats.normality_tests}\n"

            pretty_print += f"\n{'_'*50}\n\nConcatenated Folds:\n"
            for metric_type, metric in self.combined.metrics.items():
                pretty_print += f"\n  {metric_type}: {metric.value:.2%}"
            if isinstance(self.combined, ClassificationResult):
                pretty_print += f"\n\nConfusion Matrix: \n\n{self.combined.confusion_matrix.T}"
            pretty_print += f"\n{'_'*50}\n\nMeta information about the results:\n  {self.meta}"
            pretty_print += f"\n\n{'*'*50}\n"
        else:
            pretty_print = f'Results only contain predictions! Use "ds-results export" to export them to a predictions csv.\n\nMeta information:\n\n{self.meta}\n'
        return pretty_print

    def export_predictions(self, directory: str, prefix: str):
        combined_path = join(directory, f'{prefix}_predictions_combined.csv')
        self.combined.export_predictions(combined_path)
        for i, fold in enumerate(self.folds):
            fold_path = join(directory, f'{prefix}_predictions_fold-{i}.csv')
            fold.export_predictions(fold_path)

    @classmethod
    def from_csv(cls, csvs: List[str], comparison_metric: Metric = None, labels=None) -> 'CVResultSet':
        assert len(
            csvs) > 1, f'{cls.__name__} only supports loading from a multiple prediction csvs! ' \
                       f'If you want to load a EvalPartitionResultSet use {EvalPartitionResultSet.__name__} instead.'
        fold_results = [Result.from_csv(csv=csv, comparison_metric=comparison_metric, labels=labels) for csv in csvs]
        return CVResultSet(folds=fold_results, _comparison_metric=fold_results[0]._comparison_metric, meta=None,
                           description=f'CVResults generated from "{csvs}"')


def compare(first_result: ResultSet, second_result: ResultSet, comparison_metric: Metric) -> Dict[str, tuple]:
    assert type(first_result) == type(
        second_result), f'Type of first resultset ({type(first_result)}) does not match type of second resultset ({type(second_result)}).'
    first_result.comparison_metric = comparison_metric
    second_result.comparison_metric = comparison_metric
    stats = dict()
    description = "No comparison made."
    if isinstance(first_result, CVResultSet) and isinstance(second_result, CVResultSet):
        description = f'Comparing CVResultSets. Using {comparison_metric.__name__} as comparison metric.\n' \
                      f'\nMetric stats for first result:\n {first_result.metric_stats[comparison_metric.__name__]}\n' \
                      f'\nConcatenated folds of first result:\n {first_result.combined.metrics[comparison_metric.__name__]}\n' \
                      f'\nMetric stats for second result:\n {second_result.metric_stats[comparison_metric.__name__]}\n' \
                      f'\nConcatenated folds of second result:\n {second_result.combined.metrics[comparison_metric.__name__]}\n'

        stats['t-test'] = tuple(ttest(first_result, second_result))

    stats['mcnemar'] = tuple(mcnemar_from_results(first_result, second_result))
    return description, stats


def ttest(first_result: CVResultSet, second_result: CVResultSet) -> tuple:
    first_samples = [fold.metrics[fold.comparison_metric.__name__].value for fold in first_result.folds]
    second_samples = [fold.metrics[fold.comparison_metric.__name__].value for fold in second_result.folds]
    return ttest_ind(first_samples, second_samples)


def mcnemar_from_results(first_resultset: ResultSet, second_resultset: ResultSet) -> tuple:
    if isinstance(first_resultset, CVResultSet) and isinstance(second_resultset, CVResultSet):
        first_result = first_resultset.combined
        second_result = second_resultset.combined
    elif isinstance(first_resultset, EvalPartitionResultSet) and isinstance(second_resultset, EvalPartitionResultSet):
        first_result = first_resultset.eval
        second_result = second_resultset.eval

    left_frame = true_false_frame_from_results(first_result)
    right_frame = true_false_frame_from_results(second_result)
    return mcnemar_from_frames(left_frame, right_frame)


def true_false_frame_from_results(result: ClassificationResult) -> pd.DataFrame:
    true_false_fnc = lambda x: 1 if x[0] == x[1] else 0
    frame = pd.DataFrame()
    frame['names'] = result.names
    frame['timestamps'] = result.timestamps
    frame['predictions'] = result.predictions
    frame['true'] = result.true
    frame['correct'] = frame[['predictions', 'true']].apply(true_false_fnc, axis=1)
    frame.drop(['predictions', 'true'], 1, inplace=True)
    return frame


def mcnemar_from_frames(left_frame, right_frame):
    joined_frame = pd.merge(left_frame, right_frame, how='inner', on=['names', 'timestamps'])
    return mcnemar(joined_frame['correct_x'].values, joined_frame['correct_y'].values, exact=False)


def metric_wrapper(result: Result, metric: Type[Metric]) -> Metric:
    if issubclass(metric, ClassificationMetric):
        assert isinstance(result,
                          ClassificationResult), f'Trying to use a Classification Metric for regression results!'
        return metric.compute(y_true=result.true, y_pred=result.predictions, labels=result.labels)
    elif issubclass(metric, RegressionMetric):
        assert isinstance(result, RegressionResult), f'Trying to use a Regression Metric for classification results!'
        return metric.compute(y_true=result.true, y_pred=result.predictions)
