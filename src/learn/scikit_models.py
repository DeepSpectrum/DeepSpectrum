import arff
import argparse
import csv
import numpy as np
import pandas as pd
from abc import abstractmethod
from decimal import Decimal
from os import makedirs
from os.path import abspath, dirname, join, splitext
from pprint import pprint
from sklearn.base import clone
from sklearn.externals import joblib
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.model_selection import PredefinedSplit, GridSearchCV, StratifiedKFold, KFold, cross_val_score, \
    cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from typing import List, Type

from .experiment import Experiment, DevelExperiment, DataTuple, CrossValidationExperiment, TrainExperiment
from .metrics import SCIKIT_CLASSIFICATION_SCORERS, SCIKIT_REGRESSION_SCORERS, Metric
from .results import EvalPartitionResultSet, CVResultSet, ClassificationResult
from ..learn import Modes
from ..tools.performance_stats import plot_confusion_matrix, save_fig

__MODEL_EXT = 'joblib'

RANDOM_SEED = 42

DESCRIPTION = 'Train and evaluate a linear Support Vector Machine.'



CLASSIFIER_GRID = [
    {'scaler': [StandardScaler(), MinMaxScaler()],
     'estimator': [LinearSVC(random_state=RANDOM_SEED)],
     'estimator__loss': ['hinge', 'squared_hinge'],
     'estimator__C': np.logspace(0, -8, num=9),
     'estimator__class_weight': ['balanced'],
     'estimator__max_iter': [10000]},
    {'scaler': [StandardScaler(), MinMaxScaler(), None], 'estimator': [SGDClassifier(random_state=RANDOM_SEED, tol=0.001)],
      'estimator__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
      'estimator__class_weight': ['balanced'], 'estimator__early_stopping': [True],
      'estimator__n_iter_no_change': [5],
      'estimator__max_iter': [10000]}
]

EXPERIMENT_FILE_NAME = 'experiment.joblib'
GRID_SEARCH_CSV = 'gridsearch.csv'

__TRAIN = 'train'
__EVAL = 'eval'
__PREDICT = 'predict'

REGRESSOR_GRID = None


def clean_meta_dict(meta_dict: dict) -> dict:
    cleaned_meta_dict = meta_dict.copy()
    cleaned_meta_dict['estimator'] = str(cleaned_meta_dict['estimator'])
    cleaned_meta_dict['scaler'] = str(cleaned_meta_dict['scaler'])
    return cleaned_meta_dict


def load(file, label_columns, timestamp_column):
    if file.endswith('.arff'):
        return __load_arff(file, label_columns, timestamp_column)
    elif file.endswith('.csv'):
        return __load_csv(file, label_columns, timestamp_column)


def __load_csv(file, label_columns, timestamp_column):
    df = pd.read_csv(file, sep=',')
    target_names = [df.columns[index] for index in label_columns]
    num_columns = df.shape[1]
    feature_indices, label_columns, timestamp_column = __calculate_indices(num_columns, label_columns, timestamp_column)
    names = df.iloc[:, 0].astype(str).values
    timestamps = df.iloc[:, timestamp_column].astype(float).values if timestamp_column is not None else timestamp_column
    features = df.iloc[:, feature_indices].astype(float).values
    labels = df.iloc[:, label_columns].astype(str).values
    return DataTuple(names=names, timestamps=timestamps, features=features, labels=labels, target_names=target_names)


def __load_arff(file, label_columns, timestamp_column):
    with open(file) as input:
        dataset = arff.load(input)
    data = np.array(dataset['data'])
    target_names = [dataset['attributes'][index][0] for index in label_columns]
    num_columns = data.shape[1]
    feature_indices, label_columns, timestamp_column = __calculate_indices(num_columns, label_columns, timestamp_column)
    names = data[:, 0]
    timestamps = data[:, timestamp_column].astype(float) if timestamp_column is not None else timestamp_column
    features = data[:, feature_indices].astype(float)
    labels = data[:, label_columns]
    return DataTuple(names=names, timestamps=timestamps, features=features, labels=labels, target_names=target_names)


def __calculate_indices(num_columns: int, label_columns: List[int], timestamp_column: int):
    feature_indices = list(range(1, num_columns))
    for lc_i, lc in enumerate(label_columns):
        label_columns[lc_i] = num_columns + lc if lc < 0 else lc  # EC
        feature_indices.remove(label_columns[lc_i])  # EC
    label_columns = label_columns[0] if len(label_columns) == 1 else label_columns
    timestamp_column = num_columns + timestamp_column if timestamp_column is not None and timestamp_column < 0 else timestamp_column
    if timestamp_column is not None:
        feature_indices.remove(timestamp_column)
    return feature_indices, label_columns, timestamp_column


class ScikitExperiment(Experiment):

    def __init__(self, mode: Modes = Modes.CLASSIFICATION, label_columns=[-1], timestamp_column=None, labels=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.parameter_grid = CLASSIFIER_GRID if mode == Modes.CLASSIFICATION else REGRESSOR_GRID
        self.pipeline = Pipeline([('scaler', None), ('estimator', LinearSVC())])
        self.mode = mode
        self.best_estimator = None
        self._grid_search = None
        self._label_columns = label_columns
        self._timestamp_column = timestamp_column
        self._labels = labels

    def _run_grid_search(self, X, y, metric: Metric, cv: int or PredefinedSplit = 5):
        if self.mode == Modes.CLASSIFICATION:
            self._grid_search = GridSearchCV(estimator=self.pipeline, param_grid=self.parameter_grid,
                                             scoring=SCIKIT_CLASSIFICATION_SCORERS,
                                             n_jobs=-1, cv=cv, refit=metric.__name__, verbose=1, iid=False,
                                             return_train_score=False)
        else:
            self._grid_search = GridSearchCV(estimator=self.pipeline, param_grid=self.parameter_grid,
                                             scoring=SCIKIT_REGRESSION_SCORERS,
                                             n_jobs=-1, cv=cv, refit=metric.__name__, verbose=1, iid=False,
                                             return_train_score=False)

        self._grid_search.fit(X, y)
        self.best_estimator = self._grid_search.best_estimator_

    def evaluate(self, eval_file: str, metric: Metric):
        assert self.best_estimator is not None, 'Call "optimize" first!'
        eval_data = self.load_feature_file(eval_file)

        estimator = self.best_estimator
        predictions = estimator.predict(eval_data.features)
        decision_function = None
        probabilities = None
        if hasattr(estimator, 'decision_function'):
            decision_function = estimator.decision_function(eval_data.features)
        if hasattr(estimator, 'predict_proba'):
            probabilities = estimator.predict_proba(eval_data.features)
        if self.mode == Modes.CLASSIFICATION:
            evaluation_results = ClassificationResult(predictions=predictions, true=eval_data.labels,
                                                      labels=self._labels,
                                                      _comparison_metric=metric, decision_func=decision_function,
                                                      probabilities=probabilities, names=eval_data.names,
                                                      timestamps=eval_data.timestamps)
        else:
            evaluation_results = None
            # implement regression

        description = f'Evaluation results for: "{eval_file}"'
        model_params = clean_meta_dict(self._grid_search.best_params_)

        self.results[f'evaluation_on_{splitext(eval_file)[0]}'] = EvalPartitionResultSet(eval=evaluation_results,
                                                                                         description=description,
                                                                                         _comparison_metric=metric,
                                                                                         meta={
                                                                                             'model-parameters': model_params})

    def predict(self, predict_file: str) -> EvalPartitionResultSet:
        assert self.best_estimator is not None, 'Call "optimize" first!'
        eval_data = self.load_feature_file(predict_file)

        estimator = self.best_estimator
        predictions = estimator.predict(eval_data.features)
        decision_function = None
        probabilities = None
        if hasattr(estimator, 'decision_function'):
            decision_function = estimator.decision_function(eval_data.features)
        if hasattr(estimator, 'predict_proba'):
            probabilities = estimator.predict_proba(eval_data.features)
        if self.mode == Modes.CLASSIFICATION:
            evaluation_results = ClassificationResult(predictions=predictions, true=None,
                                                      labels=self._labels,
                                                      _comparison_metric=None, decision_func=decision_function,
                                                      probabilities=probabilities, names=eval_data.names,
                                                      timestamps=eval_data.timestamps)
        else:
            evaluation_results = None
            # implement regression

        description = f'Predictions for: "{predict_file}"'
        model_params = clean_meta_dict(self._grid_search.best_params_)

        self.results[f'predictions_for_{splitext(predict_file)[0]}'] = EvalPartitionResultSet(eval=evaluation_results,
                                                                                           description=description,
                                                                                           _comparison_metric=None,
                                                                                           meta={
                                                                                               'model-parameters': model_params})
        return self.results[f'predictions_for_{splitext(predict_file)[0]}']

    @classmethod
    def load(cls, path: str):
        experiment = joblib.load(path)
        assert isinstance(experiment, cls), f'Not a {cls.__name__} file!'
        return experiment

    @abstractmethod
    def _create_split(self):
        pass

    def load_feature_file(self, path: str) -> DataTuple:
        assert not (len(
            self._label_columns) > 1 and self.mode == Modes.CLASSIFICATION), 'Multilabel classification not supported!'
        return load(path, self._label_columns, self._timestamp_column)

    def save(self, path: str):
        super().save(path)
        joblib.dump(self, join(path, EXPERIMENT_FILE_NAME))
        pd.DataFrame(self._grid_search.cv_results_).to_csv(join(path, GRID_SEARCH_CSV))


class ScikitDevelExperiment(ScikitExperiment, DevelExperiment):

    def __init__(self, train_file: str, devel_file: str, **kwargs):
        super().__init__(train_file=train_file, devel_file=devel_file, **kwargs)
        self._description = f'Scikit-learn models trained on "{train_file}" and optimized on {devel_file}.'

    def _create_split(self):
        num_train = self.train.names.shape[0]
        num_devel = self.devel.names.shape[0]
        split_indices = np.repeat([-1, 0], [num_train, num_devel])
        self._split = PredefinedSplit(split_indices)

    def optimize(self, metric: Metric):
        self._load_training_data()
        self._create_split()
        X = np.append(self.train.features, self.devel.features, axis=0)
        y = np.append(self.train.labels, self.devel.labels)
        if self._labels is None and self.mode == Modes.CLASSIFICATION:
            self._labels = sorted(set(y))
        self._run_grid_search(X, y, metric=metric, cv=self._split)
        estimator = clone(self._grid_search.estimator).set_params(
            **self._grid_search.best_params_)
        estimator.fit(X=self.train.features, y=self.train.labels)
        predictions = estimator.predict(self.devel.features)
        decision_function = None
        probabilities = None
        if hasattr(self.best_estimator, 'decision_function'):
            decision_function = self.best_estimator.decision_function(self.devel.features)
        if hasattr(self.best_estimator, 'predict_proba'):
            probabilities = self.best_estimator.predict_proba(self.devel.features)
        if self.mode == Modes.CLASSIFICATION:
            devel_results = ClassificationResult(predictions=predictions, true=self.devel.labels,
                                                 labels=self._labels, _comparison_metric=metric,
                                                 names=self.devel.names, timestamps=self.devel.timestamps,
                                                 decision_func=decision_function, probabilities=probabilities)
        else:
            # Implement Regression
            pass

        model_params = clean_meta_dict(self._grid_search.best_params_)
        self.results['validation'] = EvalPartitionResultSet(eval=devel_results, description=self._description,
                                                            _comparison_metric=metric,
                                                            meta={'model-parameters': model_params})
        print(
            f'Optimization finished. Best score: \n\n {self._grid_search.best_score_}\n\nFor parameters: \n\n {self._grid_search.best_params_}\n')


class ScikitCrossValidationExperiment(ScikitExperiment, CrossValidationExperiment):

    def __init__(self, fold_files: List[str], **kwargs):
        super().__init__(fold_files=fold_files, **kwargs)
        self._split = None
        self._description = f'Scikit-learn models trained on {fold_files} via {len(fold_files)}-fold crossvalidation.'

    def _create_split(self):
        split_indices = np.repeat(range(len(self.folds)), [fold.names.shape[0] for fold in self.folds])
        self._split = PredefinedSplit(split_indices)

    def optimize(self, metric: Metric):
        self._load_training_data()
        self._create_split()
        combined_features = np.concatenate([fold.features for fold in self.folds], axis=0)
        combined_labels = np.concatenate([fold.labels for fold in self.folds], axis=0)
        combined_names = np.concatenate([fold.names for fold in self.folds], axis=0)
        combined_timestamps = np.concatenate([fold.timestamps for fold in self.folds], axis=0) if self.folds[
                                                                                                      0].timestamps is not None else None

        if self._labels is None and self.mode == Modes.CLASSIFICATION:
            self._labels = sorted(set(combined_labels))
        self._run_grid_search(combined_features, combined_labels, metric=metric, cv=self._split)
        estimator = clone(self._grid_search.estimator).set_params(
            **self._grid_search.best_params_)
        predictions = cross_val_predict(estimator, combined_features, combined_labels, cv=self._split)
        decision_function = None
        probabilities = None
        if hasattr(self.best_estimator, 'decision_function'):
            decision_function = cross_val_predict(estimator, combined_features, combined_labels, cv=self._split,
                                                  method='decision_function')
        if hasattr(self.best_estimator, 'predict_proba'):
            probabilities = cross_val_predict(estimator, combined_features, combined_labels, cv=self._split,
                                              method='predict_proba')

        fold_results = []
        for i, (_, test_index) in enumerate(self._split.split(np.zeros(len(combined_labels)), combined_labels)):
            if self.mode == Modes.CLASSIFICATION:
                fold_decision = decision_function[test_index] if decision_function is not None else None
                fold_predictions = predictions[test_index]
                fold_probabilities = probabilities[test_index] if probabilities is not None else None
                fold_labels = combined_labels[test_index]
                fold_names = combined_names[test_index]
                fold_timestamps = combined_timestamps[test_index] if combined_timestamps is not None else None
                fold_results.append(ClassificationResult(predictions=fold_predictions, true=fold_labels,
                                                         labels=self._labels, _comparison_metric=metric,
                                                         names=fold_names, timestamps=fold_timestamps,
                                                         decision_func=fold_decision, probabilities=fold_probabilities))

        if self.mode == Modes.CLASSIFICATION:
            combined_results = ClassificationResult(predictions=predictions, true=combined_labels,
                                                    labels=self._labels, _comparison_metric=metric,
                                                    names=combined_names, timestamps=combined_timestamps,
                                                    decision_func=decision_function, probabilities=probabilities)

        model_params = clean_meta_dict(self._grid_search.best_params_)
        self.results['validation'] = CVResultSet(folds=fold_results, combined=combined_results,
                                                 description=self._description,
                                                 _comparison_metric=metric,
                                                 meta={'model-parameters': model_params})
        print(
            f'Optimization finished. Best score: \n\n {self._grid_search.best_score_}\n\nFor parameters: \n\n {self._grid_search.best_params_}\n')


class ScikitRandomCrossValidationExperiment(ScikitExperiment, TrainExperiment):

    def __init__(self, train_file: str, cv: int = 10, **kwargs):
        super().__init__(train_file=train_file, **kwargs)
        self._cv = cv
        self._split = None
        self._description = f'Scikit-learn models trained on "{train_file}" and optimized with random stratified {self._cv}-fold crossvalidation.'

    def _create_split(self):
        if self.mode == Modes.CLASSIFICATION:
            self._split = StratifiedKFold(n_splits=self._cv)
        else:
            self._split = KFold(n_splits=self._cv)

    def optimize(self, metric: Metric):
        self._load_training_data()
        self._create_split()
        if self._labels is None and self.mode == Modes.CLASSIFICATION:
            self._labels = sorted(set(self.train.labels))
        self._run_grid_search(self.train.features, self.train.labels, metric=metric, cv=self._split)
        estimator = clone(self._grid_search.estimator).set_params(
            **self._grid_search.best_params_)
        predictions = cross_val_predict(estimator, self.train.features, self.train.labels, cv=self._split)
        decision_function = None
        probabilities = None
        if hasattr(self.best_estimator, 'decision_function'):
            decision_function = cross_val_predict(estimator, self.train.features, self.train.labels, cv=self._split,
                                                  method='decision_function')
        if hasattr(self.best_estimator, 'predict_proba'):
            probabilities = cross_val_predict(estimator, self.train.features, self.train.labels, cv=self._split,
                                              method='predict_proba')

        fold_results = []
        for i, (_, test_index) in enumerate(self._split.split(np.zeros(len(self.train.labels)), self.train.labels)):
            if self.mode == Modes.CLASSIFICATION:
                fold_decision = decision_function[test_index] if decision_function is not None else None
                fold_predictions = predictions[test_index]
                fold_probabilities = probabilities[test_index] if probabilities is not None else None
                fold_labels = self.train.labels[test_index]
                fold_names = self.train.names[test_index]
                fold_timestamps = self.train.timestamps[test_index] if self.train.timestamps is not None else None
                fold_results.append(ClassificationResult(predictions=fold_predictions, true=fold_labels,
                                                         labels=self._labels, _comparison_metric=metric,
                                                         names=fold_names, timestamps=fold_timestamps,
                                                         decision_func=fold_decision, probabilities=fold_probabilities))
        if self.mode == Modes.CLASSIFICATION:
            combined_results = ClassificationResult(predictions=predictions, true=self.train.labels,
                                                    labels=self._labels, _comparison_metric=metric,
                                                    names=self.train.names, timestamps=self.train.timestamps,
                                                    decision_func=decision_function, probabilities=probabilities)

        model_params = clean_meta_dict(self._grid_search.best_params_)
        self.results['validation'] = CVResultSet(folds=fold_results, combined=combined_results,
                                                 description=self._description,
                                                 _comparison_metric=metric,
                                                 meta={'model-parameters': model_params})
        print(
            f'Optimization finished. Best score: \n\n {self._grid_search.best_score_}\n\nFor parameters: \n\n {self._grid_search.best_params_}\n')


def _load(file):
    if file.endswith('.arff'):
        return _load_arff(file)
    elif file.endswith('.npz'):
        return _load_npz(file)
    elif file.endswith('.csv'):
        return _load_csv(file)


def _load_csv(file):
    df = pd.read_csv(file, sep=',')
    names = df.iloc[:, 0].astype(str)
    features = df.iloc[:, 1:-1].astype(float)
    labels = df.iloc[:, -1].astype(str)
    return names, features, labels


def _load_npz(file):
    with np.load(file) as data:
        return data['features'], np.reshape(data['labels'],
                                            (data['labels'].shape[0],))


def _load_arff(file):
    with open(file) as input:
        dataset = arff.load(input)
    data = np.array(dataset['data'])
    names = data[:, 0]
    features = data[:, 1:-1].astype(float)
    labels = data[:, -1]
    return names, features, labels


def write_predictions(filepath, predictions, names=None, true_labels=None):
    columns = []
    if names is not None:
        columns.append('name')
    columns.append('prediction')
    if true_labels is not None:
        columns.append('true')
    prediction_frame = pd.DataFrame(columns=columns)
    prediction_frame['prediction'] = predictions
    if names is not None:
        prediction_frame['name'] = names
    if true_labels is not None:
        prediction_frame['true'] = true_labels
    prediction_frame.to_csv(filepath, index=False)


def parameter_search_train_devel_test(train_X,
                                      train_y,
                                      devel_X,
                                      devel_y,
                                      test_X,
                                      test_y,
                                      Cs=np.logspace(0, -6, num=7),
                                      output=None,
                                      standardize=False):
    csv_writer = None
    csv_file = None
    best_uar = 0
    traindevel_X = np.append(train_X, devel_X, axis=0)
    traindevel_y = np.append(train_y, devel_y)
    try:
        if standardize:
            print('Standardizing input...')
            train_scaler = StandardScaler().fit(train_X)
            train_X = train_scaler.transform(train_X)
            devel_X = train_scaler.transform(devel_X)

            traindevel_scaler = StandardScaler().fit(traindevel_X)
            traindevel_X = traindevel_scaler.transform(traindevel_X)
            test_X = traindevel_scaler.transform(test_X)

        if output:
            output = abspath(output)
            dir = dirname(output)
            makedirs(dir, exist_ok=True)
            csv_file = open(output, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Complexity', 'UAR Development', 'UAR Test'])
        for C in Cs:
            clf = LinearSVC(
                C=C, class_weight='balanced', random_state=RANDOM_SEED)
            clf.fit(train_X, train_y)
            predicted_devel = clf.predict(devel_X)
            UAR_devel = recall_score(devel_y, predicted_devel, average='macro')

            clf = LinearSVC(C=C, class_weight='balanced', random_state=42)
            clf.fit(traindevel_X, traindevel_y)
            predicted_test = clf.predict(test_X)
            UAR_test = recall_score(test_y, predicted_test, average='macro')
            result = ClassificationResult(names=None, predictions=list(predicted_test), true=list(test_y),
                                          labels=set(test_y),
                                          meta={'C': f'{C:.1E}'}, comparison_metric=ClassificationMetrics.UAR)
            pprint(result)
            result.save('test.json')
            print('C: {:.1E} UAR development: {:.2%} UAR test: {:.2%}'.format(
                Decimal(C), UAR_devel, UAR_test))
            if csv_writer:
                csv_writer.writerow([
                    '{:.1E}'.format(Decimal(C)), '{:.2%}'.format(UAR_devel),
                    '{:.2%}'.format(UAR_test)
                ])
            if UAR_test > best_uar:
                best_uar = UAR_test
                best_prediction = predicted_test
        return best_uar, best_prediction

    finally:
        if csv_file:
            csv_file.close()


def parameter_search_cross_validation(folds_X: list, folds_y: list, Cs=np.logspace(0, -9, num=10), output=None,
                                      standardize=False):
    csv_writer = None
    csv_file = None
    best_uar = 0

    try:

        if output:
            output = abspath(output)
            dir = dirname(output)
            makedirs(dir, exist_ok=True)
            csv_file = open(output, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                ['Complexity'] + ['UAR fold {}'.format(idx) for idx in range(len(folds_y))] + ['UAR combined'])
        for C in Cs:
            scores = []
            predictions = None
            for eval_index in range(len(folds_X)):
                print('Processing fold {}...'.format(eval_index))
                eval_X, eval_y = folds_X[eval_index], folds_y[eval_index]
                train_X, train_y = np.concatenate(
                    [X for index, X in enumerate(folds_X) if index is not eval_index]), np.concatenate(
                    [y for index, y in enumerate(folds_y) if index is not eval_index])
                clf = LinearSVC(
                    C=C,
                    class_weight='balanced',
                    random_state=RANDOM_SEED)
                if standardize:
                    print('Standardizing input...')
                    scaler = StandardScaler().fit(train_X)
                    train_X = scaler.transform(train_X)
                    eval_X = scaler.transform(eval_X)
                clf.fit(train_X, train_y)
                predicted_eval = clf.predict(eval_X)
                scores.append(recall_score(eval_y, predicted_eval, average='macro'))
                if predictions is None:
                    predictions = predicted_eval
                else:
                    predictions = np.append(predictions, predicted_eval)

            UAR = recall_score(np.concatenate(folds_y), predictions, average='macro')

            print(
                'C: {:.1E} UAR (CV): {:.2%} (+/- {:.2%})'.
                    format(C, UAR,
                           np.std(scores, ddof=1) * 2))
            if csv_writer:
                csv_writer.writerow([
                                        '{:.1E}'.format(Decimal(C))] + ['{:.2%}'.format(UAR) for UAR in scores] + [
                                        '{:.2%}'.format(UAR)])
            if UAR > best_uar:
                best_uar = UAR
                best_prediction = predictions
        return best_uar, best_prediction
    finally:
        if csv_file:
            csv_file.close()


def parameter_search_train_devel(train_X,
                                 train_y,
                                 devel_X,
                                 devel_y,
                                 Cs=np.logspace(0, -9, num=10),
                                 output=None,
                                 standardize=False):
    csv_writer = None
    csv_file = None
    best_uar = 0
    try:

        if output:
            output = abspath(output)
            dir = dirname(output)
            makedirs(dir, exist_ok=True)
            csv_file = open(output, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Complexity', 'UAR Train', 'UAR Development'])
        for C in Cs:
            clf = LinearSVC(
                C=C,
                class_weight='balanced',
                random_state=RANDOM_SEED)
            if standardize:
                print('Standardizing input...')
                scaler = StandardScaler().fit(train_X)
                train_X = scaler.transform(train_X)
                devel_X = scaler.transform(devel_X)

            scores = cross_val_score(clf, train_X, train_y, cv=10, scoring='recall_macro')
            UAR_train = scores.mean()
            clf.fit(train_X, train_y)
            predicted_devel = clf.predict(devel_X)
            UAR_devel = recall_score(devel_y, predicted_devel, average='macro')

            print(
                'C: {:.1E} UAR train (CV): {:.2%} (+/- {:.2%}) UAR development: {:.2%}'.
                    format(C, UAR_train,
                           scores.std() * 2, UAR_devel))
            if csv_writer:
                csv_writer.writerow([
                    '{:.1E}'.format(Decimal(C)), '{:.2%}'.format(UAR_train),
                    '{:.2%}'.format(UAR_devel)
                ])
            if UAR_devel > best_uar:
                best_uar = UAR_devel
                best_prediction = predicted_devel
        return best_uar, best_prediction
    finally:
        if csv_file:
            csv_file.close()


def main_old():
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required_named = parser.add_argument_group('Required named arguments')
    required_named.add_argument(
        'i',
        nargs='+',
        help='arff files')
    parser.add_argument(
        '-C',
        nargs='+',
        type=Decimal,
        help='Complexities for SVM.',
        required=False,
        default=np.logspace(0, -9, num=10))
    parser.add_argument(
        '-o', help='Output path.', required=False, default=None)
    parser.add_argument(
        '-cm', help='Confusion matrix path.', required=False, default=None)
    parser.add_argument(
        '--standardize',
        help=
        'Standardize input data. Standardization parameters are determined on the training partition and applied to the test set.',
        action='store_true')
    parser.add_argument(
        '-pred', help='Filepath for prediction output in csv format.', required=False, default=None)
    args = vars(parser.parse_args())
    print('Loading input...')
    if len(args['i']) < 2:
        parser.error('Unsupported number of partitions.')
        return
    elif len(args['i']) > 3:
        folds_names, folds_X, folds_y = zip(*(_load(file) for file in args['i']))
        labels = sorted(set((label for fold in folds_y for label in fold)))
        print('Starting training...')
        UAR, best_prediction = parameter_search_cross_validation(list(folds_X), list(folds_y), args['C'],
                                                                 output=args['o'],
                                                                 standardize=args['standardize'])
        true_labels = np.concatenate(list(folds_y))
        cm = confusion_matrix(true_labels, best_prediction, labels=labels)
        names = np.concatenate(list(folds_names))
    elif len(args['i']) > 1:
        clf = LinearSVC()
        experiment = ScikitDevelExperiment(train_file=args['i'][0], devel_file=args['i'][1], estimator=clf,
                                           parameter_grid=[{'C': [0, 1]}])
        print(experiment.train, experiment.estimator, experiment.pipeline)
        train_names, train_X, train_y = _load(args['i'][0])
        devel_names, devel_X, devel_y = _load(args['i'][1])
        labels = sorted(set(train_y))
        if len(args['i']) > 2:
            test_names, test_X, test_y = _load(args['i'][2])
            print('Starting training...')
            UAR, best_prediction = parameter_search_train_devel_test(
                train_X,
                train_y,
                devel_X,
                devel_y,
                test_X,
                test_y,
                args['C'],
                output=args['o'],
                standardize=args['standardize'])
            cm = confusion_matrix(test_y, best_prediction, labels=labels)
            true_labels = test_y
            names = test_names
        else:
            print('Starting training...')
            UAR, best_prediction = parameter_search_train_devel(
                train_X,
                train_y,
                devel_X,
                devel_y,
                args['C'],
                output=args['o'],
                standardize=args['standardize'])
            cm = confusion_matrix(devel_y, best_prediction, labels=labels)
            true_labels = devel_y
            names = devel_names
    if args['cm']:
        cm_path = abspath(args['cm'])
        makedirs(dirname(cm_path), exist_ok=True)
        fig = plot_confusion_matrix(
            cm,
            classes=labels,
            normalize=True,
            title='UAR {:.1%}'.format(UAR))
        save_fig(fig, cm_path)

    if args['pred']:
        prediction_path = abspath(args['pred'])
        makedirs(dirname(prediction_path), exist_ok=True)
        write_predictions(prediction_path, best_prediction, names=names, true_labels=true_labels)


# if __name__ == '__main__':
#     main()
