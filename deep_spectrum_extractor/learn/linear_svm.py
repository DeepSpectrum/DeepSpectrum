import argparse
import csv
import arff
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, confusion_matrix, make_scorer, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from decimal import Decimal
from os.path import abspath, dirname
from os import makedirs
from ..tools.performance_stats import plot_confusion_matrix, save_fig
from deep_spectrum_extractor.learn.results import ClassificationResult, ClassificationMetrics
from dataclasses import asdict
from .experiment import Experiment, DevelExperiment, DataTuple, Metrics
from .results import DevelTestResultSet
from pprint import pprint
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from typing import List, Dict
from sklearn.model_selection import PredefinedSplit, GridSearchCV

CLASSIFICATION_SCORERS = {Metrics.UAR: make_scorer(recall_score, average='macro'),
                          Metrics.F1: make_scorer(f1_score, average='macro'),
                          Metrics.ACCURACY: accuracy_score}

RANDOM_SEED = 42


def load(file):
    if file.endswith('.arff'):
        return __load_arff(file)
    elif file.endswith('.csv'):
        return __load_csv(file)


def __load_csv(file):
    df = pd.read_csv(file, sep=',')
    names = df.iloc[:, 0].astype(str).values
    features = df.iloc[:, 1:-1].astype(float).values
    labels = df.iloc[:, -1].astype(str).values
    return DataTuple(names=names, timestamps=None, features=features, labels=labels)


def __load_arff(file):
    with open(file) as input:
        dataset = arff.load(input)
    data = np.array(dataset['data'])
    names = data[:, 0]
    features = data[:, 1:-1].astype(float)
    labels = data[:, -1]
    return DataTuple(names=names, timestamps=None, features=features, labels=labels)


class ScikitExperiment(Experiment):

    def __init__(self, estimator: BaseEstimator, parameter_grid: List[Dict], standardize=False, **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator
        self.parameter_grid = parameter_grid
        self.standardize = standardize
        if self.standardize:
            self.pipeline = Pipeline([('standard-scaler', StandardScaler), (estimator.__class__.__name__, estimator)])
        else:
            self.pipeline = Pipeline([(estimator.__class__.__name__, estimator)])

    @classmethod
    def load_file(cls, path: str) -> DataTuple:
        return load(path)


class ScikitDevelExperiment(ScikitExperiment, DevelExperiment):

    def __init__(self, train_file: str, devel_file: str, **kwargs):
        super().__init__(train_file=train_file, devel_file=devel_file, **kwargs)
        num_train = self.train.names.shape
        num_devel = self.devel.names.shape
        self._all_features = np.append(self.train.features, self.devel.features, axis=0)
        self._all_labels = np.append(self.train.labels, self.devel.labels)
        split_indices = np.repeat([-1, 0], [num_train, num_devel])
        self._split = PredefinedSplit(split_indices)

    def optimize(self, metric: Metrics):
        scorer = CLASSIFICATION_SCORERS[metric]
        self._grid_search = GridSearchCV(estimator=self.pipeline, param_grid=self.parameter_grid,
                                         scoring=CLASSIFICATION_SCORERS,
                                         n_jobs=-1, cv=self._split, refit=scorer)
        self._grid_search.fit(self._all_features, self._all_labels)
        estimator = clone(self._grid_search.estimator).set_params(
            **self._grid_search.best_params_)
        estimator.fit(X=self.train.features, y=self.train.labels)
        predictions = estimator.predict(self.devel.features)
        devel_results = ClassificationResult(predictions=predictions, true=self.devel.labels, labels=set(self.devel.labels), _comparison_metric=metric)
        self.results = DevelTestResultSet(devel=devel_results, test=None)


    def evaluate(self, eval_file: str, metric: Metrics):
        pass


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


def main():
    parser = argparse.ArgumentParser(
        description=
        'Evaluate linear SVM for given Cs on data in arff format',
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


if __name__ == '__main__':
    main()
