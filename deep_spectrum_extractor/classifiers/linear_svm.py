import argparse
import csv
import arff
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from decimal import Decimal
from os.path import abspath, dirname
from os import makedirs
from ..tools.performance_stats import plot_confusion_matrix

RANDOM_SEED = 42


def _load(file):
    if file.endswith('.arff'):
        return _load_arff(file)
    elif file.endswith('.npz'):
        return _load_npz(file)


def _load_npz(file):
    with np.load(file) as data:
        return data['features'], np.reshape(data['labels'],
                                            (data['labels'].shape[0], ))


def _load_arff(file):
    with open(file) as input:
        dataset = arff.load(input)
    data = np.array(dataset['data'])
    features = data[:, 1:-1].astype(float)
    labels = data[:, -1]
    return features, labels


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
    labels = sorted(set(train_y))
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
        cm = confusion_matrix(test_y, best_prediction, labels=labels)
        return best_uar, cm

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
    labels = sorted(set(train_y))
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
                random_state=RANDOM_SEED,
                scoring='recall_macro')
            if standardize:
                print('Standardizing input...')
                scaler = StandardScaler().fit(train_X)
                train_X = scaler.transform(train_X)
                devel_X = scaler.transform(devel_X)

            scores = cross_val_score(clf, train_X, train_y, cv=10)
            predicted_devel = clf.predict(devel_X)
            UAR_train = scores.mean()
            UAR_devel = recall_score(devel_y, predicted_devel, average='macro')

            print(
                'C: {:.1E} UAR train (CV): {:.2%} (+/- {:.2%}) UAR development: {:.2%}'.
                format(C, UAR_train,
                       scores.std() * 2, UAR_devel))
            if csv_writer:
                csv_writer.writerow([
                    '{:.1E}'.format(Decimal(C)), '{:.2%}'.format(UAR_devel),
                    '{:.2%}'.format(UAR_devel)
                ])
            if UAR_devel > best_uar:
                best_uar = UAR_devel
                best_prediction = predicted_devel
        cm = confusion_matrix(devel_y, best_prediction, labels=labels)
        return best_uar, cm
    finally:
        if csv_file:
            csv_file.close()


def main():
    parser = argparse.ArgumentParser(
        description=
        'Evaluate linear SVM for given Cs on a train, devel, test split of data in arff format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required_named = parser.add_argument_group('Required named arguments')
    required_named.add_argument(
        'i',
        nargs='+',
        help='arff files of training, development and test sets')
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
    args = vars(parser.parse_args())
    if len(args['i']) > 1:
        print('Loading input...')
        train_X, train_y = _load(args['i'][0])
        devel_X, devel_y = _load(args['i'][1])
        labels = sorted(set(train_y))
        if len(args['i']) > 2:
            test_X, test_y = _load(args['i'][2])
            print('Starting training...')
            UAR, cm = parameter_search_train_devel_test(
                train_X,
                train_y,
                devel_X,
                devel_y,
                test_X,
                test_y,
                args['C'],
                output=args['o'],
                standardize=args['standardize'])
        else:
            print('Starting training...')
            UAR, cm = parameter_search_train_devel(
                train_X,
                train_y,
                devel_X,
                devel_y,
                args['C'],
                output=args['o'],
                standardize=args['standardize'])

        if args['cm']:
            cm_path = abspath(args['cm'])
            makedirs(dirname(cm_path), exist_ok=True)
            plot_confusion_matrix(
                cm,
                classes=labels,
                normalize=True,
                title='UAR {:.1%}'.format(UAR),
                save_path=cm_path)

    else:
        parser.error('Unsupported number of partitions.')


if __name__ == '__main__':
    main()
