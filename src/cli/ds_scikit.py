import argparse
from os.path import join, splitext, abspath

from .ds_help import DESCRIPTION_SCIKIT
from ..learn import Modes
from ..learn.metrics import KEY_TO_METRIC, UAR, RegressionMetric, ClassificationMetric
from ..learn.scikit_models import ScikitDevelExperiment, ScikitCrossValidationExperiment, \
    ScikitRandomCrossValidationExperiment, ScikitExperiment

EXPERIMENT_FILE_NAME = 'experiment.joblib'
GRID_SEARCH_CSV = 'gridsearch.csv'

__TRAIN = 'train'
__EVAL = 'eval'
__PREDICT = 'predict'


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(
        help=DESCRIPTION_SCIKIT)
    _ = __train_subparser(subparsers)

    _ = __eval_subparser(subparsers)
    _ = __predict_subparser(subparsers)
    args = parser.parse_args()
    args.validation_metric = KEY_TO_METRIC[
        args.validation_metric] if args.validation_metric is not None else args.validation_metric
    if args.validation_metric is None:
        if args.mode == Modes.CLASSIFICATION:
            args.validation_metric = UAR
        else:
            args.validation_metric = None
    assert not (args.mode == Modes.CLASSIFICATION and issubclass(
        args.validation_metric, RegressionMetric) or args.mode == Modes.REGRESSION and issubclass(
        args.validation_metric,
        ClassificationMetric)), f'{args.validation_metric} is incompatible with mode {args.mode}!'
    return parser.parse_args()


def __add_basic_args(parser):
    parser.add_argument(
        '-md',
        '--model_dir',
        required=True,
        help=
        'Directory for saving and restoring model and results.'
    )
    parser.add_argument('-m', '--mode', help='Specify what type of machine learning problem should be solved.',
                        default=Modes.CLASSIFICATION, type=Modes,
                        choices=list(Modes))
    parser.add_argument('-vm', '--validation_metric', type=str, choices=KEY_TO_METRIC.keys(),
                        default=None,
                        help='Metric that should be used for finding optimal model parameters and serve as primary metric in the results.')


def __train_subparser(subparsers):
    train_parser = subparsers.add_parser(
        __TRAIN,
        help='Train and optimize a range of scikit-learn models. The best configuration is saved and can be used for evaluation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_parser.add_argument(
        'training_data',
        default=None,
        nargs='+',
        help='Files used for training the model. If no separate validation partition is given, Cross Validation is used. '
             'For a single training file, this means that K (as defined by --cross_validation_folds) random stratified '
             'folds are created.')
    train_parser.add_argument('-vd', '--validation_data', default=None,
                              help='Define a separate validation partition that should be used for model optimization.')
    train_parser.add_argument('-cvf', '--cross_validation_folds', default=10,
                              help='Define number of random stratified crossvalidation folds that should be created from the input file. Only applicable if using single training file and no separate validation data.')
    train_parser.set_defaults(action=__train)
    __add_basic_args(train_parser)
    return train_parser


def __eval_subparser(subparsers):
    eval_parser = subparsers.add_parser(
        __EVAL,
        help='Evaluate a model stored in the given directory on a single data file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    eval_parser.add_argument(
        'evaluation_data',
        help='File to evaluate the model on.')

    eval_parser.set_defaults(action=__eval)
    __add_basic_args(eval_parser)
    return eval_parser


def __predict_subparser(subparsers):
    parser = subparsers.add_parser(
        __PREDICT,
        help='Make prediction with a model stored in the given directory on a single data file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'prediction_data',
        help='File to make predictions on.')
    parser.add_argument(
        '-o', '--output', default=None,
        help='CSV file to output predictions to.')

    parser.set_defaults(action=__predict)
    __add_basic_args(parser)
    return parser


def __train(args):
    if len(args.training_data) > 1:
        print(f'Optimizing sklearn models via crossvalidation of {args.training_data}.')
        if args.validation_data is not None:
            print(
                f'WARNING: Disregarding separate validation file {args.validation_data}. If this is your final evaluation data for the model, call "{__EVAL}" when training is finished!')
        experiment = ScikitCrossValidationExperiment(fold_files=args.training_data, mode=args.mode)

    else:
        if args.validation_data is not None:
            print(
                f'Optimizing sklearn models: Training on {args.training_data} and validating on {args.validation_data}.')
            experiment = ScikitDevelExperiment(train_file=args.training_data[0], devel_file=args.validation_data,
                                               mode=args.mode)
        else:
            print(
                f'Optimizing sklearn models: Training on "{args.training_data[0]}" with random stratified {args.cross_validation_folds}-fold CV.')
            experiment = ScikitRandomCrossValidationExperiment(train_file=args.training_data[0],
                                                               cv=args.cross_validation_folds,
                                                               mode=args.mode)

    print(f'Using {args.validation_metric} as key metric for the {args.mode}.')
    experiment.optimize(KEY_TO_METRIC[args.validation_metric])
    print('\nDetailed information about the achieved results:\n')
    print(experiment.results['validation'])
    experiment.save(args.model_dir)


def __eval(args):
    print(f'Evaluating model in "{args.model_dir}" on "{args.evaluation_data}"\n')
    experiment = ScikitExperiment.load(join(args.model_dir, EXPERIMENT_FILE_NAME))
    experiment.evaluate(eval_file=args.evaluation_data, metric=KEY_TO_METRIC[args.validation_metric])
    print(experiment.results[f'evaluation_on_{splitext(args.evaluation_data)[0]}'])
    experiment.save(args.model_dir)
    print(f'Evaluation results saved to model directory: {args.model_dir}')


def __predict(args):
    print(f'Making predictions with model in "{args.model_dir}" on "{args.prediction_data}"')
    experiment = ScikitExperiment.load(join(args.model_dir, EXPERIMENT_FILE_NAME))
    results = experiment.predict(predict_file=args.prediction_data)
    experiment.save(args.model_dir)
    if args.output is not None:
        results.eval.export_predictions(abspath(args.output))
        print(f'Predictions saved to: "{abspath(args.output)}"')


def main():
    args = parse_args()
    args.action(args)


if __name__ == '__main__':
    main()
