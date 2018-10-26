import tensorflow as tf
from ..command_line_helper import basic_parser, __REGRESSION, __CLASSIFICATION, config, basic_train, basic_eval, basic_predict, save_params, load_params, write_results
from ..data_loader import DataLoader
from .dnn import DNNClassifier, DNNRegressor
from os.path import join


def main():
    parser, train_parser, eval_parser, predict_parser = basic_parser(
        net_name='Deep Neural Network')
    __add_train_args(train_parser)
    __add_eval_args(eval_parser)
    __add_predict_args(predict_parser)
    args = parser.parse_args()
    args.action(args)


def __add_train_args(train_parser):
    train_parser.add_argument(
        '-l',
        '--layers',
        default=[100, 50],
        nargs='+',
        type=int,
        help='Number of units on layers.')
    train_parser.add_argument(
        '-s',
        '--sequences',
        action='store_true',
        help='Whether the input data contains sequences which should be considered in whole.')
    train_parser.add_argument('-o', '--output', default=None, help='Where to store the final evaluation results.')
    train_parser.set_defaults(action=__train)


def __add_predict_args(predict_parser):
    predict_parser.set_defaults(action=__predict)


def __add_eval_args(eval_parser):
    eval_parser.add_argument('-o', '--output', default=None, help='Where to store the evaluation results.')
    eval_parser.set_defaults(action=__eval)


def __train(args):
    loader_params = {
        'sequences': args.sequences,
        'regression': args.mode == __REGRESSION,
        'sequence_classification': args.sequences
    }
    train_data_loader = DataLoader(
        args.training_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_epochs=None,
        **loader_params)
    loader_params['class_weights'] = {class_key: 1.0 for class_key in train_data_loader.class_weights}
    loader_params['max_sequence_len'] = train_data_loader.max_sequence_len
    eval_data_loader = DataLoader(
        args.evaluation_data, batch_size=args.batch_size, **loader_params)
    optimizer = tf.train.AdadeltaOptimizer(args.learning_rate, rho=args.decay_rate)
    configuration = config(model_dir=args.model_dir, keep_checkpoints=args.keep_checkpoints, steps_per_epoch=train_data_loader.steps_per_epoch)
    model_params = {
        'hidden_units': args.layers,
        'feature_columns': train_data_loader.feature_columns,
        'weight_column': train_data_loader.weight_column,
        'dropout': args.dropout,
        'config': configuration,
        'loss_reduction': tf.losses.Reduction.MEAN
    }
    if args.mode == __CLASSIFICATION:
        model_params['n_classes'] = len(train_data_loader.label_dict.keys())
        model_params['label_vocabulary'] = sorted(train_data_loader.label_dict.keys())
        model = DNNClassifier(**model_params,
            optimizer=optimizer)
    elif args.mode == __REGRESSION:
        model = DNNRegressor(**model_params)
    save_params(loader_params=loader_params, model_params=model_params, model_dir=args.model_dir, mode=args.mode)
    metrics = basic_train(model, train_data_loader, eval_data_loader, args.model_dir,
                args.number_of_epochs, args.keep_checkpoints, args.eval_period,
                args.mode)
    if args.output:
        write_results(metrics, args.output)

def __eval(args):
    loader_params, model_params, mode = load_params(args.model_dir)
    eval_data_loader = DataLoader(args.evaluation_data, batch_size=args.batch_size, **loader_params)
    if mode == __CLASSIFICATION:
        model = tf.estimator.DNNClassifier(**model_params)
    metrics = basic_eval(model, eval_data_loader, args.model_dir, args.checkpoint, evaluation_key='on {}'.format(args.evaluation_data), mode=mode)
    if args.output:
        write_results(metrics, args.output)

def __predict(args):
    loader_params, model_params, mode = load_params(args.model_dir)
    predict_data_loader = DataLoader(args.prediction_data, batch_size=args.batch_size, **loader_params)
    if mode == __CLASSIFICATION:
        model = tf.estimator.DNNClassifier(**model_params)
    basic_predict(model, predict_data_loader, args.model_dir, args.output, args.checkpoint, mode=mode)


if __name__ == '__main__':
    main()
