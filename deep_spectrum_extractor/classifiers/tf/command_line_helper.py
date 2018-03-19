import argparse
import tensorflow as tf
import pickle
import csv
from .rnn.rnn import RNNClassifier
from .dnn.dnn import DNNClassifier, DNNRegressor
from .data_loader import DataLoader, _WEIGHT_COLUMN
from .hooks import EvalConfusionMatrixHook
from os.path import join, basename
from os import makedirs

__TF_PLOT = True
try:
    import tfplot
except ImportError:
    __TF_PLOT = False

# Modes (Classification and Regression)
__CLASSIFICATION = 'classification'
__REGRESSION = 'regression'
__MODE_KEYS = [__CLASSIFICATION, __REGRESSION]

# Available network architectures:
_RNN = 'Recurrent Neural Network'
_DNN = 'Deep Neural Network'

__NETWORKS = {
    _DNN: {
        __CLASSIFICATION: DNNClassifier
    },
    _RNN: {
        __CLASSIFICATION: RNNClassifier
    }
}
__NETWORK_KEYS = __NETWORKS.keys()

# Phases
__TRAIN = 'train'
__EVAL = 'eval'
__PREDICT = 'predict'
__PHASE_KEYS = [__TRAIN, __EVAL, __PREDICT]


def __train_parser(net_name='classifier'):
    parser = argparse.ArgumentParser(
        description=
        'Train a {} on Deep Spectrum features in arff or csv format.'.format(
            net_name))
    parser.add_argument(
        '-train',
        required=True,
        default=None,
        help='file used for training the model.')
    parser.add_argument(
        '-eval',
        required=True,
        default=None,
        help='file used for validation during training.')
    parser.add_argument(
        '--batch_size', default=32, type=int, help='Batchsize for training.')
    parser.add_argument(
        '--ne',
        default=10,
        type=int,
        help='Number of epochs to train the model.')
    parser.add_argument(
        '--model_dir',
        default='model',
        help=
        'Directory for saving and restoring model checkpoints, summaries and exports.'
    )
    parser.add_argument(
        '--layers',
        default=[500, 100],
        nargs='+',
        type=int,
        help='Number of units of layers.')
    parser.add_argument(
        '--eval_period',
        default=5,
        type=int,
        help='Evaluation interval in seconds.')
    parser.add_argument(
        '--keep_checkpoints',
        default=20,
        type=int,
        help='How many checkpoints to keep stored on disk.')
    parser.add_argument(
        '--lr', default=0.001, type=float, help='Learning rate.')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout')
    parser.add_argument(
        '--mode',
        default=__CLASSIFICATION,
        help='Perform regression or classification task.',
        choices=__MODE_KEYS)
    return parser


def __eval_parser(net_name='classifier'):
    parser = argparse.ArgumentParser(
        description=
        'Evaluate a {} on numerical features in arff or csv format.'.format(
            net_name))
    parser.add_argument(
        '-eval', required=True, help='Features to evaluate model on.')
    parser.add_argument(
        '--model_dir', default='model', help='Directory of trained model.')
    parser.add_argument(
        '--ckpt', default=None, help='Specific checkpoint to evaluate.')
    parser.add_argument('--batch_size', default=32, help='Batchsize.')
    return parser


def __predict_parser(net_name='classifier'):
    parser = argparse.ArgumentParser(
        description=
        'Make predictions with a {} on numerical features in arff or csv format.'.
        format(net_name))
    parser.add_argument(
        '-predict', required=True, help='Features to make predictions for.')
    parser.add_argument(
        '--model_dir', default='model', help='Directory of trained model.')
    parser.add_argument(
        '--ckpt', default=None, help='Specific checkpoint to evaluate.')
    parser.add_argument('--batch_size', default=32, help='Batchsize.')
    parser.add_argument(
        '-output', default='predictions.csv', help='Path to output csv.')
    return parser


def __extend_parser(parser, network_key):
    if network_key == _RNN:
        parser = __add_rnn_args(parser)
    elif network_key == _DNN:
        parser = __add_dnn_args(parser)
    return parser


def __add_rnn_args(parser):
    parser.add_argument(
        '--type',
        default='lstm',
        choices=['lstm', 'gru'],
        help='Type of RNN cell to use in model.')
    parser.add_argument(
        '--sequenced_labels',
        action='store_true',
        help='Whether each element of a sequence has its own label.')
    return parser


def __add_dnn_args(parser):
    parser.add_argument(
        '--sequences',
        action='store_true',
        help=
        'Whether the input data contains sequences which should be considered as a whole.'
    )
    return parser


def __get_data_loader(filepath,
                      network,
                      phase,
                      mode,
                      batch_size,
                      sequences=False,
                      sequenced_labels=False,
                      class_weights=None,
                      max_sequence_len=None):
    shuffle = phase == __TRAIN
    num_epochs = None if phase == __TRAIN else 1
    regression = mode == __REGRESSION
    if network == _RNN:
        sequence_classification = not sequenced_labels

    elif network == _DNN:
        sequences = max_sequence_len is not None or sequences
        sequence_classification = True

    data_loader = DataLoader(
        filepath,
        sequences=sequences,
        sequence_classification=sequence_classification,
        batch_size=batch_size,
        shuffle=shuffle,
        num_epochs=num_epochs,
        num_threads=1,
        queue_capacity=10000,
        regression=regression,
        class_weights=class_weights,
        max_sequence_len=max_sequence_len)
    return data_loader


def __create_model(network_key, data, optimizer, args, config):
    if network_key == _DNN:
        return __create_dnn(data, optimizer, args, config)
    elif network_key == _RNN:
        return __create_rnn(data, optimizer, args, config)


def __create_dnn(data, optimizer, args, config):
    params = {
        'feature_columns': data.feature_columns,
        'hidden_units': args.layers,
        'n_classes': len(data.label_dict),
        'model_dir': args.model_dir,
        'dropout': args.dropout,
        'optimizer': optimizer,
        'label_vocabulary': sorted(data.label_dict.keys()),
        'weight_column': _WEIGHT_COLUMN,
        'class_weights': data.class_weights,
        'mode': args.mode,
        'max_sequence_len': data.max_sequence_len
    }
    if args.mode == __REGRESSION:
        del params['weight_column']
        del params['class_weights']
        del params['n_classes']
        del params['label_vocabulary']

    save_params = params
    del save_params['optimizer']
    with open(join(args.model_dir, 'params.pickle'), 'wb') as handle:
        pickle.dump(save_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del params['max_sequence_len']
    del params['class_weights']
    del params['mode']
    return __NETWORKS[_DNN][args.mode](config=config, **params)


def __load_model(model_key, model_dir, config):
    with open(join(model_dir, 'params.pickle'), 'rb') as handle:
        params = pickle.load(handle)
    weights = params['class_weights']
    mode = params['mode']
    max_sequence_len = params['max_sequence_len']
    del params['max_sequence_len']
    del params['class_weights']
    del params['mode']
    if 'sequence_classification' in params:
        sequenced_labels = not params['sequence_classification']
    else:
        sequenced_labels = False
    return __NETWORKS[model_key][mode](
        config=config,
        **params), weights, mode, sequenced_labels, max_sequence_len


def __create_rnn(data, optimizer, args, config):
    params = {
        'feature_columns': data.feature_columns,
        'hidden_units': args.layers,
        'n_classes': len(data.label_dict),
        'model_dir': args.model_dir,
        'dropout': args.dropout,
        'optimizer': optimizer,
        'label_vocabulary': sorted(data.label_dict.keys()),
        'weight_column': _WEIGHT_COLUMN,
        'class_weights': data.class_weights,
        'mode': args.mode,
        'cell_type': args.type,
        'sequence_classification': data.sequence_classification,
        'max_sequence_len': data.max_sequence_len
    }
    if args.mode == __REGRESSION:
        del params['sequence_classification']
        del params['weight_column']
        del params['n_classes']
        del params['label_vocabulary']

    save_params = params
    del save_params['optimizer']
    with open(join(args.model_dir, 'params.pickle'), 'wb') as handle:
        pickle.dump(save_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del params['max_sequence_len']
    del params['class_weights']
    del params['mode']
    return __NETWORKS[_RNN][args.mode](config=config, **params)


def __load_dnn(model_dir):
    return __load_model(_DNN, model_dir)


def __load_rnn(model_dir):
    return __load_model(_RNN, model_dir)


def train(network_key):
    assert network_key in __NETWORKS, 'No support for {}.'.format(network_key)

    basic_parser = __train_parser(network_key)
    parser = __extend_parser(basic_parser, network_key)
    args = parser.parse_args()

    assert args.mode in __NETWORKS[
        network_key], 'No support for {} in {}.'.format(
            args.mode, network_key)

    if network_key == _RNN:
        sequenced_labels = args.sequenced_labels
        sequences = True
    elif network_key == _DNN:
        sequenced_labels = True
        sequences = args.sequences
    train_data = __get_data_loader(
        filepath=args.train,
        network=network_key,
        phase=__TRAIN,
        batch_size=args.batch_size,
        mode=args.mode,
        sequenced_labels=sequenced_labels,
        sequences=sequences)
    eval_data = __get_data_loader(
        filepath=args.eval,
        network=network_key,
        batch_size=args.batch_size,
        phase=__EVAL,
        mode=args.mode,
        sequenced_labels=sequenced_labels,
        sequences=sequences,
        class_weights=train_data.class_weights,
        max_sequence_len=train_data.max_sequence_len)

    cm_hook = [
        EvalConfusionMatrixHook(
            join(args.model_dir, 'eval'),
            vocabulary=sorted(train_data.class_weights.keys()))
    ] if __TF_PLOT else None

    steps = int(train_data.steps_per_epoch * args.ne)

    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        keep_checkpoint_max=args.keep_checkpoints,
        session_config=session_config,
        save_checkpoints_secs=args.eval_period)

    optimizer = tf.train.RMSPropOptimizer(args.lr)
    makedirs(args.model_dir, exist_ok=True)
    classifier = __create_model(network_key, train_data, optimizer, args,
                                config)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_data.input_fn,
        start_delay_secs=args.eval_period,
        throttle_secs=args.eval_period,
        hooks=cm_hook)
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_data.input_fn, max_steps=steps)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


def evaluate(network_key):
    assert network_key in __NETWORKS, 'No support for {}.'.format(network_key)

    parser = __eval_parser(network_key)
    args = parser.parse_args()
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.estimator.RunConfig(
        model_dir=args.model_dir, session_config=session_config)
    trained_model, class_weights, mode, sequenced_labels, max_sequence_len = __load_model(
        network_key, args.model_dir, config)
    eval_data = __get_data_loader(
        filepath=args.eval,
        network=network_key,
        phase=__EVAL,
        mode=mode,
        batch_size=args.batch_size,
        sequenced_labels=sequenced_labels,
        class_weights=class_weights,
        max_sequence_len=max_sequence_len)
    cm_hook = [
        EvalConfusionMatrixHook(
            vocabulary=sorted(class_weights.keys()),
            output_dir=join(args.model_dir, 'eval_on {}'.format(
                basename(args.eval))))
    ] if __TF_PLOT else None
    trained_model.evaluate(
        eval_data.input_fn,
        checkpoint_path=args.ckpt,
        name='on {}'.format(basename(args.eval)),
        hooks=cm_hook)


def predict(network_key):
    assert network_key in __NETWORKS, 'No support for {}.'.format(network_key)

    parser = __predict_parser(network_key)
    args = parser.parse_args()
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.estimator.RunConfig(
        model_dir=args.model_dir, session_config=session_config)
    trained_model, class_weights, mode, sequenced_labels, max_sequence_len = __load_model(
        network_key, args.model_dir, config)
    predict_data = __get_data_loader(
        filepath=args.predict,
        network=network_key,
        phase=__EVAL,
        mode=mode,
        batch_size=args.batch_size,
        sequenced_labels=sequenced_labels,
        class_weights=class_weights,
        max_sequence_len=max_sequence_len)
    predictions = trained_model.predict(predict_data.input_fn)

    with open(args.output, 'w') as of:
        writer = csv.writer(of, delimiter=',')
        if mode == __CLASSIFICATION:
            writer.writerow(['name', 'prediction'] + [
                class_name + ' probability'
                for class_name in sorted(class_weights.keys())
            ])

        for name, prediction in zip(predict_data.names, predictions):
            if mode == __CLASSIFICATION:
                if not sequenced_labels:
                    writer.writerow([
                        name,
                        list(
                            map(lambda x: x.decode('utf-8'),
                                prediction['classes']))[0]
                    ] + [
                        probability
                        for probability in prediction['probabilities']
                    ])
                else:
                    for class_name, probabilities in zip(
                            prediction['classes'],
                            prediction['probabilities']):
                        writer.writerow([
                            name,
                            list(map(lambda x: x.decode('utf-8'), class_name))
                            [0]
                        ] + [probability for probability in probabilities])
