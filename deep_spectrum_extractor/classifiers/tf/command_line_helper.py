import argparse
import tensorflow as tf
import pickle
import csv
from os.path import join, basename
from os import makedirs

__CM_PLOT = True
try:
    from .hooks import EvalConfusionMatrixHook
except ImportError:
    __CM_PLOT = False

# Modes (Classification and Regression)
__CLASSIFICATION = 'classification'
__REGRESSION = 'regression'
__MODE_KEYS = [__CLASSIFICATION, __REGRESSION]

# Available network architectures:
_RNN = 'Recurrent Neural Network'
_DNN = 'Deep Neural Network'

# Phases
__TRAIN = 'train'
__EVAL = 'eval'
__PREDICT = 'predict'
__PHASE_KEYS = [__TRAIN, __EVAL, __PREDICT]


def basic_parser(net_name='classifier'):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(
        help='Either train or evaluate a {} or make predictions.'.format(
            net_name))
    train_subparser = __basic_train_subparser(subparsers, net_name)
    eval_subparser = __basic_eval_subparser(subparsers, net_name)
    predict_subparser = __basic_predict_subparser(subparsers, net_name)
    return parser, train_subparser, eval_subparser, predict_subparser

def __basic_train_subparser(subparsers, net_name='classifier'):
    train_parser = subparsers.add_parser(
        __TRAIN,
        help='Train and evaluate a {} on given data.'.format(net_name),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_parser.add_argument(
        'training_data',
        default=None,
        help='file used for training the model.')
    train_parser.add_argument(
        'evaluation_data',
        default=None,
        help='file used for validation during training.')
    train_parser.add_argument(
        '-ne',
        '--number_of_epochs',
        default=10,
        type=int,
        help='Number of epochs to train the model.')
    train_parser.add_argument(
        '-ep',
        '--eval_period',
        default=5,
        type=int,
        help='Evaluation interval in seconds.')
    train_parser.add_argument(
        '-kc',
        '--keep_checkpoints',
        default=5,
        type=int,
        help='How many checkpoints to keep stored on disk.')
    train_parser.add_argument(
        '-lr',
        '--learning_rate',
        default=0.001,
        type=float,
        help='Learning rate.')
    train_parser.add_argument(
        '-d', '--dropout', default=0.2, type=float, help='Dropout')
    train_parser.add_argument(
        '-m',
        '--mode',
        default=__CLASSIFICATION,
        help='Perform regression or classification task.',
        choices=__MODE_KEYS)
    train_parser.add_argument(
        '-bs', '--batch_size', default=32, type=int, help='Batchsize')
    train_parser.add_argument(
        '-md',
        '--model_dir',
        required=True,
        help=
        'Directory for saving and restoring model checkpoints, summaries and exports.'
    )
    return train_parser


def __basic_eval_subparser(subparsers, net_name='classifier'):
    eval_parser = subparsers.add_parser(
        __EVAL,
        help='Evaluate a {} on numerical features in arff or csv format.'.
        format(net_name),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    eval_parser.add_argument(
        'evaluation_data', help='Features to evaluate model on.')
    eval_parser.add_argument(
        '-cp',
        '--checkpoint',
        default=None,
        help='Specific checkpoint to evaluate.')
    eval_parser.add_argument(
        '-bs', '--batch_size', default=32, type=int, help='Batchsize')
    eval_parser.add_argument(
        '-md',
        '--model_dir',
        required=True,
        help=
        'Directory for saving and restoring model checkpoints, summaries and exports.'
    )
    return eval_parser


def __basic_predict_subparser(subparsers, net_name='classifier'):
    predict_parser = subparsers.add_parser(
        __PREDICT,
        help=
        'Make predictions with a {} on numerical features in arff or csv format.'.
        format(net_name),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    predict_parser.add_argument(
        'prediction_data', help='Features to make predictions for.')
    predict_parser.add_argument(
        '-cp',
        '--checkpoint',
        default=None,
        help='Specific checkpoint to evaluate.')
    predict_parser.add_argument(
        '-o',
        '--output',
        default='predictions.csv',
        help='Path to output csv.')
    predict_parser.add_argument(
        '-bs', '--batch_size', default=32, type=int, help='Batchsize')
    predict_parser.add_argument(
        '-md',
        '--model_dir',
        required=True,
        help=
        'Directory for saving and restoring model checkpoints, summaries and exports. If possible, continues training from checkpoint in this directory.'
    )
    return predict_parser


def save_params(loader_params, model_params, model_dir, mode=__CLASSIFICATION):
    params = {'mode': mode, 'model': model_params, 'loader': loader_params}
    makedirs(model_dir, exist_ok=True)
    with open(join(model_dir, 'params.pickle'), 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_params(model_dir):
    with open(join(model_dir, 'params.pickle'), 'rb') as handle:
        params = pickle.load(handle)
    return params['loader'], params['model'], params['mode']


def basic_train(model,
                loader_train,
                loader_eval,
                model_dir,
                number_of_epochs,
                keep_checkpoints,
                eval_period,
                mode=__CLASSIFICATION):
    cm_hook = None
    if mode == __CLASSIFICATION:
        if __CM_PLOT:
            cm_hook = [
                EvalConfusionMatrixHook(
                    join(model_dir, 'eval'),
                    vocabulary=sorted(loader_train.class_weights.keys()))
            ]
    steps = int(loader_train.steps_per_epoch * number_of_epochs)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=loader_eval.input_fn,
        start_delay_secs=eval_period,
        throttle_secs=eval_period,
        hooks=cm_hook)
    train_spec = tf.estimator.TrainSpec(
        input_fn=loader_train.input_fn, max_steps=steps)

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


def write_results(metrics, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = sorted(metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics)


def basic_eval(model,
               loader_eval,
               model_dir,
               checkpoint_path=None,
               evaluation_key='manual',
               mode=__CLASSIFICATION):
    cm_hook = None
    if mode == __CLASSIFICATION:
        if __CM_PLOT:
            cm_hook = [
                EvalConfusionMatrixHook(
                    join(model_dir, 'eval'),
                    vocabulary=sorted(loader_eval.class_weights.keys()))
            ]
    model.evaluate(
        loader_eval.input_fn,
        checkpoint_path=checkpoint_path,
        name='{}'.format(evaluation_key),
        hooks=cm_hook)


def basic_predict(model,
                  loader_predict,
                  model_dir,
                  output,
                  checkpoint=None,
                  mode=__CLASSIFICATION):
    predictions = model.predict(loader_predict.input_fn)

    with open(output, 'w') as of:
        writer = csv.writer(of, delimiter=',')
        if mode == __CLASSIFICATION:
            writer.writerow(['name', 'prediction'] + [
                class_name + ' probability'
                for class_name in sorted(loader_predict.class_weights.keys())
            ])

        for name, prediction in zip(loader_predict.names, predictions):
            if mode == __CLASSIFICATION:
                if loader_predict.sequence_classification:
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


def config(model_dir, keep_checkpoints, steps_per_epoch):
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.estimator.RunConfig(
        model_dir=model_dir,
        keep_checkpoint_max=keep_checkpoints,
        session_config=session_config,
        save_checkpoints_steps=steps_per_epoch,
        tf_random_seed=42)
    return config
