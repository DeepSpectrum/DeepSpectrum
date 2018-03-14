import os
import argparse
import tensorflow as tf
from .data_loader import DataLoader
from .dnn import DNNClassifier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.INFO)


def input_fn(filenames, shuffle=True, num_epochs=None, batch_size=32):
    dataset = tf.data.TFRecordDataset(filenames)

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(example_proto):
        features = {
            "features": tf.FixedLenFeature((), tf.string),
            "label": tf.FixedLenFeature((), tf.int64)
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        return {
            'x': tf.decode_raw(parsed_features["features"], tf.float32)
        }, tf.cast(parsed_features["label"], tf.int64)

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()
    return features, labels





def main(sysargs):
    parser = argparse.ArgumentParser(
        description=
        'Train a mlp classifier on Deep Spectrum features in arff or csv format.')
    parser.add_argument(
        '-train',
        required=True,
        default=None,
        help='file used for training the classifier.')
    parser.add_argument(
        '-eval',
        required=True,
        default=None,
        help='file used for classifier validation during training.')
    parser.add_argument(
        '--batch_size', default=32, type=int, help='Batchsize for training.')
    parser.add_argument(
        '--max_steps',
        default=10000,
        type=int,
        help='Number of epochs to train the model.')
    parser.add_argument(
        '--model_dir',
        default=None,
        help=
        'Directory for saving and restoring model checkpoints, summaries and exports.'
    )
    parser.add_argument(
        '--layers', default=[500, 100], nargs='+', type=int, help='Shapes of hidden layers.')
    parser.add_argument(
        '--eval_period',
        default=10,
        type=int,
        help='Evaluation interval in seconds.')
    parser.add_argument(
        '--keep_checkpoints',
        default=20,
        type=int,
        help='How many checkpoints to keep stored on disk.')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate.')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout')
    args = parser.parse_args()
    train_data = DataLoader(args.train, sequences=True, batch_size=args.batch_size, shuffle=True, num_epochs=1, num_threads=1, queue_capacity=10000)
    eval_data = DataLoader(args.eval, sequences=True, batch_size=args.batch_size, shuffle=False, num_epochs=1, num_threads=1)

    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        keep_checkpoint_max=args.keep_checkpoints,
        session_config=session_config,
        save_checkpoints_secs=args.eval_period)
    optimizer = tf.train.AdagradOptimizer(learning_rate=args.lr)
    classifier = DNNClassifier(
        feature_columns=train_data.feature_columns,
        hidden_units=args.layers,
        n_classes=len(train_data.label_dict),
        model_dir=args.model_dir,
        dropout=args.dropout,
        config=config, optimizer=optimizer, label_vocabulary=sorted(train_data.label_dict.keys()), weight_column=train_data.weight_column)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_data.input_fn,
        start_delay_secs=args.eval_period,
        throttle_secs=args.eval_period)
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_data.input_fn,
        max_steps=args.max_steps)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


if __name__ == '__main__':
    tf.app.run()
