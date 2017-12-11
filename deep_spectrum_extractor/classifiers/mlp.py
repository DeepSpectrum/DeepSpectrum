import os
import argparse
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.INFO)

# class EvalCheckpointSaverListener(tf.train.CheckpointSaverListener):
#     def __init__(self, estimator, input_fn):
#         self.estimator = estimator
#         self.input_fn = input_fn

#     def after_save(self, session, global_step):
#         metrics = self.estimator.evaluate(self.input_fn)
#         print('Step {}: Accuracy: {} UAR: {}'.format(
#             metrics['global_step'], metrics['accuracy'], metrics['uar']))


def uar(labels, predictions, num_classes=6):
    return {
        'uar':
        tf.metrics.mean_per_class_accuracy(
            labels,
            tf.reshape(predictions['class_ids'], [-1]),
            num_classes=num_classes)
    }


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
        'Train a mlp classifier on Deep Spectrum features in tfrecord format.')
    parser.add_argument(
        '-train',
        required=True,
        default=None,
        nargs='+',
        help='tfrecord files used for training the classifier.')
    parser.add_argument(
        '-eval',
        required=True,
        default=None,
        nargs='+',
        help='tfrecord files used for classifier validation during training.')
    parser.add_argument(
        '-classes',
        required=True,
        type=int,
        default=None,
        help='Number of classes for the classification problem.')
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
        '--layers', default=2, type=int, help='Number of hidden layers.')
    parser.add_argument(
        '--num_features',
        default=4096,
        type=int,
        help='Number of numerical input features in tfrecords files.')
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
    args = parser.parse_args()
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        keep_checkpoint_max=args.keep_checkpoints,
        session_config=session_config,
        save_checkpoints_secs=args.eval_period)
    feature_columns = [
        tf.feature_column.numeric_column("x", shape=[args.num_features])
    ]
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[2**(7 + i) for i in range(args.layers)][::-1],
        n_classes=args.classes,
        model_dir=args.model_dir,
        dropout=0.2,
        config=config)

    classifier = tf.contrib.estimator.add_metrics(
        classifier,
        lambda labels, predictions: uar(labels, predictions, num_classes=args.classes))
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(args.eval, shuffle=False, num_epochs=1),
        start_delay_secs=args.eval_period,
        throttle_secs=args.eval_period)
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(args.train, shuffle=True),
        max_steps=args.max_steps)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


if __name__ == '__main__':
    tf.app.run()
