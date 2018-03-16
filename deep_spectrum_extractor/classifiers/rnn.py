from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import six
import argparse
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.estimator import model_fn
from .data_loader import DataLoader
from . import head as head_lib

_LEARNING_RATE = 0.05


def _add_hidden_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                      tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)


def _length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def _extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res


def _rnn_logit_fn_builder(units,
                          hidden_units,
                          feature_columns,
                          activation_fn,
                          dropout,
                          input_layer_partitioner,
                          cell_type='lstm',
                          sequence_classification=True):
    """Function builder for a dnn logit_fn.

  Args:
    units: An int indicating the dimension of the logit layer.  In the
      MultiHead case, this should be the sum of all component Heads' logit
      dimensions.
    hidden_units: Iterable of integer number of hidden units per layer.
    feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
    activation_fn: Activation function applied to each layer.
    dropout: When not `None`, the probability we will drop out a given
      coordinate.
    input_layer_partitioner: Partitioner for input layer.
    cell_type: type of rnn cell to use in model
  Returns:
    A logit_fn (see below).

  Raises:
    ValueError: If units is not an int.
  """
    if not isinstance(units, int):
        raise ValueError('units must be an int.  Given type: {}'.format(
            type(units)))

    def rnn_logit_fn(features, mode):
        """Deep Neural Network logit_fn.

    Args:
      features: This is the first item returned from the `input_fn`
                passed to `train`, `evaluate`, and `predict`. This should be a
                single `Tensor` or `dict` of same.
      mode: Optional. Specifies if this training, evaluation or prediction. See
            `ModeKeys`.

    Returns:
      A `Tensor` representing the logits, or a list of `Tensor`'s representing
      multiple logits in the MultiHead case.
    """
        with tf.variable_scope(
                'input_from_feature_columns',
                values=tuple(six.itervalues(features)),
                partitioner=input_layer_partitioner):
            net = features['features']
        with tf.variable_scope(
                'dynamic_rnn', values=(net, )) as hidden_layer_scope:
            if cell_type == 'lstm':
                rnn_cell = tf.nn.rnn_cell.LSTMCell
            elif cell_type == 'gru':
                rnn_cell = tf.nn.rnn_cell.GRUCell
            else:
                raise ValueError('cell_type must be either "lstm" or "gru"')

            rnn_layers = [
                rnn_cell(
                    size, name=cell_type + '_%d' % i, activation=activation_fn)
                for i, size in enumerate(hidden_units)
            ]

            if dropout is not None and mode == model_fn.ModeKeys.TRAIN:
                rnn_layers = [
                    tf.nn.rnn_cell.DropoutWrapper(
                        cell, output_keep_prob=1 - dropout)
                    for cell in rnn_layers
                ]
            # create a RNN cell composed sequentially of a number of RNNCells
            multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

            # 'outputs' is a tensor of shape [batch_size, max_time, 256]
            # 'state' is a N-tuple where N is the number of LSTMCells containing a
            # tf.contrib.rnn.LSTMStateTuple for each cell
            output, state = tf.nn.dynamic_rnn(
                cell=multi_rnn_cell,
                inputs=net,
                dtype=tf.float32,
                sequence_length=_length(net))
            if sequence_classification:
                if cell_type == 'lstm':
                    net = state[len(hidden_units) - 1][1]
                elif cell_type == 'gru':
                    net = state[len(hidden_units) - 1]
            else:
                net = output
            _add_hidden_layer_summary(net, hidden_layer_scope.name)
        with tf.variable_scope('logits', values=(net, )) as logits_scope:
            logits = tf.layers.dense(
                net,
                units=units,
                activation=None,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name=logits_scope)
        _add_hidden_layer_summary(logits, logits_scope.name)

        return logits

    return rnn_logit_fn


def _rnn_model_fn(features,
                  labels,
                  mode,
                  head,
                  hidden_units,
                  feature_columns,
                  optimizer='SGD',
                  activation_fn=tf.nn.relu,
                  dropout=None,
                  input_layer_partitioner=None,
                  config=None,
                  sequence_classification=True,
                  cell_type='lstm'):
    """Deep Neural Net model_fn.

  Args:
    features: dict of `Tensor`.
    labels: `Tensor` of shape [batch_size, max_time, 1] or [batch_size] labels of
      dtype `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    head: A `head_lib._Head` instance.
    hidden_units: Iterable of integer number of hidden units per layer.
    feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
    optimizer: String, `tf.Optimizer` object, or callable that creates the
      optimizer to use for training. If not specified, will use the Adagrad
      optimizer with a default learning rate of 0.05.
    activation_fn: Activation function applied to each layer.
    dropout: When not `None`, the probability we will drop out a given
      coordinate.
    input_layer_partitioner: Partitioner for input layer. Defaults
      to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
    config: `RunConfig` object to configure the runtime settings.
    sequence_classification: Boolean indicating whether to perform classification
      of whole sequences or every sequence element individually.
    cell_type: String indicating the type of rnn cell to use in the model. 
  Returns:
    An `EstimatorSpec` instance.

  Raises:
    ValueError: If features has the wrong type.
  """
    if not isinstance(features, dict):
        raise ValueError('features should be a dictionary of `Tensor`s. '
                         'Given type: {}'.format(type(features)))
    optimizer = optimizers.get_optimizer_instance(
        optimizer, learning_rate=_LEARNING_RATE)
    num_ps_replicas = config.num_ps_replicas if config else 0
    partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas)
    with tf.variable_scope(
            'rnn', values=tuple(six.itervalues(features)),
            partitioner=partitioner):
        input_layer_partitioner = input_layer_partitioner or (
            partitioned_variables.min_max_variable_partitioner(
                max_partitions=num_ps_replicas, min_slice_size=64 << 20))

        logit_fn = _rnn_logit_fn_builder(
            units=head.logits_dimension,
            hidden_units=hidden_units,
            feature_columns=feature_columns,
            activation_fn=activation_fn,
            dropout=dropout,
            input_layer_partitioner=input_layer_partitioner,
            cell_type=cell_type)
        logits = logit_fn(features=features, mode=mode)

        def _train_op_fn(loss):
            """Return the loss minimization op."""
            return optimizer.minimize(loss, tf.train.get_global_step())

        return head.create_estimator_spec(
            features=features,
            mode=mode,
            labels=labels,
            train_op_fn=_train_op_fn,
            logits=logits)


class RNNClassifier(tf.estimator.Estimator):
    """A classifier for TensorFlow RNN models."""

    def __init__(self,
                 hidden_units,
                 feature_columns,
                 cell_type='lstm',
                 sequence_classification=True,
                 model_dir=None,
                 n_classes=2,
                 weight_column=None,
                 label_vocabulary=None,
                 optimizer='Adagrad',
                 activation_fn=tf.tanh,
                 dropout=None,
                 input_layer_partitioner=None,
                 config=None,
                 loss_reduction=tf.losses.Reduction.SUM):
        """Initializes a `RNNClassifier` instance.

    Args:
      hidden_units: Iterable of number hidden units per recurrent layer.
        Ex. `[64, 32]` means first layer has 64 nodes and second one has 32.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `_FeatureColumn`.
      cell_type: String specifying the type of recurrent cell to use in the model.
        Available types are "lstm" and "gru".
      sequence_classification: Boolean indicating whether to perform
        classification of whole sequences (True) or individual elements of the
        sequences (False).
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      n_classes: Number of label classes. Defaults to 2, namely binary
        classification. Must be > 1.
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
        then weight_column.normalizer_fn is applied on it to get weight tensor.
      label_vocabulary: A list of strings represents possible label values. If
        given, labels must be string type and have any value in
        `label_vocabulary`. If it is not given, that means labels are
        already encoded as integer or float within [0, 1] for `n_classes=2` and
        encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 .
        Also there will be errors if vocabulary is not provided and labels are
        string.
      optimizer: An instance of `tf.Optimizer` used to train the model. Defaults
        to Adagrad optimizer.
      activation_fn: Activation function applied to each layer. If `None`, will
        use `tf.nn.relu`.
      dropout: When not `None`, the probability we will drop out a given
        coordinate.
      input_layer_partitioner: Optional. Partitioner for input layer. Defaults
        to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
      config: `RunConfig` object to configure the runtime settings.
      warm_start_from: A string filepath to a checkpoint to warm-start from, or
        a `WarmStartSettings` object to fully configure warm-starting.  If the
        string filepath is provided instead of a `WarmStartSettings`, then all
        weights are warm-started, and it is assumed that vocabularies and Tensor
        names are unchanged.
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how
        to reduce training loss over batch. Defaults to `SUM`.
    """
        if n_classes == 2:
            head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
                weight_column=weight_column,
                label_vocabulary=label_vocabulary,
                loss_reduction=loss_reduction,
                model_dir=model_dir)
        else:
            head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(  # pylint: disable=protected-access
                n_classes,
                weight_column=weight_column,
                label_vocabulary=label_vocabulary,
                loss_reduction=loss_reduction,
                model_dir=model_dir)

        def _model_fn(features, labels, mode, config):
            """Call the defined shared _dnn_model_fn."""
            return _rnn_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head,
                hidden_units=hidden_units,
                feature_columns=tuple(feature_columns or []),
                optimizer=optimizer,
                activation_fn=activation_fn,
                dropout=dropout,
                input_layer_partitioner=input_layer_partitioner,
                config=config,
                cell_type=cell_type,
                sequence_classification=sequence_classification)

        super(RNNClassifier, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config)


def main(sysargs):
    parser = argparse.ArgumentParser(
        description=
        'Train a RNN classifier on Deep Spectrum features in arff or csv format.'
    )
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
        default='model',
        help=
        'Directory for saving and restoring model checkpoints, summaries and exports.'
    )
    parser.add_argument(
        '--layers',
        default=[100],
        nargs='+',
        type=int,
        help='Shapes of hidden layers.')
    parser.add_argument(
        '--eval_period',
        default=10,
        type=int,
        help='Evaluation interval in seconds.')
    parser.add_argument(
        '--keep_checkpoints',
        default=5,
        type=int,
        help='How many checkpoints to keep stored on disk.')
    parser.add_argument(
        '--lr', default=0.001, type=float, help='Learning rate.')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout')
    parser.add_argument(
        '--type',
        default='lstm',
        choices=['lstm', 'gru'],
        help='Type of RNN cell to use in model.')
    args = parser.parse_args()
    train_data = DataLoader(
        args.train,
        sequences=True,
        batch_size=args.batch_size,
        shuffle=True,
        num_epochs=None,
        num_threads=1,
        queue_capacity=10000,
        sequence_classification=True)
    eval_data = DataLoader(
        args.eval,
        sequences=True,
        batch_size=args.batch_size,
        shuffle=False,
        num_epochs=1,
        num_threads=1,
        sequence_classification=True)
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        keep_checkpoint_max=args.keep_checkpoints,
        session_config=session_config)
    optimizer = tf.train.RMSPropOptimizer(args.lr)
    classifier = RNNClassifier(
        feature_columns=train_data.feature_columns,
        hidden_units=args.layers,
        n_classes=len(train_data.label_dict),
        model_dir=config.model_dir,
        dropout=args.dropout,
        config=config,
        optimizer=optimizer,
        label_vocabulary=sorted(train_data.label_dict.keys()),
        weight_column=train_data.weight_column)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_data.input_fn,
        start_delay_secs=args.eval_period,
        throttle_secs=args.eval_period)
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_data.input_fn, max_steps=args.max_steps)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


if __name__ == '__main__':
    tf.app.run()
