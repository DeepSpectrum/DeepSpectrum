from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import six
import argparse
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.estimator import model_fn
from ..data_loader import DataLoader
from .. import head as head_lib

_LEARNING_RATE = 0.05



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
                          return_sequences=False):
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

            # if dropout is not None and mode == model_fn.ModeKeys.TRAIN:
            #     rnn_layers = [
            #         tf.nn.rnn_cell.DropoutWrapper(
            #             cell, output_keep_prob=1 - dropout)
            #         for cell in rnn_layers
            #     ]
            # create a RNN cell composed sequentially of a number of RNNCells
            multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
            if dropout is not None and mode == model_fn.ModeKeys.TRAIN:
                multi_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(multi_rnn_cell, output_keep_prob=1-dropout, state_keep_prob=1-dropout, input_keep_prob=1-dropout)
            # 'state' is a N-tuple where N is the number of LSTMCells containing a
            # tf.contrib.rnn.LSTMStateTuple for each cell
            output, states = tf.nn.dynamic_rnn(
                cell=multi_rnn_cell,
                inputs=net,
                dtype=tf.float32,
                sequence_length=_length(net))
            if return_sequences:
                net = output
            else:
                if cell_type == 'lstm':
                    hidden_states = [state.c for state in states]
                elif cell_type == 'gru':
                    hidden_states = states
                net = tf.concat(hidden_states, name='final_h_states', axis=-1)

            net = tf.layers.batch_normalization(
                net,
                # The default momentum 0.99 actually crashes on certain
                # problem, so here we use 0.999, which is the default of
                # tf.contrib.layers.batch_norm.
                momentum=0.999,
                training=(mode == model_fn.ModeKeys.TRAIN),
                name='batchnorm')
        with tf.variable_scope('logits', values=(net, )) as logits_scope:
            logits = tf.layers.dense(
                net,
                units=units,
                activation=None,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name=logits_scope)

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
                  return_sequences=False,
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
            cell_type=cell_type,
            return_sequences=return_sequences)
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
                 return_sequences=False,
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
                return_sequences=return_sequences)

        super(RNNClassifier, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config)

class RNNRegressor(tf.estimator.Estimator):
    """A regressor for TensorFlow RNN models.

  Example:

  ```python
  categorical_feature_a = categorical_column_with_hash_bucket(...)
  categorical_feature_b = categorical_column_with_hash_bucket(...)

  categorical_feature_a_emb = embedding_column(
      categorical_column=categorical_feature_a, ...)
  categorical_feature_b_emb = embedding_column(
      categorical_column=categorical_feature_b, ...)

  estimator = DNNRegressor(
      feature_columns=[categorical_feature_a_emb, categorical_feature_b_emb],
      hidden_units=[1024, 512, 256])

  # Or estimator using the ProximalAdagradOptimizer optimizer with
  # regularization.
  estimator = DNNRegressor(
      feature_columns=[categorical_feature_a_emb, categorical_feature_b_emb],
      hidden_units=[1024, 512, 256],
      optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Or estimator with warm-starting from a previous checkpoint.
  estimator = DNNRegressor(
      feature_columns=[categorical_feature_a_emb, categorical_feature_b_emb],
      hidden_units=[1024, 512, 256],
      warm_start_from="/path/to/checkpoint/dir")

  # Input builders
  def input_fn_train: # returns x, y
    pass
  estimator.train(input_fn=input_fn_train, steps=100)

  def input_fn_eval: # returns x, y
    pass
  metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
  def input_fn_predict: # returns x, None
    pass
  predictions = estimator.predict(input_fn=input_fn_predict)
  ```

  Input of `train` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

  * if `weight_column` is not `None`, a feature with
    `key=weight_column` whose value is a `Tensor`.
  * for each `column` in `feature_columns`:
    - if `column` is a `_CategoricalColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `_WeightedCategoricalColumn`, two features: the first
      with `key` the id column name, the second with `key` the weight column
      name. Both features' `value` must be a `SparseTensor`.
    - if `column` is a `_DenseColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.

  Loss is calculated by using mean squared error.

  @compatibility(eager)
  Estimators are not compatible with eager execution.
  @end_compatibility
  """

    def __init__(
            self,
            hidden_units,
            feature_columns,
            cell_type='lstm',
            return_sequences=False,
            model_dir=None,
            label_dimension=1,
            weight_column=None,
            optimizer='Adagrad',
            activation_fn=tf.nn.relu,
            dropout=None,
            input_layer_partitioner=None,
            config=None,
            loss_reduction=tf.losses.Reduction.SUM,
    ):
        """Initializes a `DNNRegressor` instance.

    Args:
      hidden_units: Iterable of number hidden units per layer. All layers are
        fully connected. Ex. `[64, 32]` means first layer has 64 nodes and
        second one has 32.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `_FeatureColumn`.
      cell_type: Type of rnn cell to use. Either `lstm` or `gru`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      label_dimension: Number of regression targets per example. This is the
        size of the last dimension of the labels and logits `Tensor` objects
        (typically, these have shape `[batch_size, label_dimension]`).
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
        then weight_column.normalizer_fn is applied on it to get weight tensor.
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

        def _model_fn(features, labels, mode, config):
            """Call the defined shared _dnn_model_fn."""
            return _rnn_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head_lib.  # pylint: disable=protected-access
                _regression_head_with_mean_squared_error_loss(
                    label_dimension=label_dimension,
                    weight_column=weight_column,
                    loss_reduction=loss_reduction),
                hidden_units=hidden_units,
                feature_columns=tuple(feature_columns or []),
                optimizer=optimizer,
                activation_fn=activation_fn,
                dropout=dropout,
                input_layer_partitioner=input_layer_partitioner,
                config=config,
                return_sequences=return_sequences,
                cell_type=cell_type)

        super(RNNRegressor, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config)
