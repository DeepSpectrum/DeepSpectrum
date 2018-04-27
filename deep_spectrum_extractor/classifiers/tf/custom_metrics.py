import tensorflow as tf
from tensorflow.contrib.metrics import streaming_covariance, streaming_mean, streaming_pearson_correlation


def streaming_concordance_correlation_coefficient(predictions,
                                                  labels,
                                                  metrics_collections=None,
                                                  updates_collections=None,
                                                  weights=1.0,
                                                  name=None):
    with tf.variable_scope(name, 'ccc', (predictions, labels, weights)):
        predictions = tf.squeeze(predictions)
        labels = tf.squeeze(labels)
        weights = tf.squeeze(weights)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        cov, update_cov = streaming_covariance(
            predictions, labels, weights=weights, name='covariance')
        var_predictions, update_var_predictions = streaming_covariance(
            predictions,
            predictions,
            weights=weights,
            name='variance_predictions')
        var_labels, update_var_labels = streaming_covariance(
            labels, labels, weights=weights, name='variance_labels')
        mean_predictions, update_mean_predictions = streaming_mean(
            predictions, weights=weights, name='mean_predictions')
        mean_labels, update_mean_labels = streaming_mean(
            labels, weights=weights, name='mean_labels')

        ccc = tf.truediv(
            tf.multiply(tf.constant(2, dtype=tf.float32), cov),
            tf.add_n((var_predictions, var_labels,
                      tf.square(tf.subtract(mean_predictions, mean_labels)))),
            name='ccc')

        update_op = tf.truediv(
            tf.multiply(tf.constant(2, dtype=tf.float32), update_cov),
            tf.add_n((update_var_predictions, update_var_labels,
                      tf.square(
                          tf.subtract(update_mean_predictions,
                                      update_mean_labels)))),
            name='update_op')

        if metrics_collections:
            for mc in metrics_collections:
                tf.add_to_collection(mc, ccc)

        if updates_collections:
            for uc in updates_collections:
                tf.add_to_collection(uc, update_op)

        return {'ccc': (ccc, update_op)}


def pearson_r(predictions,
              labels,
              metrics_collection=None,
              updates_collections=None,
              weights=1.0):
    return {
        'pearson_r':
        streaming_pearson_correlation(
            predictions,
            labels,
            weights=weights,
            metrics_collections=metrics_collection,
            updates_collections=updates_collections)
    }


def confusion_matrix(labels, predictions, num_classes, label_vocabulary):
    with tf.name_scope('confusion_matrix', (predictions, labels, num_classes)):
        mapping_strings = tf.constant(label_vocabulary)
        table = tf.contrib.lookup.index_table_from_tensor(
            mapping=mapping_strings, num_oov_buckets=1, default_value=-1)
        labels = table.lookup(labels)
        labels = tf.reshape(labels, [-1])
        predictions = tf.reshape(predictions, [-1])
        con_matrix = tf.confusion_matrix(
            labels=labels, predictions=predictions, num_classes=num_classes)

        con_matrix_sum = tf.Variable(
            tf.zeros(shape=(num_classes, num_classes), dtype=tf.int32),
            trainable=False,
            name="sum",
            collections=[tf.GraphKeys.LOCAL_VARIABLES])
        update_op = tf.assign_add(con_matrix_sum, con_matrix, name='update')
        con_matrix_tensor = tf.convert_to_tensor(con_matrix_sum)
        return {'confusion_matrix': (con_matrix_tensor, update_op)}


def uar(labels, predictions, num_classes, label_vocabulary):
    mapping_strings = tf.constant(label_vocabulary)
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings, num_oov_buckets=1, default_value=-1)
    ids = table.lookup(labels)
    return {
        'uar': tf.metrics.mean_per_class_accuracy(ids, predictions,
                                                  num_classes)
    }


def ccc_loss(labels, logits):
    """Adds a concordance correlation loss to training.
  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  `[batch_size]`, then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.
  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.
  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or `predictions`
      is None.
  """
    if labels is None:
        raise ValueError("labels must not be None.")
    if logits is None:
        raise ValueError("logits must not be None.")
    with tf.name_scope("ccc", (logits, labels)) as scope:
        predictions = tf.to_float(logits)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        ones = tf.ones_like(predictions)
        losses = tf.subtract(ones,
                             streaming_concordance_correlation_coefficient(
                                 predictions, labels)[0])
        return losses
