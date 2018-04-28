import tensorflow as tf
from tensorflow.contrib.metrics import streaming_covariance, streaming_mean


def streaming_concordance_correlation_coefficient(predictions,
                                                  labels,
                                                  metrics_collections=None,
                                                  updates_collections=None,
                                                  weights=None,
                                                  name=None):
    with tf.variable_scope(name, 'CCC',
                                          (predictions, labels, weights)):
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
            tf.add_n((tf.square(var_predictions), tf.square(var_labels),
                      tf.square(tf.subtract(mean_predictions, mean_labels)))),
            name='ccc')

        update_op = tf.truediv(
            tf.multiply(tf.constant(2, dtype=tf.float32), update_cov),
            tf.add_n((tf.square(update_var_predictions),
                      tf.square(update_var_labels),
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

        return ccc, update_op
