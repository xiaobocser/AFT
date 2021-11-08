import tensorflow as tf
# coding=utf-8
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops


def _cal_top(scores):
    """
    Calculate topK recommends precision
    """
    hit_5 = hit_10 = hit_20 = hit_50 = 0
    num_5 = num_10 = num_20 = num_50 = 0
    for i, s in enumerate(scores[:50]):
        if i < 5:
            hit_5 += s[1]
            num_5 += 1
        if i < 10:
            if len(scores) > 5:
                hit_10 += s[1]
            num_10 += 1
        if i < 20:
            if len(scores) > 10:
                hit_20 += s[1]
            num_20 += 1
        if i < 50:
            if len(scores) > 20:
                hit_50 += s[1]
            num_50 += 1
    return hit_5 / float(num_5), hit_10 / float(num_10), hit_20 / float(num_20), hit_50 / float(num_50)


def topK(labels, predict):
    """
    Online data metrics
    """
    scores = sorted(zip(predict, labels), key=lambda x: x[0], reverse=True)[:50]
    top5, top10, top20, top50 = _cal_top(scores)
    return top5, top10, top20, top50


def _remove_squeezable_dimensions(predictions, labels, weights):
    predictions = ops.convert_to_tensor(predictions)
    if labels is not None:
        labels, predictions = confusion_matrix.remove_squeezable_dimensions(labels, predictions)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    if weights is None:
        return predictions, labels, None

    weights = ops.convert_to_tensor(weights)
    weights_shape = weights.get_shape()
    weights_rank = weights_shape.ndims
    if weights_rank == 0:
        return predictions, labels, weights

    predictions_shape = predictions.get_shape()
    predictions_rank = predictions_shape.ndims
    if (predictions_rank is not None) and (weights_rank is not None):
        # Use static rank.
        if weights_rank - predictions_rank == 1:
            weights = array_ops.squeeze(weights, [-1])
        elif predictions_rank - weights_rank == 1:
            weights = array_ops.expand_dims(weights, [-1])
    else:
        # Use dynamic rank.
        weights_rank_tensor = array_ops.rank(weights)
        rank_diff = weights_rank_tensor - array_ops.rank(predictions)

        def _maybe_expand_weights():
            return control_flow_ops.cond(math_ops.equal(rank_diff, -1),
                                         lambda: array_ops.expand_dims(weights, [-1]), lambda: weights)

        # Don't attempt squeeze if it will fail based on static check.
        if (weights_rank is not None) and (not weights_shape.dims[-1].is_compatible_with(1)):
            maybe_squeeze_weights = lambda: weights
        else:
            maybe_squeeze_weights = lambda: array_ops.squeeze(weights, [-1])

        def _maybe_adjust_weights():
            return control_flow_ops.cond(math_ops.equal(rank_diff, 1), maybe_squeeze_weights, _maybe_expand_weights)

        # If weights are scalar, do nothing. Otherwise, try to add or remove a
        # dimension to match predictions.
        weights = control_flow_ops.cond(math_ops.equal(weights_rank_tensor, 0), lambda: weights, _maybe_adjust_weights)
    return predictions, labels, weights


def _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights=None):
    with ops.control_dependencies(
            [check_ops.assert_greater_equal(predictions, math_ops.cast(0.0, dtype=predictions.dtype),
                                            message='predictions must be in [0, 1]'),
             check_ops.assert_less_equal(predictions, math_ops.cast(1.0, dtype=predictions.dtype),
                                         message='predictions must be in [0, 1]')]):
        predictions, labels, weights = _remove_squeezable_dimensions(
            predictions=math_ops.to_float(predictions),
            labels=math_ops.cast(labels, dtype=dtypes.bool),
            weights=weights)

    num_thresholds = len(thresholds)

    # Reshape predictions and labels.
    predictions_2d = array_ops.reshape(predictions, [-1, 1])
    labels_2d = array_ops.reshape(math_ops.cast(labels, dtype=dtypes.bool), [1, -1])

    # Use static shape if known.
    num_predictions = predictions_2d.get_shape().as_list()[0]

    # Otherwise use dynamic shape.
    if num_predictions is None:
        num_predictions = array_ops.shape(predictions_2d)[0]
    thresh_tiled = array_ops.tile(
        array_ops.expand_dims(array_ops.constant(thresholds), [1]),
        array_ops.stack([1, num_predictions]))

    # Tile the predictions after threshold them across different thresholds.
    pred_is_pos = math_ops.greater(
        array_ops.tile(array_ops.transpose(predictions_2d), [num_thresholds, 1]), thresh_tiled)
    pred_is_neg = math_ops.logical_not(pred_is_pos)
    label_is_pos = array_ops.tile(labels_2d, [num_thresholds, 1])
    label_is_neg = math_ops.logical_not(label_is_pos)

    if weights is not None:
        weights = weights_broadcast_ops.broadcast_weights(
            math_ops.to_float(weights), predictions)
        weights_tiled = array_ops.tile(
            array_ops.reshape(weights, [1, -1]), [num_thresholds, 1])
        thresh_tiled.get_shape().assert_is_compatible_with(
            weights_tiled.get_shape())
    else:
        weights_tiled = None

    values = {}

    # tp
    is_true_positive = math_ops.to_float(math_ops.logical_and(label_is_pos, pred_is_pos))
    if weights_tiled is not None:
        is_true_positive *= weights_tiled
    values['tp'] = math_ops.reduce_sum(is_true_positive, 1)

    # fn
    is_false_negative = math_ops.to_float(math_ops.logical_and(label_is_pos, pred_is_neg))
    if weights_tiled is not None:
        is_false_negative *= weights_tiled
    values['fn'] = math_ops.reduce_sum(is_false_negative, 1)

    # tn
    is_true_negative = math_ops.to_float(math_ops.logical_and(label_is_neg, pred_is_neg))
    if weights_tiled is not None:
        is_true_negative *= weights_tiled
    values['tn'] = math_ops.reduce_sum(is_true_negative, 1)

    # fp
    is_false_positive = math_ops.to_float(math_ops.logical_and(label_is_neg, pred_is_pos))
    if weights_tiled is not None:
        is_false_positive *= weights_tiled
    values['fp'] = math_ops.reduce_sum(is_false_positive, 1)

    return values


def auc_fun(labels, predictions, weights=None, num_thresholds=200, name=None, summation_method='trapezoidal'):
    #if context.executing_eagerly():

    with variable_scope.variable_scope(name, 'auc', (labels, predictions, weights)):

        kepsilon = 1e-7  # to account for floating point imprecisions
        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

        values = _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights)

        # Add epsilons to avoid dividing by 0.
        epsilon = 1.0e-6

        def compute_auc(tp, fn, tn, fp, auc_name):
            """Computes the roc-auc or pr-auc based on confusion counts."""

            rec = math_ops.div(tp + epsilon, tp + fn + epsilon)
            fp_rate = math_ops.div(fp, fp + tn + epsilon)
            x = fp_rate
            y = rec

            if summation_method in ('trapezoidal', 'careful_interpolation'):
                return math_ops.reduce_sum(
                    math_ops.multiply(x[:num_thresholds - 1] - x[1:], (y[:num_thresholds - 1] + y[1:]) / 2.),
                    name=auc_name)
            elif summation_method == 'minoring':
                return math_ops.reduce_sum(
                    math_ops.multiply(
                        x[:num_thresholds - 1] - x[1:], math_ops.minimum(y[:num_thresholds - 1], y[1:])), name=auc_name)
            elif summation_method == 'majoring':
                return math_ops.reduce_sum(math_ops.multiply(
                    x[:num_thresholds - 1] - x[1:], math_ops.maximum(y[:num_thresholds - 1], y[1:])), name=auc_name)
            else:
                raise ValueError('Invalid summation_method: %s' % summation_method)

        auc_value = compute_auc(values['tp'], values['fn'], values['tn'], values['fp'], 'value')

        return auc_value


def recall_fun(labels, predictions, topk = 20, weights=None, name=None, summation_method='trapezoidal'):
    #if context.executing_eagerly():

    with variable_scope.variable_scope(name, 'recall', (labels, predictions, weights)):
        pos_num = tf.reduce_sum(labels)

        top_k_truth = tf.gather(labels, tf.math.top_k(predictions, topk).indices)

        recall_value = tf.reduce_sum(top_k_truth) / pos_num

        return recall_value


