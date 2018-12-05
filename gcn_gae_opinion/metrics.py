import tensorflow as tf
import numpy as np


def masked_mse_square(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # loss = tf.square(preds - labels)
    loss = tf.square(preds - labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def masked_mse(preds, labels, mask):
    """ MSE with masking."""
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # loss = tf.square(preds - labels)
    # preds = tf.reduce_mean(preds, axis=1)
    loss = tf.square(preds - labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_mae(preds, labels, mask):
    """ MSE with masking."""
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # loss = tf.square(preds - labels)
    # preds = tf.reduce_mean(preds, axis=1)
    loss = tf.abs(preds - labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_decode(preds, mask):
    """MSE decode with masking."""
    mask = tf.cast(mask, dtype=tf.float32)
    loss = mask - preds
    mask /= tf.reduce_mean(mask)
    loss = loss * mask
    return tf.reduce_mean(loss)


def masked_decode_sparse(pred, adj):
    logits = tf.convert_to_tensor(pred, name="logits")
    targets = tf.convert_to_tensor(adj, name="targets")
    # targets = tf.reshape(targets, [70317, 70317])
    try:
        targets.get_shape().merge_with(logits.get_shape())
    except ValueError:
        raise ValueError(
            "logits and targets must have the same shape (%s vs %s)" %
            (logits.get_shape(), targets.get_shape()))
    loss = targets - logits
    targets /= tf.reduce_mean(targets)
    loss = loss * targets
    return tf.reduce_mean(loss)


def masked_mae_np(preds, labels, mask):
    """ MSE with masking."""
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # loss = tf.square(preds - labels)
    # preds = tf.reduce_mean(preds, axis=1)
    loss = np.abs(preds - labels)
    mask = np.asarray(mask, dtype=float)
    mask /= np.mean(mask)
    loss *= mask
    return np.mean(loss)


def masked_guassian_loss(preds, labels, mask, sigma):
    """ MSE with masking."""
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # loss = tf.square(preds - labels)
    # preds = tf.reduce_mean(preds, axis=1)
    loss = tf.square(preds - labels)
    loss = 0.5 * tf.exp(-sigma) * loss + 0.5 * sigma
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_laplace_loss(preds, labels, mask, sigma):
    """ MSE with masking."""
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # loss = tf.square(preds - labels)
    # preds = tf.reduce_mean(preds, axis=1)
    loss = tf.abs(preds - labels)
    loss = 0.5 * tf.exp(-sigma) * loss + 0.5 * sigma
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_dir_error(p1, p2, p3, l1, l2, l3, mask):
    sum1 = p1 + p2 + p3
    sum2 = l1 + l2 + l3
    e1 = np.square((p1) / sum1 - (l1) / sum2)
    e2 = np.square((p2) / sum1 - (l2) / sum2)
    e3 = np.square((p3) / sum1 - (l3) / sum2)
    error = (e1 + e2 + e3) / 3
    mask = np.asarray(mask, dtype=float)
    mask /= np.mean(mask)
    error *= mask
    return np.mean(error)


def masked_dir_error2(p1, p2, un_p, l1, l2, uncertainty, mask):
    num = 3 / uncertainty
    num_p = 3 / un_p
    p3 = 1 - p1 - p2
    l3 = 1 - l1 - l2
    alpha1_p = num_p * p1
    alpha2_p = num_p * p2
    alpha3_p = num_p * p3
    alpha1 = num * l1
    alpha2 = num * l2
    alpha3 = num * l3
    e1 = np.square(alpha1_p - alpha1)
    e2 = np.square(alpha2_p - alpha2)
    e3 = np.square(alpha3_p - alpha3)
    error = (e1 + e2 + e3) / 3
    mask = np.asarray(mask, dtype=float)
    mask /= np.mean(mask)
    error *= mask
    return np.mean(error)
