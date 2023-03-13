import tensorflow as tf


def masked_sigmoid_cross_entropy(outputs, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=tf.cast(outputs, dtype=tf.float32), labels=tf.cast(labels, dtype=tf.float32))
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss = tf.multiply(loss, tf.reshape(mask, (-1, 1)))
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(input=preds, axis=1), tf.argmax(input=labels, axis=1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(input_tensor=mask)
    accuracy_all *= mask
    return tf.reduce_mean(input_tensor=accuracy_all)

# this funciton will recieve raw outputs and labelss
# trying to fix pre masking hamming loss shape


def masked_hamming_loss(outputs, labels, mask):
    """hamming loss with masking."""
    # make prediction
    # define a threshold
    thresh = tf.constant(0.5)
    # prediciton based on output and threshold
    preds = tf.math.greater(outputs, thresh, name=None)
    bool_labels = tf.equal(tf.cast(labels, tf.int32),
                           tf.constant(1))  # transform to bool

    # find where there are mistakes
    exclusive_or = tf.math.not_equal(bool_labels, preds)
    # count how many true in exclusive or of label (how many are wrong)
    ham_loss_all = tf.math.reduce_mean(
        tf.cast(exclusive_or, tf.float32), axis=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(input_tensor=mask)
    ham_loss_all *= mask
    return tf.reduce_mean(input_tensor=ham_loss_all)


def masked_alpha_eval(outputs, labels, mask):
    """hamming loss with masking."""
    # make prediction
    # define a threshold
    thresh = tf.constant(0.5)
    # prediciton based on output and threshold
    preds = tf.math.greater(outputs, thresh, name=None)
    bool_labels = tf.equal(tf.cast(labels, tf.int32),
                           tf.constant(1))  # transform to bool

    # find where there are mistakes
    exclusive_or = tf.math.not_equal(bool_labels, preds)

    # find all labels
    union = tf.math.logical_or(bool_labels, preds)

    # divide
    alpha_eval_all = tf.divide(tf.reduce_sum(tf.cast(exclusive_or, dtype=tf.int32), axis=1),
                        tf.reduce_sum(tf.cast(union, dtype=tf.int32), axis=1))
    alpha_eval_all = tf.cast(alpha_eval_all,dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(input_tensor=mask)
    alpha_eval_all *= mask
    return tf.reduce_mean(input_tensor=alpha_eval_all)
