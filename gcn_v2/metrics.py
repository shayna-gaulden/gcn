import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
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
def masked_hamming_loss(outputs, labels, mask):
    """hamming loss with masking."""
    # make prediction
    # define a threshold
    thresh = tf.constant(0.3)
    # prediciton based on output and threshold
    preds = tf.math.greater(outputs, thresh, name=None)
    bool_labels = tf.equal(tf.cast(labels, tf.int32), tf.constant(1))  # transform to bool
    # logical "and" used to get intersection
    intersect = tf.math.logical_and(bool_labels, preds)
    # count how many true in intersection of label
    intersect_size = tf.reduce_sum(tf.cast(intersect, tf.float32))
    # get number of labels
    num_labels = tf.constant(28)
    ham_loss_all = tf.divide(intersect_size, tf.cast(num_labels, tf.float32))
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(input_tensor=mask)
    ham_loss_all *= mask
    return tf.reduce_mean(input_tensor=ham_loss_all)
