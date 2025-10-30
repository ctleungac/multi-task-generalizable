# +
import tensorflow as tf
import tensorflow.keras.backend as K

def marginloss(margin=1):
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
    margin: Integer, defines the baseline for distance for which pairs
        should be classified as dissimilar. - (default is 1).

    Returns:
    'contrastive_loss' function with data ('margin') attached.
    """

    def contrastive_loss(y, preds):
        # explicitly cast the true class label data type to the predicted
        # class label data type (otherwise we run the risk of having two
        # separate data types, causing TensorFlow to error out)
        y = tf.cast(y, preds.dtype)
        tf.print(y.shape)
        tf.print(preds.shape)
        # calculate the contrastive loss between the true labels and
        # the predicted labels
        squaredPreds = K.square(preds)
        squaredMargin = K.square(K.maximum(margin - preds, 0))
        loss = K.mean((1-y) * squaredPreds + (y) * squaredMargin)
        
        return loss

    return contrastive_loss


# -

def RelaxMarginloss(margin=1,tau=999999999999):
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
    margin: Integer, defines the baseline for distance for which pairs
        should be classified as dissimilar. - (default is 1).
        
   tau: how much relax you want. default = inf means no relax

    Returns:
    'contrastive_loss' function with data ('margin') attached.
    
    """

    def contrastive_loss(y, preds):
        # explicitly cast the true class label data type to the predicted
        # class label data type (otherwise we run the risk of having two
        # separate data types, causing TensorFlow to error out)
        y = tf.cast(y, preds.dtype)
        # calculate the contrastive loss between the true labels and the predicted labels
        squaredPreds_tau = K.maximum(K.square(preds - (margin/tau)) ,0)
        squaredMargin = K.square(K.maximum(margin - preds, 0))
        loss = K.mean((1-y) * squaredPreds_tau + (y) * squaredMargin)
        
        return loss

    return contrastive_loss


# +

def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.square(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
