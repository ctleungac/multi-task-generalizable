import tensorflow as tf
import numpy as np
import random


def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """
    y = y.flatten()
    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []
    labels_first = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]
        labels_first.append(label1)

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]
        labels_first.append(label2)

    return np.array(pairs), np.array(labels).astype("float32")


def data_preprocessing(label_filter):
    (x_train_val, y_train_val), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train_val = x_train_val.astype("float32")/255.0
    x_test = x_test.astype("float32")/255.0
    
    train_filter = np.where(np.in1d(y_train_val, label_filter))
    test_filter =  np.where(np.in1d(y_test, label_filter))
    x_train_val = x_train_val[train_filter]
    y_train_val = y_train_val[train_filter]
    x_test = x_test[test_filter]
    y_test = y_test[test_filter]
    
    train_val_ratio = 1
    train_num = int(x_train_val.shape[0]*train_val_ratio)
    
    x_train, x_val = x_train_val[:train_num], x_train_val[train_num:]
    y_train, y_val = y_train_val[:train_num], y_train_val[train_num:]
    del x_train_val, y_train_val
    
    pairs_train, labels_train = make_pairs(x_train, y_train)

    # make validation pairs
    pairs_val, labels_val = make_pairs(x_test, y_test)

    # make test pairs
    pairs_test, labels_test = make_pairs(x_test, y_test)
    
    x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (60000, 28, 28)
    x_train_2 = pairs_train[:, 1]

    x_val_1 = pairs_val[:, 0]  # x_val_1.shape = (60000, 28, 28)
    x_val_2 = pairs_val[:, 1]

    x_test_1 = pairs_test[:, 0]  # x_test_1.shape = (20000, 28, 28)
    x_test_2 = pairs_test[:, 1]

    return x_train_1,x_train_2,x_val_1,x_val_2,x_test_1,x_test_2,labels_train,labels_val


def dataphase2():
    (x_train_val, y_train_val), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train_val = x_train_val/255.0
    x_test = x_test/255.0

    train_val_ratio = 0.8
    train_num = int(x_train_val.shape[0]*train_val_ratio)

    Y_train_val = tf.keras.utils.to_categorical(y_train_val, 10)
    Y_test = tf.keras.utils.to_categorical(y_test, 10)

    x_train, x_val = x_train_val[:train_num], x_train_val[train_num:]
    Y_train, Y_val = Y_train_val[:train_num], Y_train_val[train_num:]

    del x_train_val, Y_train_val
    
    return x_train,x_test,x_val,Y_train,Y_test,Y_val


def dataphase2():
    (x_train_val, y_train_val), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train_val = x_train_val/255.0
    x_test = x_test/255.0

    train_val_ratio = 0.8
    train_num = int(x_train_val.shape[0]*train_val_ratio)

    Y_train_val = tf.keras.utils.to_categorical(y_train_val, 10)
    Y_test = tf.keras.utils.to_categorical(y_test, 10)

    x_train, x_val = x_train_val[:train_num], x_train_val[train_num:]
    Y_train, Y_val = Y_train_val[:train_num], Y_train_val[train_num:]

    del x_train_val, Y_train_val
    
    return x_train,x_test,x_val,Y_train,Y_test,Y_val


def dataoriginal():
    (x_train_val, y_train_val), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train_val = x_train_val/255.0
    x_test = x_test/255.0
    
    label_permutation = np.arange(10)
    LABEL_FIRST_HALF = label_permutation[:5]
    train_filter = np.where(np.in1d(y_train_val, LABEL_FIRST_HALF))
    test_filter =  np.where(np.in1d(y_test, LABEL_FIRST_HALF))

    x_train_val = x_train_val[train_filter]
    y_train_val = y_train_val[train_filter]
    x_test = x_test[test_filter]
    y_test = y_test[test_filter]

    train_val_ratio = 0.8
    train_num = int(x_train_val.shape[0]*train_val_ratio)

    Y_train_val = tf.keras.utils.to_categorical(y_train_val, 10)
    Y_test = tf.keras.utils.to_categorical(y_test, 10)

    x_train, x_val = x_train_val[:train_num], x_train_val[train_num:]
    Y_train, Y_val = Y_train_val[:train_num], Y_train_val[train_num:]

    del x_train_val, Y_train_val
    
    return x_train,x_test,x_val,Y_train,Y_test,Y_val


def getnoisevariance(SNR,rate,P=1):
    # the SNR in args[0] is actually EbN0
    snrdB = SNR + 10*np.log10(rate)
    snr = 10.0**(snrdB/10.0)
    #P_avg = 1
    N0 = P/snr
    return (N0/2)


# +
def make_pairs_with_label(x, y):
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """
    y = y.flatten()
    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []
    labels_first = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]
        labels_first.append(label1)

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]
        labels_first.append(label1)

    return np.array(pairs), np.array(labels).astype("float32"), np.array(labels_first).astype("float32")


def data_preprocessing_labels(label_filter):
    (x_train_val, y_train_val), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train_val = x_train_val.astype("float32")/255.0
    x_test = x_test.astype("float32")/255.0
    
    train_filter = np.where(np.in1d(y_train_val, label_filter))
    test_filter =  np.where(np.in1d(y_test, label_filter))
    x_train_val = x_train_val[train_filter]
    y_train_val = y_train_val[train_filter]
    x_test = x_test[test_filter]
    y_test = y_test[test_filter]
    
    train_val_ratio = 1
    train_num = int(x_train_val.shape[0]*train_val_ratio)
    
    x_train, x_val = x_train_val[:train_num], x_train_val[train_num:]
    y_train, y_val = y_train_val[:train_num], y_train_val[train_num:]
    del x_train_val, y_train_val
    
    pairs_train, labels_train, label1_train = make_pairs_with_label(x_train, y_train)

    # make validation pairs
    pairs_val, labels_val, label1_val = make_pairs_with_label(x_test, y_test)

    # make test pairs
    pairs_test, labels_test, label1_test = make_pairs_with_label(x_test, y_test)
    
    x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (60000, 28, 28)
    x_train_2 = pairs_train[:, 1]

    x_val_1 = pairs_val[:, 0]  # x_val_1.shape = (60000, 28, 28)
    x_val_2 = pairs_val[:, 1]

    x_test_1 = pairs_test[:, 0]  # x_test_1.shape = (20000, 28, 28)
    x_test_2 = pairs_test[:, 1]

    return x_train_1,x_train_2,x_val_1,x_val_2,x_test_1,x_test_2,labels_train,labels_val, label1_train,label1_val

