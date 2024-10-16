from enum import Enum
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


class DatasetID(Enum):
    Mnist = 1,
    BostonHousing = 2,
    Cifar10 = 3,
    Cifar100 = 4,
    Iris = 5
    FordA = 6

# ===== Load data =====

def getMnistDataset(seed):
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # scale the image values to [0, 1]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # x_train = x_train[1:3000]
    # y_train = y_train[1:3000]

    # expand the image dimensions from (28, 28) to (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # one-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # concatenate images with labels
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # split train dataset into train and validation set
    train, val = tf.keras.utils.split_dataset(train, left_size=0.9, right_size=None,
        shuffle=True, seed=seed)

    return train, val, test

def getBostonHousingDataset(seed):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

    # concatenate images with labels
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # split train dataset into train and validation set
    train, val = tf.keras.utils.split_dataset(train, left_size=0.9, right_size=None,
        shuffle=True, seed=seed)

    return train, val, test

def getCifar10Dataset(seed):
    # https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/

    num_classes = 10

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # scale the image values to [0, 1]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # x_train = x_train[1:3000]
    # y_train = y_train[1:3000]

    # one-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train, val = tf.keras.utils.split_dataset(train, left_size=0.9, right_size=None,
        shuffle=True, seed=seed)

    return train, val, test

def getCifar100Dataset(seed):
    # https://www.geeksforgeeks.org/image-classification-using-cifar-10-and-cifar-100-dataset-in-tensorflow/

    num_classes = 100
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    # scale the image values to [0, 1]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # x_train = x_train[1:3000]
    # y_train = y_train[1:3000]

    # one-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train, val = tf.keras.utils.split_dataset(train, left_size=0.9, right_size=None,
        shuffle=True, seed=seed)

    return train, val, test

def getIrisDataset(seed):
    # https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

    num_classes = 3

    df = pd.read_csv("data/iris.csv", header=None)
    raw_data = df.values
    X = raw_data[:, 0:4].astype(float)
    Y = raw_data[:, 4]

    y_enc = LabelEncoder()
    y_enc.fit(Y)
    Y = y_enc.transform(Y)

    X = tf.convert_to_tensor(X)
    Y = tf.convert_to_tensor(Y)

    # one-hot encode the labels
    Y = tf.keras.utils.to_categorical(Y, num_classes)

    data = tf.data.Dataset.from_tensor_slices((X, Y))

    train, test = tf.keras.utils.split_dataset(data, left_size=0.9, right_size=None,
        shuffle=True, seed=seed)

    train, val = tf.keras.utils.split_dataset(train, left_size=0.9, right_size=None,
        shuffle=True, seed=seed)

    return train, val, test

def getFordADataset(seed):
    # https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

    train_data = np.loadtxt(
        "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv")
    test_data = np.loadtxt(
        "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TEST.tsv")

    x_train, y_train = (train_data[:, 1:], train_data[:, 0].astype(int))
    x_test, y_test = (test_data[:, 1:], test_data[:, 0].astype(int))

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    num_classes = len(np.unique(y_train))

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train, val = tf.keras.utils.split_dataset(train, left_size=0.9, right_size=None,
        shuffle=True, seed=seed)

    return train, val, test

def getDataset(dataset_id, seed):
    match dataset_id:
        case DatasetID.Mnist:
            train, val, test = getMnistDataset(seed)
        case DatasetID.BostonHousing:
            train, val, test = getBostonHousingDataset(seed)
        case DatasetID.Cifar10:
            train, val, test = getCifar10Dataset(seed)
        case DatasetID.Cifar100:
            train, val, test = getCifar100Dataset(seed)
        case DatasetID.Iris:
            train, val, test = getIrisDataset(seed)
        case DatasetID.FordA:
            train, val, test = getFordADataset(seed)
    print(f'Found {train.cardinality().numpy()} train instances, {val.cardinality().numpy()} '
        + f'validation instances, and {test.cardinality().numpy()} test instances.')
    return train, val, test
