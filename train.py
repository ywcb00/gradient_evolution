from dataset import DatasetID

import numpy as np
import os
from pathlib import Path
import pickle
import tensorflow as tf


# ===== Build model =====

def getMnistModel(data_element_spec, loss_obj, seed):
    num_classes = data_element_spec[1].shape[0]

    keras_model = tf.keras.Sequential()
    keras_model.add(tf.keras.Input(shape=data_element_spec[0].shape))

    keras_model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    keras_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    keras_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    keras_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    keras_model.add(tf.keras.layers.Flatten())
    keras_model.add(tf.keras.layers.Dropout(0.5, seed=seed))
    keras_model.add(tf.keras.layers.Dense(num_classes, activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.01)
    keras_model.compile(optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.metrics.CategoricalCrossentropy(),
            tf.metrics.CategoricalAccuracy()])

    return keras_model

def getBostonHousingModel(data_element_spec, loss_obj, seed):
    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=data_element_spec[0].shape))

    model.add(tf.keras.layers.Dense(64, activation='relu',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.Dense(1, activation='linear',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    model.compile(optimizer=optimizer, loss=loss_obj(),
        metrics=[tf.metrics.MeanSquaredError(), tf.metrics.MeanAbsoluteError()])

    return model

def getCifar10Model(data_element_spec, loss_obj, seed):
    # https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/

    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=data_element_spec[0].shape))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.2, seed=seed))

    model.add(tf.keras.layers.Dense(1024, activation='relu',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.Dropout(0.2, seed=seed))

    model.add(tf.keras.layers.Dense(10, activation='softmax',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
    model.compile(optimizer=optimizer, loss=loss_obj(),
        metrics=[tf.metrics.CategoricalCrossentropy(), tf.metrics.CategoricalAccuracy()])

    return model

def getCifar100Model(data_element_spec, loss_obj, seed):
    # https://www.geeksforgeeks.org/image-classification-using-cifar-10-and-cifar-100-dataset-in-tensorflow/

    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=data_element_spec[0].shape))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.Dropout(0.3, seed=seed))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(100, activation='softmax',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=optimizer, loss=loss_obj(),
        metrics=[tf.metrics.CategoricalCrossentropy(), tf.metrics.CategoricalAccuracy()])

    return model

def getIrisModel(data_element_spec, loss_obj, seed):
    # https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=data_element_spec[0].shape))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=loss_obj(),
        metrics=[tf.metrics.CategoricalCrossentropy(), tf.metrics.CategoricalAccuracy()])

    return model

def getFordAModel(data_element_spec, loss_obj, seed):
    # https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

    num_classes = 2

    model = tf.keras.Sequential()
    
    model.add(tf.keras.Input(shape=data_element_spec[0].shape))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=loss_obj(),
        metrics=[tf.metrics.SparseCategoricalCrossentropy(),
            tf.metrics.SparseCategoricalAccuracy()])

    return model

# ===== Fit model =====
def fitGradient(model, train, loss_obj):
    train_metrics = None
    individual_gradients = list()

    for step, (x_batch_train, y_batch_train) in enumerate(train):
        with tf.GradientTape() as tape:
            preds = model(x_batch_train, training=True)
            loss_value = loss_obj()(y_batch_train, preds)
        grad = tape.gradient(loss_value, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grad, model.trainable_variables))
        grad = np.array([g.numpy() for g in grad], dtype=object)
        individual_gradients.append(grad)
        if(step == 0):
            accumulated_grad = grad.copy()
        else:
            accumulated_grad += grad

    evaluation_scalars = model.evaluate(train, verbose=2)
    scalar_train_metrics = dict(zip(model.metrics_names, evaluation_scalars))
    if(train_metrics == None):
        train_metrics = {mname: [mval] for mname, mval in scalar_train_metrics.items()}
    else:
        for mname, mval in scalar_train_metrics.items():
            train_metrics[mname].append(mval)

    return accumulated_grad, train_metrics, individual_gradients

def buildAndFit(dataset_id, train, figures_dir, seed):
    match dataset_id:
        case DatasetID.Mnist:
            BATCH_SIZE = 256
            NUM_EPOCHS = 10
            LOSS = tf.keras.losses.CategoricalCrossentropy
        case DatasetID.BostonHousing:
            BATCH_SIZE = 32
            NUM_EPOCHS = 100
            LOSS = tf.keras.losses.MeanSquaredError
        case DatasetID.Cifar10:
            BATCH_SIZE = 1024
            NUM_EPOCHS = 15
            LOSS = tf.keras.losses.CategoricalCrossentropy
        case DatasetID.Cifar100:
            BATCH_SIZE = 1024
            NUM_EPOCHS = 15
            LOSS = tf.keras.losses.CategoricalCrossentropy
        case DatasetID.Iris:
            BATCH_SIZE = 15
            NUM_EPOCHS = 100
            LOSS = tf.keras.losses.CategoricalCrossentropy
        case DatasetID.FordA:
            BATCH_SIZE = 32
            NUM_EPOCHS = 500
            LOSS = tf.keras.losses.SparseCategoricalCrossentropy

    match dataset_id:
        case DatasetID.Mnist:
            model = getMnistModel(train.element_spec, LOSS, seed)
        case DatasetID.BostonHousing:
            model = getBostonHousingModel(train.element_spec, LOSS, seed)
        case DatasetID.Cifar10:
            model = getCifar10Model(train.element_spec, LOSS, seed)
        case DatasetID.Cifar100:
            model = getCifar100Model(train.element_spec, LOSS, seed)
        case DatasetID.Iris:
            model = getIrisModel(train.element_spec, LOSS, seed)
        case DatasetID.FordA:
            model = getFordAModel(train.element_spec, LOSS, seed)

    gradients = list()
    metrics = list()
    individual_gradients = list()
    for epoch in range(NUM_EPOCHS):
        grad, metr, ind_grad = fitGradient(model, train.batch(batch_size=BATCH_SIZE), LOSS)

        gradients.append(grad)
        metrics.append(metr)
        individual_gradients.extend(ind_grad)

    # save gradients to disk
    filehandler = open(figures_dir/r'gradients.pkl', "wb")
    pickle.dump(gradients, filehandler)
    filehandler.close()
    # save individual gradients to disk
    filehandler = open(figures_dir/r'individual_gradients.pkl', "wb")
    pickle.dump(individual_gradients, filehandler)
    filehandler.close()
    # save metrics to disk
    filehandler = open(figures_dir/r'metrics.pkl', "wb")
    pickle.dump(metrics, filehandler)
    filehandler.close()

    return gradients, individual_gradients, metrics
