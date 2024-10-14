from enum import Enum
import matplotlib.animation as animplt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

class DatasetID(Enum):
    Mnist = 1,
    BostonHousing = 2,
    Cifar10 = 3,
    Cifar100 = 4,
    Iris = 5

dataset_id = DatasetID.Mnist

seed = 13

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
        BATCH_SIZE = 32
        NUM_EPOCHS = 15
        LOSS = tf.keras.losses.CategoricalCrossentropy
    case DatasetID.Cifar100:
        BATCH_SIZE = 64
        NUM_EPOCHS = 15
        LOSS = tf.keras.losses.CategoricalCrossentropy
    case DatasetID.Iris:
        BATCH_SIZE = 15
        NUM_EPOCHS = 100
        LOSS = tf.keras.losses.CategoricalCrossentropy

FIGURES_DIR = Path(f'figures_{dataset_id.name}')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ===== Load data =====

def getMnistDataset():
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

def getBostonHousingDataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

    # concatenate images with labels
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # split train dataset into train and validation set
    train, val = tf.keras.utils.split_dataset(train, left_size=0.9, right_size=None,
        shuffle=True, seed=seed)

    return train, val, test

def getCifar10Dataset():
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

def getCifar100Dataset():
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

def getIrisDataset():
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

match dataset_id:
    case DatasetID.Mnist:
        train, val, test = getMnistDataset()
    case DatasetID.BostonHousing:
        train, val, test = getBostonHousingDataset()
    case DatasetID.Cifar10:
        train, val, test = getCifar10Dataset()
    case DatasetID.Cifar100:
        train, val, test = getCifar100Dataset()
    case DatasetID.Iris:
        train, val, test = getIrisDataset()
print(f'Found {train.cardinality().numpy()} train instances, {val.cardinality().numpy()} '
    + f'validation instances, and {test.cardinality().numpy()} test instances.')


# ===== Build model =====

def getMnistModel(data_element_spec):
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

def getBostonHousingModel(data_element_spec):
    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=data_element_spec[0].shape))

    model.add(tf.keras.layers.Dense(64, activation='relu',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))
    model.add(tf.keras.layers.Dense(1, activation='linear',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer=tf.keras.initializers.Zeros()))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    model.compile(optimizer=optimizer, loss=LOSS(),
        metrics=[tf.metrics.MeanSquaredError(), tf.metrics.MeanAbsoluteError()])

    return model

def getCifar10Model(data_element_spec):
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

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=LOSS(),
        metrics=[tf.metrics.CategoricalCrossentropy(), tf.metrics.CategoricalAccuracy()])

    return model

def getCifar100Model(data_element_spec):
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

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=LOSS(),
        metrics=[tf.metrics.CategoricalCrossentropy(), tf.metrics.CategoricalAccuracy()])

    return model

def getIrisModel(data_element_spec):
    # https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=data_element_spec[0].shape))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=LOSS(),
        metrics=[tf.metrics.CategoricalCrossentropy(), tf.metrics.CategoricalAccuracy()])

    return model


match dataset_id:
    case DatasetID.Mnist:
        model = getMnistModel(train.element_spec)
    case DatasetID.BostonHousing:
        model = getBostonHousingModel(train.element_spec)
    case DatasetID.Cifar10:
        model = getCifar10Model(train.element_spec)
    case DatasetID.Cifar100:
        model = getCifar100Model(train.element_spec)
    case DatasetID.Iris:
        model = getIrisModel(train.element_spec)


# ===== Fit model =====

def fitGradient(model, train):
    train_metrics = None
    individual_gradients = list()

    for step, (x_batch_train, y_batch_train) in enumerate(train):
        with tf.GradientTape() as tape:
            preds = model(x_batch_train, training=True)
            loss_value = LOSS()(y_batch_train, preds)
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

gradients = list()
metrics = list()
individual_gradients = list()
for epoch in range(NUM_EPOCHS):
    grad, metr, ind_grad = fitGradient(model, train.batch(batch_size=BATCH_SIZE))

    gradients.append(grad)
    metrics.append(metr)
    individual_gradients.extend(ind_grad)

# ===== Create Animation =====
def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

# Plot heatmap animation of gradient
fig = plt.figure()
gs = fig.add_gridspec(len(gradients[0]), 2)
ax_grad = fig.add_subplot(gs[:, 0])
ax_grad.axis("off")
ax_grad.set_title("Gradient")

# compute factors to scale the gradients to an overall maximum of 1 for the heatmap (layerwise or global)
grad_layer_scaling = [np.max(np.absolute(np.array(layer_grads).flatten())) for layer_grads in zip(*gradients)]
grad_layer_scaling = [gls if gls != 0 else 1 for gls in grad_layer_scaling]
grad_scaling = np.max(np.array(grad_layer_scaling))
def showGrad(i):
    for counter, layer_grad in enumerate(gradients[i]):
        primfac = prime_factors(layer_grad.size)
        p = int(np.prod([elem for idx, elem in enumerate(primfac) if idx % 2 == 0]))
        q = int(np.prod([elem for idx, elem in enumerate(primfac) if idx % 2 == 1]))
        ax = fig.add_subplot(gs[counter, 0])
        # layer-wise scaling between 0 and 1
        # ax.imshow(np.absolute(layer_grad.reshape((min(p, q), max(p, q))) / grad_layer_scaling[counter]), vmin=0, vmax=1)
        # global scaling between 0 and 1
        ax.imshow(np.absolute(layer_grad.reshape((min(p, q), max(p, q))) / grad_scaling), vmin=0, vmax=1)
        ax.set_axis_off()

# Compute cumsum of gradients (i.e., accumulated gradient)
accumulated_gradients = np.cumsum(gradients, axis=0)

ax_grad = fig.add_subplot(gs[:, 1])
ax_grad.axis("off")
ax_grad.set_title("Accumulated Gradient")

accgrad_layer_scaling = [np.max(np.absolute(np.array(layer_grads).flatten())) for layer_grads in zip(*accumulated_gradients)]
accgrad_layer_scaling = [als if als != 0 else 1 for als in accgrad_layer_scaling]
accgrad_scaling = np.max(np.array(accgrad_layer_scaling))
def showAccGrad(i):
    for counter, layer_grad in enumerate(accumulated_gradients[i]):
        primfac = prime_factors(layer_grad.size)
        p = int(np.prod([elem for idx, elem in enumerate(primfac) if idx % 2 == 0]))
        q = int(np.prod([elem for idx, elem in enumerate(primfac) if idx % 2 == 1]))
        ax = fig.add_subplot(gs[counter, 1])
        # layer-wise scaling of gradient between 0 and 1
        # ax.imshow(np.absolute(layer_grad.reshape((min(p, q), max(p, q))) / accgrad_layer_scaling[counter]), vmin=0, vmax=1)
        # global scaling of gradient between 0 and 1
        ax.imshow(np.absolute(layer_grad.reshape((min(p, q), max(p, q))) / accgrad_scaling), vmin=0, vmax=1)
        ax.set_axis_off()

def showGradients(i):
    fig.suptitle(f'Epoch {i}')
    showGrad(i)
    showAccGrad(i)
anim = animplt.FuncAnimation(fig, showGradients, frames=NUM_EPOCHS, interval=1000)
anim.save(FIGURES_DIR/r'animation.gif', writer=animplt.PillowWriter(fps=1))


# ===== Obtain the highest gradient elements =====
grad_layer_highest_elements = [[np.max(np.absolute(layer_grad.flatten())) / grad_layer_scaling[counter]
        for counter, layer_grad in enumerate(grad)]
    for grad in gradients]
grad_highest_elements = np.max(np.array(grad_layer_highest_elements), axis=0)
grad_layer_highest_elements = list(zip(*grad_layer_highest_elements))

accgrad_layer_highest_elements = [[np.max(np.absolute(layer_grad.flatten())) / accgrad_layer_scaling[counter]
        for counter, layer_grad in enumerate(accgrad)]
    for accgrad in accumulated_gradients]
accgrad_highest_elements = np.max(np.array(accgrad_layer_highest_elements), axis=0)
accgrad_layer_highest_elements = list(zip(*accgrad_layer_highest_elements))

plt.figure()
for counter, glhe in enumerate(grad_layer_highest_elements):
    plt.plot(range(NUM_EPOCHS), glhe, label=f'Grad Layer {counter}')
for counter, alhe in enumerate(accgrad_layer_highest_elements):
    plt.plot(range(NUM_EPOCHS), alhe, linestyle='dashed', label=f'Accgrad Layer {counter}')
plt.title("Highest Gradient Elements")
plt.legend(loc="upper center")
ax = plt.gca()
ax.get_yaxis().set_visible(False)
plt.savefig(FIGURES_DIR/r'highestgradelement.png')


# ===== Compute statistics =====
statistics = dict()

# computing the norm of the gradients as the sum of the layer gradient norms
statistics["l1_norm"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=1) for layer_grad in grad]) for grad in gradients]
statistics["l1_norm_standardized"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=1) / layer_grad.size for layer_grad in grad]) for grad in gradients]
statistics["l2_norm"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=2) for layer_grad in grad]) for grad in gradients]
statistics["l2_norm_standardized"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=2) / layer_grad.size for layer_grad in grad]) for grad in gradients]

# computing the norm of the individual gradients as the sum of the layer gradient norms
statistics["l1_norm_individual"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=1) for layer_grad in grad]) for grad in individual_gradients]
statistics["l1_norm_individual_standardized"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=1) / layer_grad.size for layer_grad in grad]) for grad in individual_gradients]
statistics["l2_norm_individual"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=2) for layer_grad in grad]) for grad in individual_gradients]
statistics["l2_norm_individual_standardized"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=2) / layer_grad.size for layer_grad in grad]) for grad in individual_gradients]

# computing the norm of the accumulated gradients (cumsum) as the sum of the layer gradient norms
statistics["l1_norm_acc"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=1) for layer_grad in accgrad]) for accgrad in accumulated_gradients]
statistics["l1_norm_acc_standardized"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=1) / layer_grad.size for layer_grad in accgrad]) for accgrad in accumulated_gradients]
statistics["l2_norm_acc"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=2) for layer_grad in accgrad]) for accgrad in accumulated_gradients]
statistics["l2_norm_acc_standardized"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=2) / layer_grad.size for layer_grad in accgrad]) for accgrad in accumulated_gradients]


statistics["loss"] = [metr[list(metr.keys())[0]] for metr in metrics]
statistics["metric"] = [metr[list(metr.keys())[-1]] for metr in metrics]

# ===== Plot statistics =====
def scaleToMax1(arr):
    return np.array(arr) / np.max(np.array(arr))

plt.figure()

plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l1_norm"]), label="L1 Norm")
# plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l2_norm"]), label="L2 Norm")
# plt.plot(scaleToMax1(range(0, len(statistics["l1_norm_individual"])))*NUM_EPOCHS, scaleToMax1(statistics["l1_norm_individual"]), label="L1 Norm Individual")
# plt.plot(scaleToMax1(range(0, len(statistics["l2_norm_individual"])))*NUM_EPOCHS, scaleToMax1(statistics["l2_norm_individual"]), label="L2 Norm Individual")
# plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l1_norm_acc"]), label="L1 Norm Acc.")
# plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l2_norm_acc"]), label="L2 Norm Acc.")

plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l1_norm_standardized"]), label="L1 Norm Std.")
plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l2_norm_standardized"]), label="L2 Norm Std.")
# plt.plot(scaleToMax1(range(0, len(statistics["l1_norm_individual_standardized"])))*NUM_EPOCHS, scaleToMax1(statistics["l1_norm_individual_standardized"]), label="L1 Norm Individual Std.")
# plt.plot(scaleToMax1(range(0, len(statistics["l2_norm_individual_standardized"])))*NUM_EPOCHS, scaleToMax1(statistics["l2_norm_individual_standardized"]), label="L2 Norm Individual Std.")
plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l1_norm_acc_standardized"]), label="L1 Norm Acc. Std.")
plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l2_norm_acc_standardized"]), label="L2 Norm Acc. Std.")

plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["loss"]), label="Loss")
plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["metric"]), label="Metric")

plt.legend(loc="upper center")
ax = plt.gca()
ax.get_yaxis().set_visible(False)
plt.savefig(FIGURES_DIR/r'statistics.png')
