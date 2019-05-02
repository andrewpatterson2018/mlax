import argparse, numpy, time, keras
import tensorflow as tf
from time import time
from keras.losses import categorical_crossentropy
from keras.callbacks import TensorBoard
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import SGD, Adadelta
from keras.utils import to_categorical
import keras.backend as K


# three 'root' architectures:
# 1: shallow but wide network.
# 2: Deep network with multiple convolutions before pooling and dropout applied.
# 3: factorised convolutional layers, (in essence conv-pooling-conv-pooling)
# 4: Same as 3, but using ELU activation
# 5: Same as 4, less feature maps
# 6: Same as 5 but dropout to bring training/test accuracy into similar range.

# retrives the data, normalises it, and encodes the labels for the softmax outputs.
def GetData():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # reshape data to fit model
    train_images = train_images.reshape(60000, 28, 28, 1)
    test_images = test_images.reshape(10000, 28, 28, 1)

    # one-hot encode target column
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, test_images, train_labels, test_labels


def cone(combination, _learning_rate, _epochs, _batches, seed):
    train_images, test_images, train_labels, test_labels = GetData()
    model = keras.Sequential()

    # The feature detection layers.
    model.add(Conv2D(32, kernel_size=(3, 1), activation="elu", input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(1, 3), activation="elu"))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Conv2D(32, kernel_size=(3, 1), activation="elu"))
    model.add(Conv2D(32, kernel_size=(1, 3), activation="elu"))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Flatten())
    model.add(Dense(256, activation="elu"))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation="elu"))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation="softmax", name="output-layer"))

    model_optimizer = SGD(lr=_learning_rate, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=model_optimizer, loss=categorical_crossentropy, metrics=["accuracy"]
    )

    tensorboard = buildTensorBoard(combination, _learning_rate, epochs, batches, seed)

    model.summary()

    model.fit(
        train_images,
        train_labels,
        validation_data=(test_images, test_labels),
        epochs=int(_epochs),
        batch_size=int(_batches),
        callbacks=[tensorboard],
    )

    evaluate(model, train_images, train_labels, test_images, test_labels)

    model.save(
        "fashion-{0}-{1}-{2}-{3}-{4}.cpkt".format(
            combination, _learning_rate, epochs, batches, seed
        )
    )


def csix(combination, _learning_rate, _epochs, _batches, seed):
    train_images, test_images, train_labels, test_labels = GetData()

    model = keras.Sequential()
    # The feature detection layers.
    model.add(
        Conv2D(64, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1))
    )
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax", name="output-layer"))

    model_optimizer = SGD(lr=_learning_rate, momentum=0.9, nesterov=True)

    model.compile(
        optimizer=model_optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    tensorboard = buildTensorBoard(combination, _learning_rate, epochs, batches, seed)

    model.fit(
        train_images,
        train_labels,
        validation_data=(test_images, test_labels),
        epochs=int(_epochs),
        batch_size=int(_batches),
        callbacks=[tensorboard],
    )

    evaluate(model, train_images, train_labels, test_images, test_labels)

    model.save(
        "fashion-{0}-{1}-{2}-{3}-{4}.cpkt".format(
            combination, _learning_rate, epochs, batches, seed
        )
    )


def ctwo(combination, _learning_rate, _epochs, _batches, seed):
    train_images, test_images, train_labels, test_labels = GetData()
    model = keras.Sequential()

    # The feature detection layers.
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1))
    )
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax", name="output-layer"))

    model_optimizer = Adadelta(lr=_learning_rate)
    model.compile(
        optimizer=model_optimizer, loss=categorical_crossentropy, metrics=["accuracy"]
    )

    tensorboard = buildTensorBoard(combination, _learning_rate, epochs, batches, seed)

    model.summary()

    model.fit(
        train_images,
        train_labels,
        validation_data=(test_images, test_labels),
        epochs=int(_epochs),
        batch_size=int(_batches),
        callbacks=[tensorboard],
    )

    evaluate(model, train_images, train_labels, test_images, test_labels)

    model.save(
        "fashion-{0}-{1}-{2}-{3}-{4}.cpkt".format(
            combination, _learning_rate, epochs, batches, seed
        )
    )


def cthree(combination, _learning_rate, _epochs, _batches, seed):
    train_images, test_images, train_labels, test_labels = GetData()
    model = keras.Sequential()

    # The feature detection layers.
    model.add(
        Conv2D(64, kernel_size=(3, 1), activation="relu", input_shape=(28, 28, 1))
    )
    model.add(Conv2D(64, kernel_size=(1, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Conv2D(64, kernel_size=(3, 1), activation="relu"))
    model.add(Conv2D(64, kernel_size=(1, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(10, activation="softmax", name="output-layer"))

    model_optimizer = SGD(lr=_learning_rate, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=model_optimizer, loss=categorical_crossentropy, metrics=["accuracy"]
    )

    tensorboard = buildTensorBoard(combination, _learning_rate, epochs, batches, seed)

    model.summary()

    model.fit(
        train_images,
        train_labels,
        validation_data=(test_images, test_labels),
        epochs=int(_epochs),
        batch_size=int(_batches),
        callbacks=[tensorboard],
    )
    evaluate(model, train_images, train_labels, test_images, test_labels)

    model.save(
        "fashion-{0}-{1}-{2}-{3}-{4}.cpkt".format(
            combination, _learning_rate, epochs, batches, seed
        )
    )


def cfour(combination, _learning_rate, _epochs, _batches, seed):
    train_images, test_images, train_labels, test_labels = GetData()
    model = keras.Sequential()

    # The feature detection layers.
    model.add(Conv2D(64, kernel_size=(3, 1), activation="elu", input_shape=(28, 28, 1)))
    model.add(Conv2D(64, kernel_size=(1, 3), activation="elu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Conv2D(64, kernel_size=(3, 1), activation="elu"))
    model.add(Conv2D(64, kernel_size=(1, 3), activation="elu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Flatten())
    model.add(Dense(256, activation="elu"))
    model.add(Dense(256, activation="elu"))
    model.add(Dense(10, activation="softmax", name="output-layer"))

    model_optimizer = SGD(lr=_learning_rate, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=model_optimizer, loss=categorical_crossentropy, metrics=["accuracy"]
    )

    tensorboard = buildTensorBoard(combination, _learning_rate, epochs, batches, seed)

    model.summary()

    model.fit(
        train_images,
        train_labels,
        validation_data=(test_images, test_labels),
        epochs=int(_epochs),
        batch_size=int(_batches),
        callbacks=[tensorboard],
    )

    evaluate(model, train_images, train_labels, test_images, test_labels)

    model.save(
        "fashion-{0}-{1}-{2}-{3}-{4}.cpkt".format(
            combination, _learning_rate, epochs, batches, seed
        )
    )


def cfive(combination, _learning_rate, _epochs, _batches, seed):
    train_images, test_images, train_labels, test_labels = GetData()
    model = keras.Sequential()

    # The feature detection layers.
    model.add(Conv2D(32, kernel_size=(3, 1), activation="elu", input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(1, 3), activation="elu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Conv2D(32, kernel_size=(3, 1), activation="elu"))
    model.add(Conv2D(32, kernel_size=(1, 3), activation="elu"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

    model.add(Flatten())
    model.add(Dense(256, activation="elu"))
    model.add(Dense(256, activation="elu"))
    model.add(Dense(10, activation="softmax", name="output-layer"))

    model_optimizer = SGD(lr=_learning_rate, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=model_optimizer, loss=categorical_crossentropy, metrics=["accuracy"]
    )

    tensorboard = buildTensorBoard(combination, _learning_rate, epochs, batches, seed)

    model.summary()

    model.fit(
        train_images,
        train_labels,
        validation_data=(test_images, test_labels),
        epochs=int(_epochs),
        batch_size=int(_batches),
        callbacks=[tensorboard],
    )

    evaluate(model, train_images, train_labels, test_images, test_labels)

    model.save(
        "fashion-{0}-{1}-{2}-{3}-{4}.cpkt".format(
            combination, _learning_rate, epochs, batches, seed
        )
    )


def buildTensorBoard(combination, _learning_rate, epochs, batches, seed):
    boardString = "logs/fashion-{0}-{1}-{2}-{3}-{4}.cpkt".format(
        combination, _learning_rate, epochs, batches, seed
    )
    tensorboard = TensorBoard(log_dir=boardString, histogram_freq=2, write_grads=True)

    return tensorboard


def evaluate(model, train_images, train_labels, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    train_loss, train_acc = model.evaluate(train_images, train_labels)
    print(
        "Train Loss: {}, Train Accuracy: {}; Test Loss: {}, Test Accuracy: {}".format(
            train_loss, train_acc, test_loss, test_acc
        )
    )


def main(combination, learning_rate, epochs, batches, seed):
    # Set Seed
    numpy.random.seed(int(seed))
    tf.random.set_random_seed(int(seed))

    print("Seed: {}".format(seed))
    if int(combination) == 1:
        cone(combination, learning_rate, epochs, batches, seed)
    if int(combination) == 2:
        ctwo(combination, learning_rate, epochs, batches, seed)
    if int(combination) == 3:
        cthree(combination, learning_rate, epochs, batches, seed)
    if int(combination) == 4:
        cfour(combination, learning_rate, epochs, batches, seed)
    if int(combination) == 5:
        cfive(combination, learning_rate, epochs, batches, seed)
    if int(combination) == 6:
        csix(combination, learning_rate, epochs, batches, seed)


def check_param_is_numeric(param, value):
    try:
        value = float(value)
    except:
        print("{} must be numeric".format(param))
        quit(1)
    return value


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("combination", help="Flag to indicate which network to run")
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")
    arg_parser.add_argument("iterations", help="Number of iterations to perform")
    arg_parser.add_argument("batches", help="Number of batches to use")
    arg_parser.add_argument("seed", help="Seed to initialize the network")

    args = arg_parser.parse_args()

    combination = check_param_is_numeric("combination", args.combination)
    learning_rate = check_param_is_numeric("learning_rate", args.learning_rate)
    epochs = check_param_is_numeric("epochs", args.iterations)
    batches = check_param_is_numeric("batches", args.batches)
    seed = check_param_is_numeric("seed", args.seed)

    main(combination, learning_rate, epochs, batches, seed)
