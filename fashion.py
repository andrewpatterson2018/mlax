import argparse
import time
import keras
import tensorflow as tf
from time import time
from keras.callbacks import TensorBoard
from keras.utils import to_categorical


# FashionMINST convolutional neural network, classify images of clothing into one of 10 classes.
def feed_network(learning_rate, epochs, batches):

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0


    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='SGD',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Create a TensorBoard instance with the path to the logs directory
    tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

    model.fit(train_images, train_labels, epochs=5, callbacks=[tensorboard])

    # Test the model on the test collection
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

def conv_network(_epochs):
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #reshape data to fit model
    train_images = train_images.reshape(60000,28,28,1)
    test_images = test_images.reshape(10000,28,28,1)

    #one-hot encode target column
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64,kernel_size=3, activation= tf.nn.relu, input_shape=(28,28,1)))
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation= tf.nn.relu))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam', 
                  loss = 'categorical_crossentropy',
                  metrics=['accuracy'])


    # Create a TensorBoard instance with the path to the logs directory
    tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
    
    model.fit(train_images, train_labels, 
              validation_data=(test_images, test_labels), 
              epochs=5, callbacks=[tensorboard])

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)



def main(combination, learning_rate, epochs, batches, seed):

    # Set Seed
    print("Seed: {}".format(seed))

    if int(combination)==1:
        feed_network(learning_rate, epochs, batches)
    if int(combination)==2:
        conv_network(epochs)

    print("Done!")

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