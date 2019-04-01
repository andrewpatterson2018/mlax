import argparse
import time
import keras
import tensorflow as tf
from time import time
from keras.callbacks import TensorBoard
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adam, Adagrad
from keras.utils import to_categorical
import keras.backend as K


#three 'root' architectures:
#one: conv pooling conv pooling hidden hidden
#two: conv pooling hidden hidden
#three: hidden hidden
#from there, pick the best one and experiment with:
#1. Batch normalisation
#2. activation functions
#3. drop out/regularisation
#4. Data augmentation

#retrives the data, normalises it, and encodes the labels for the softmax outputs.
def GetData():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #reshape data to fit model
    train_images = train_images.reshape(60000,28,28, 1)
    test_images = test_images.reshape(10000,28,28, 1)

    #one-hot encode target column
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, test_images, train_labels, test_labels

def cone(combination, _learning_rate, _epochs, _batches, seed):
    train_images, test_images, train_labels, test_labels = GetData()

    model = keras.Sequential()
    #The feature detection layers.
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))     


    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation="softmax", name="output-layer"))


    model_optimizer = SGD(lr=_learning_rate, momentum=0.9, nesterov=True)

    model.compile(optimizer=model_optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
   
    model.summary()

    boardString = 'logs/fashion-{:d}-{:.3f}-{:d}-{:d}-{:d}.cpkt'.format(int(combination), _learning_rate, int(epochs), int(batches), int(seed))

    tensorboard = TensorBoard(log_dir=boardString, histogram_freq=2, write_grads=True)
    
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=int(_epochs), batch_size=int(_batches), callbacks=[tensorboard] )


    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    model.save('fashion-{:d}-{:.3f}-{:d}-{:d}-{:d}.cpkt'.format(int(combination), _learning_rate, int(epochs), int(batches), int(seed)))



def ctwo(combination, _learning_rate, _epochs, _batches, seed):
    train_images, test_images, train_labels, test_labels = GetData()

    model = keras.Sequential()
    #The feature detection layers.
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))

    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation="softmax", name="output-layer"))

    model_optimizer = SGD(lr=_learning_rate, momentum=0.9, nesterov=True)

    model.compile(optimizer=model_optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
   
    model.summary()

    boardString = 'logs/fashion-{:d}-{:.3f}-{:d}-{:d}-{:d}.cpkt'.format(int(combination), _learning_rate, int(epochs), int(batches), int(seed))

    tensorboard = TensorBoard(log_dir=boardString, histogram_freq=2, write_grads=True)
    
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=int(_epochs), batch_size=int(_batches), callbacks=[tensorboard] )


    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    model.save('fashion-{:d}-{:.3f}-{:d}-{:d}-{:d}.cpkt'.format(int(combination), _learning_rate, int(epochs), int(batches), int(seed)))

# No convolutional layers
def cthree(combination, _learning_rate, _epochs, _batches, seed):
    train_images, test_images, train_labels, test_labels = GetData()

    model = keras.Sequential()
    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation="softmax", name="output-layer"))

    model_optimizer = SGD(lr=_learning_rate)
    model.compile(optimizer=model_optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    boardString = 'logs/fashion-{:d}-{:.3f}-{:d}-{:d}-{:d}.cpkt'.format(int(combination), _learning_rate, int(epochs), int(batches), int(seed))
    tensorboard = TensorBoard(log_dir=boardString, histogram_freq=2, write_grads=True)
    
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=int(_epochs), batch_size=int(_batches), callbacks=[tensorboard] )


    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    model.save('fashion-{:d}-{:.3f}-{:d}-{:d}-{:d}.cpkt'.format(int(combination), _learning_rate, int(epochs), int(batches), int(seed)))

def cfour(combination, _learning_rate, _epochs, _batches, seed):
    train_images, test_images, train_labels, test_labels = GetData()

    model = keras.Sequential()
    #The feature detection layers.
    # input structure (28,28,1)
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))     
    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=(13,13,16)))
    
    model.add(Flatten(input_shape=(10,10,32)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(10, activation="softmax", name="output-layer"))


    model_optimizer = SGD(lr=_learning_rate, momentum=0.9, nesterov=True)
    
    model.compile(optimizer=model_optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
   
    model.summary()

    boardString = 'logs/fashion-{:d}-{:.3f}-{:d}-{:d}-{:d}.cpkt'.format(int(combination), _learning_rate, int(epochs), int(batches), int(seed))

    tensorboard = TensorBoard(log_dir=boardString, histogram_freq=2, write_grads=True)
    
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=int(_epochs), batch_size=int(_batches), callbacks=[tensorboard] )


    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    model.save('fashion-{:d}-{:.3f}-{:d}-{:d}-{:d}.cpkt'.format(int(combination), _learning_rate, int(epochs), int(batches), int(seed)))



def main(combination, learning_rate, epochs, batches, seed):
    # Set Seed
    tf.random.set_random_seed(seed)

    print("Seed: {}".format(seed))
    if int(combination)==1:
        cone(combination, learning_rate, epochs, batches, seed)
    if int(combination)==2:
        ctwo(combination, learning_rate, epochs, batches, seed)
    if int(combination)==3:
        cthree(combination, learning_rate, epochs, batches, seed)
    if int(combination)==4:
        cfour(combination, learning_rate, epochs, batches, seed)

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