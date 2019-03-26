import argparse
import time
import keras
import tensorflow as tf
from time import time
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical

#retrives the data, normalises it, and encodes the labels for the softmax outputs.
def preprocessData():
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

# FashionMINST convolutional neural network, classify images of clothing into one of 10 classes.
def feed_network(_learning_rate):
 
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28,1)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax, name="output-layer"))

    model_optimizer = SGD(lr=_learning_rate)
    model.compile(optimizer=model_optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

#loss: 0.0385 - acc: 0.9857 - val_loss: 0.4196 - val_acc: 0.9101
def conv_network(_learning_rate):
    
    #Thought process: the convolutional/pooling processes examine small details and pass on the most prominent, which are further abstracted.

    model = keras.Sequential()
    #The feature detection layers.
    model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))    
    
    model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))    

    
    model.add(keras.layers.Flatten())

    #The fully-connected layers, or where the thinking goes on.
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax, name="output-layer"))

    model_optimizer = SGD(lr=_learning_rate, momentum=0.9, nesterov=True)

    model.compile(optimizer=model_optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
   
    model.summary()

    return model

def evaluate(model, _epochs, _batches, train_images, train_labels, test_images, test_labels):
    # Create a TensorBoard instance with the path to the logs directory
    tensorboard = TensorBoard(log_dir='logs/{}'.format(time()), histogram_freq=2, write_grads=True)

    model.fit(train_images, train_labels, 
              validation_data=(test_images, test_labels), 
              epochs=int(_epochs), batch_size=int(_batches), callbacks=[tensorboard] )

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

def main(combination, learning_rate, epochs, batches, seed):

    # Set Seed
    print("Seed: {}".format(seed))

    tf.random.set_random_seed(seed)
    #get the training and test data
    train_images, test_images, train_labels, test_labels = preprocessData()


    model = None
    if int(combination)==1:
        model = feed_network(learning_rate)
    if int(combination)==2:
        model = conv_network(learning_rate)
    
    #evaluate the model
    if model == None:
        print("No model created")
        return
    evaluate(model, epochs, batches, train_images, train_labels, test_images, test_labels)

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