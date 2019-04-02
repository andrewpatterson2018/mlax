import argparse
import time
import keras
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from time import time
import numpy
from keras.callbacks import TensorBoard
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

def GetData():
    (trainX, trainY), (testX, testY) = imdb.load_data(path='imdb.pkl', num_words = 5000, seed=12345)
    return trainX, testX, trainY, testY

def check_param_is_numeric(param, value):

    try:
        value = float(value)
    except:
        print("{} must be numeric".format(param))
        quit(1)
    return value


def cone(combination, _learning_rate, _epochs, _batches, _seed):
    trainX, testX, trainY, testY = GetData()
    
    max_review_length = 500
    top_words = 5000
    embedding_vecor_length = 128

    trainX = sequence.pad_sequences(trainX, maxlen=max_review_length)
    testX = sequence.pad_sequences(testX, maxlen=max_review_length)

    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(128, dropout=0.8))
    model.add(Dense(2, activation='softmax', input_dim=1))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #the first parameter is the batches, which is set to none
    #net = tflearn.input_data([batches, 100])
    #net = tflearn.embedding(net, input_dim=5000, output_dim=128)
    #below is setting the values to lstm layer, dropout to 0.8 to prevent overfitting
    #net = tflearn.lstm(net, 128, dropout=0.8)
    #net = tflearn.fully_connected(net, 2, activation='softmax')
    #net = tflearn.regression(net, optimizer='adam', learning_rate=_learning_rate,
    #                       loss='categorical_crossentropy')

    # tensorboard
    boardString = 'logs/imdb-{:d}-{:.3f}-{:d}-{:d}-{:d}.cpkt'.format(int(combination), _learning_rate, int(_epochs), int(_batches), int(_seed))
    tensorboard = TensorBoard(log_dir=boardString, histogram_freq=2, write_grads=True)
    
    #training
    #model =tflearn.DNN(net, tensorboard_verbose=0)
    print(model.summary())
    model.fit(trainX, trainY, epochs=int(_epochs), batch_size=int(_batches), callbacks=[tensorboard], validation_data=(testX,testY))

    test_loss, test_acc = model.evaluate(testX, testY)
    print('Test accuracy:', test_acc)

    model.save('imdb-{:d}-{:.3f}-{:d}-{:d}-{:d}.cpkt'.format(int(combination), _learning_rate, int(_epochs), int(_batches), int(_seed)))


def ctwo(combination, _learning_rate, _epochs, _batches, _seed):
    trainX, testX, trainY, testY = GetData()
    max_review_length = 500
    top_words = 5000
    trainX = sequence.pad_sequences(trainX, maxlen=max_review_length)
    testX = sequence.pad_sequences(testX, maxlen=max_review_length)

    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # tensorboard
    boardString = 'logs/imdb-{:d}-{:.3f}-{:d}-{:d}-{:d}.cpkt'.format(int(combination), _learning_rate, int(_epochs), int(_batches), int(_seed))
    tensorboard = TensorBoard(log_dir=boardString, histogram_freq=2, write_grads=True)
    
    model.fit(trainX, trainY, epochs=int(_epochs), batch_size=int(_batches), callbacks=[tensorboard], validation_data=(testX,testY))
    # Final evaluation of the model
    test_loss, test_acc = model.evaluate(testX, testY, verbose=0)
    print('Test accuracy:', test_acc)

    model.save('imdb-{:d}-{:.3f}-{:d}-{:d}-{:d}.cpkt'.format(int(combination), _learning_rate, int(_epochs), int(_batches), int(_seed)))

   
def cthree(combination, _learning_rate, _epochs, _batches, _seed):
    trainX, testX, trainY, testY = GetData()
    
    max_review_length = 500
    top_words = 5000
    embedding_vecor_length = 64

    trainX = sequence.pad_sequences(trainX, maxlen=max_review_length)
    testX = sequence.pad_sequences(testX, maxlen=max_review_length)
    
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # tensorboard
    boardString = 'logs/imdb-{:d}-{:.3f}-{:d}-{:d}-{:d}.cpkt'.format(int(combination), _learning_rate, int(_epochs), int(_batches), int(_seed))
    tensorboard = TensorBoard(log_dir=boardString, histogram_freq=2, write_grads=True)

    model.fit(trainX, trainY, epochs=int(_epochs), batch_size=int(_batches), callbacks=[tensorboard], validation_data=(testX,testY))
    # Final evaluation of the model
    test_loss, test_acc = model.evaluate(testX, testY, verbose=0)
    print('Test accuracy:', test_acc)

    model.save('imdb-{:d}-{:.3f}-{:d}-{:d}-{:d}.cpkt'.format(int(combination), _learning_rate, int(_epochs), int(_batches), int(_seed)))


def main(combination, learning_rate, epochs, batches, seed):
    # Set Seed
    numpy.random.seed(int(seed))
    tf.random.set_random_seed(seed)

    print("Seed: {}".format(seed))

    if int(combination)==1:
        cone(combination, learning_rate, epochs, batches, seed)
    if int(combination)==2:
        ctwo(combination, learning_rate, epochs, batches, seed)
    if int(combination)==3:
        cthree(combination, learning_rate, epochs, batches, seed)


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