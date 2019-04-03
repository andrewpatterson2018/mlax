import sys, fashion, imdb


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("console-output.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    


if __name__ == "__main__":
    sys.stdout = Logger()
    # FASHION MNIST RUNS
    fashion.main(4,0.05,15,128,12345)
    fashion.main(4,0.01,15,256,12345)
    fashion.main(4,0.1,15,512,12345)
    fashion.main(2,0.05,15,128,12345)
    fashion.main(2,0.01,15,256,12345)
    fashion.main(2,0.1,15,512,12345)

    # IMDB RUNS
    imdb.main(3, 0.001, 15, 64, 12345)
    imdb.main(3, 0.1, 15, 128, 12345)
    imdb.main(3, 0.2, 15, 256, 12345)
    imdb.main(2, 0.001, 15, 64, 12345)
    imdb.main(2, 0.1, 15, 128, 12345)
    imdb.main(2, 0.2, 15, 256, 12345)