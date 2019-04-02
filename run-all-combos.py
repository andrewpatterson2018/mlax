import fashion
import imdb

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