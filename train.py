from models.simple_models import train_and_eval
from models.nn import train_and_test_nn
import pandas as pd
import tensorflow as tf

if __name__ == '__main__':
    data = pd.read_csv("./data/covtype.csv")
    tf.random.set_seed(123)
    train_and_eval(data)
    train_and_test_nn(data)
