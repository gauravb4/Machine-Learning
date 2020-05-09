import numpy as np
import pandas as pd
import PIL.Image as Image
import os
from os.path import isfile
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, Activation, GRU
from keras.layers import Conv2D, concatenate, MaxPooling2D, Flatten, Embedding, Lambda

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop

from keras import regularizers

import matplotlib.pyplot as plt

dict_genres = {'Electronic':0, 'Experimental':1, 'Folk':2, 'Hip-Hop':3, 
               'Instrumental':4,'International':5, 'Pop' :6, 'Rock': 7  }


reverse_map = {v: k for k, v in dict_genres.items()}
print(reverse_map)


n_rows = 15996
skip = np.arange(n_rows)
skip = np.delete(skip, np.arange(0, n_rows, 2))

f = pd.read_csv("genres_small.csv", header=0, skiprows=skip)
p = f.shape
files = f["File"]
labels = f["Label"]
mapping = {}
for index, file in enumerate(files):
    mapping[file] = labels[index]

dir  = os.getcwd() + "\\fma_small_img"
l = os.listdir(dir)

train = []
test = []
count = 0
for filepath in l:
    img = Image.open(dir + "\\" + filepath).convert('L')
    np_img = np.asarray(img)
    h = np.hstack(np_img)
    if(count < 6000):
        train.append(h)
    else:
        test.append(h)
    del np_img
    count += 1