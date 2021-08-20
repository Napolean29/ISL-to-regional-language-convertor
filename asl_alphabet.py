# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import tensorflow as tf


# %%
tf.__version__


# %%
# get_ipython().system('git clone https://github.com/nicknochnack/RealTimeObjectDetection.git')


# %%
# print(os.listdir("/Users/napolean/Downloads/ASL_alphabet/asl_alphabet_train/asl_alphabet_train"))


# %%
from tensorflow import keras


# %%
# from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization
# from keras.utils import np_utils
# from keras.utils.np_utils import to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight


# %%
# from tensorflow.keras.optimizers import SGD,RMSprop, Adam, Adagrad, Adadelta, RMSprop


# %%
# from keras.models import Sequential, model_from_json


# %%
# from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
# from keras import backend as K
# from keras.applications.vgg16 import VGG16
# from keras.models import Model
# from keras.applications.inception_v3 import InceptionV3


# %%
from tqdm import tqdm
from sklearn import model_selection
from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# %%
import matplotlib.pyplot as plt


# %%
from glob import glob


# %%
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns
import zlib
import itertools
import sklearn
import itertools
import scipy
import skimage
from skimage.transform import resize
import os


# %%
import numpy as np
batch_size = 64
from skimage import io

imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 29

train_len = 87000
train_dir ='/Users/napolean/Downloads/ASL_alphabet/asl_alphabet_train/asl_alphabet_train/'

def get_data(folder):
    X = np.empty((train_len, imageSize, imageSize, 3), dtype=np.float32)
    y = np.empty((train_len,), dtype=np.int)
    cnt = 0
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['A']:
                label = 0
            elif folderName in ['B']:
                label = 1
            elif folderName in ['C']:
                label = 2
            elif folderName in ['D']:
                label = 3
            elif folderName in ['E']:
                label = 4
            elif folderName in ['F']:
                label = 5
            elif folderName in ['G']:
                label = 6
            elif folderName in ['H']:
                label = 7
            elif folderName in ['I']:
                label = 8
            elif folderName in ['J']:
                label = 9
            elif folderName in ['K']:
                label = 10
            elif folderName in ['L']:
                label = 11
            elif folderName in ['M']:
                label = 12
            elif folderName in ['N']:
                label = 13
            elif folderName in ['O']:
                label = 14
            elif folderName in ['P']:
                label = 15
            elif folderName in ['Q']:
                label = 16
            elif folderName in ['R']:
                label = 17
            elif folderName in ['S']:
                label = 18
            elif folderName in ['T']:
                label = 19
            elif folderName in ['U']:
                label = 20
            elif folderName in ['V']:
                label = 21
            elif folderName in ['W']:
                label = 22
            elif folderName in ['X']:
                label = 23
            elif folderName in ['Y']:
                label = 24
            elif folderName in ['Z']:
                label = 25
            elif folderName in ['del']:
                label = 26
            elif folderName in ['nothing']:
                label = 27
            elif folderName in ['space']:
                label = 28           
            else:
                label = 29
            for image_filename in os.listdir(folder + folderName):
                img_file = io.imread(folder + folderName +'/'+ image_filename,plugin='matplotlib')
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file).reshape((-1, imageSize, imageSize, 3))
                    
                    X[cnt] = img_arr
                    y[cnt] = label
                    cnt += 1
    return X,y
X_train, y_train = get_data(train_dir)


# %%
X_train.shape


# %%
# plt.imshow(X_train[789])
# plt.show()


# %%
y_train[789]


# %%
#making copies of original data
X_data = X_train
y_data = y_train


# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X_data,y_data,test_size=0.25,random_state=23,stratify=y_data)


# %%
from tensorflow.keras.utils import to_categorical
#one_hot_encoding
y_cat_train = to_categorical(y_train,29)
y_cat_test = to_categorical(y_test,29)


# %%
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_cat_train.shape)
print(y_cat_test.shape)


# %%
# from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
# from keras.models import Sequential


# %%
model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(29,activation ='softmax')
    ])
model.summary()


# %%
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=2)


# %%
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# %%
model.fit(X_train,y_cat_train,epochs = 50,batch_size = 64,verbose = 2, validation_data=(X_test,y_cat_test),callbacks=[early_stop])


# %%
import pandas as pd
metrics = pd.DataFrame(model.history.history)
print("The model metrics are")
metrics


# %%
metrics[['accuracy','val_accuracy']].plot()
plt.show()


# %%
model.evaluate(X_test,y_cat_test,verbose=0)


# %%
predictions = model.predict_classes(X_test)
print("Predictions done...")


# %%
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))


# %%



# %%



# %%



