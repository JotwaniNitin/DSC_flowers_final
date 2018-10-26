
# coding: utf-8

# In[117]:


#importing packages

import sys
import os
import numpy as np
import argparse
import os
from skimage import io, color, exposure, transform
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


# Using keras and tensorflow as the main libraries for image classification. CNN model has been used for the same.

# In[118]:


DEV = False
argvs = sys.argv
argc = len(argvs)


# declaring the necessary number of epoch for best accuracy

# In[119]:


if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True

if DEV:
  epochs = 2
else:
  epochs = 20


# In[120]:


# parameters

img_width, img_height = 32, 32
batch_size = 32
samples_per_epoch = 1000
validation_steps = 300
nb_filters1 = 32
conv1_size = 5
pool_size = 2
classes_num = 5
lr = 0.0004


# Building the different layers of the CNN Model
# 
# Batch normalization is a technique for improving the performance and stability of artificial neural networks. It is a technique to provide any layer in a neural network with inputs that are zero mean/unit variance.
# 
# Maxpooling is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting.
# 
# Dropout is a regularization technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data. 

# In[121]:


def build_cnn(input_size, num_classes):
    model = Sequential()
    
    model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same", input_shape=(img_width, img_height, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

    model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Flatten())
    
    model.add(Dropout(0.5))
    
    model.add(Dense(256))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.5))
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['accuracy'])
    return model


# In[122]:


model = build_cnn(32,5)

print(model.summary())


# In[123]:


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# In[124]:


test_datagen = ImageDataGenerator(rescale=1. / 255)


# Defining functions to load and train data set. 
# 
# The given data set was divided into two subsets namely--train and test in 0.9-0.1 ratio.
# 
# Further, the validation data set is 90% of the train data set. 

# In[125]:


def load_data(path, val_size=0.1):
    if not os.path.exists(path):
        raise IOError('directory does not exist')
    classes = os.listdir(path)
    class2idx = {c: i for i, c in enumerate(classes)}
    X = []
    y = []
    for c in classes:
        cp = os.path.join(path, c)
        img_paths = os.listdir(cp)
        label = np.zeros(len(classes), dtype=np.float32)
        label[class2idx[c]] = 1.0
        for ip in img_paths:
            img = Image.open(os.path.join(cp, ip))
            img = img.resize((32, 32))
            img = np.array(img, dtype=np.float32) / 255.0
            X.append(img)
            y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size)

    return X_train, X_test, y_train, y_test, class2idx

def load_test(path):
    if not os.path.exists(path):
        raise IOError('directory does not exist')
    classes = os.listdir(path)
    class2idx = {c: i for i, c in enumerate(classes)}
    X = []
    y = []
    for c in classes:
        cp = os.path.join(path, c)
        img_paths = os.listdir(cp)
        label = np.zeros(len(classes), dtype=np.float32)
        label[class2idx[c]] = 1.0
        for ip in img_paths:
            img = Image.open(os.path.join(cp, ip))
            img = img.resize((32, 32))
            img = np.array(img, dtype=np.float32) / 255.0
            X.append(img)
            y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)


    return X,y,class2idx

X_train, X_test, y_train, y_test, class2idx = load_data(path="/home/nitin/Desktop/dsc/train/")
print(class2idx)


# In[127]:


"""
Tensorboard log
"""
log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
callbacks = [tb_cb]


# Epoch running on the training and validation data set. 
# 
# It has been set to 20 epoch after various tests for the best accuracy. 

# In[128]:


model.fit(x=X_train, 
          y=y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks,
          validation_data=(X_test, y_test),
          shuffle=True,
          )


model.save_weights('model.h5')


# In[129]:


target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models/model.h5')
model.save_weights('./models/weights.h5')


# In[130]:


import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model


# In[131]:


img_width, img_height = 32, 32
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)


# In[132]:


model.load_weights('model.h5')

class2idx = {'tulip': 0, 'rose': 1, 'sunflower': 2, 'dandelion': 3, 'daisy': 4}

idx2class = {v: key for key, v in class2idx.items()}


# In[133]:


# loading the test data ..

X_test,Y_test, class2idx = load_test(path="/home/nitin/Desktop/dsc/test/")	
X_test = X_test.reshape(X_test.shape[0],32,32,3)


# In[134]:


score = model.evaluate(X_test, Y_test, verbose=0)


# The accuracy for the test data set is as follows--

# In[135]:


print('Test accuracy:', score[1])


# In[140]:


#noting the correct class for all the predictions made from Y_test
correct_class = []
for i in range(len(Y_test)):
    for j in range(len(Y_test[i])):
        if(Y_test[i][j] == 1.0):
            correct_class.append(j)


# In[141]:


print(correct_class)


# In[142]:


#showing the incorrectly predicted flowers 
from matplotlib import pyplot as plt
for i in range(len(y_pred)):
    if(y_pred[i] != correct_class[i]):
        plt.imshow(X_test[i], interpolation='nearest')
        plt.show()
        print('Predicted as->', idx2class[(y_pred[i])])
        print('Correct class->', idx2class[(correct_class[i])])

