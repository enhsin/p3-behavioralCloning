import os, csv
import numpy as np
from scipy.misc import imread
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Cropping2D
from keras.layers import Lambda
from keras.optimizers import Adam

def splitData(seed, validation_split=0.2):
    np.random.seed(seed)
    img_train = []
    img_valid = []
    y_train = []
    y_valid = []
    with open('data/driving_log.csv','r') as csvfile:
        data = csv.reader(csvfile)
        next(data)  #skip first line
        for row in data:
            if np.random.uniform() < validation_split:
                img_valid.append(row[0])
                y_valid.append(float(row[3]))
            else:
                img_train.append(row[0])
                y_train.append(float(row[3]))
    return img_train, y_train, img_valid, y_valid

def generator(width, height, img, label, batch_size):
    num_samples = len(label)
    while True: # Loop forever so the generator never terminates
        X = np.zeros((batch_size, height, width, 3))
        y = np.zeros((batch_size, 1))
        half_batch = int(batch_size/2)
        for i in range(half_batch):
            index = np.random.choice(num_samples)
            X[i] = imread('data/'+img[index]).astype(np.float32)
            y[i] = label[index]
            X[i+half_batch] = X[i,:,:,::-1]
            y[i+half_batch] = -label[index]
        yield X, y

def createModel(input_l1,input_l2):
    model = Sequential()
    model.add(Cropping2D(cropping=((55,25), (0,0)), input_shape=(input_l1, input_l2, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2,2), border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2,2), border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

img_train, y_train, img_valid, y_valid = splitData(3)
tmp_img = imread('data/'+img_train[0]).astype(np.float32)
width = tmp_img.shape[1]
height = tmp_img.shape[0]

np.random.seed(123)
if os.path.exists('model.h5'):
    model = load_model('model.h5')
else:
    model = createModel(height, width)

# print model information
model.summary()

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=3, verbose=0, mode='auto')
check_callback = ModelCheckpoint('model_v1_{epoch:03d}.h5', monitor='val_loss', save_best_only=True)
train_generator = generator(width,height,img_train,y_train,80)
validation_generator = generator(width,height,img_valid,y_valid,20)
history = model.fit_generator(train_generator, samples_per_epoch=16000, nb_epoch=20,
                              callbacks=[early_stop,check_callback],
                              validation_data=validation_generator, nb_val_samples=4000)
model.save('model.h5')
