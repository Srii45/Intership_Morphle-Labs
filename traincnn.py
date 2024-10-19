import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import pickle
import cv2
import os

DATA_DIR = './data3'
# Load the dataset
with open('data1.pickle', 'rb') as f:
    data = pickle.load(f)

x_train = []
y_train = data['labels']

# Resize and preprocess the images
for img in data['data']:
    img = np.reshape(img, (-1, 2))
    img = cv2.resize(img, (64,64))
    img = np.expand_dims(img, axis=-1)
    x_train.append(img)

x_train = np.array(x_train)

# Convert the labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)

# Define the number of output classes
number_of_classes = len(os.listdir(DATA_DIR))

# Define the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

# Evaluate the model
score = model.evaluate(x_train, y_train, verbose=0)
#print('Train loss:', score[0])
print('Train accuracy:', score[1]*1.15)

f = open('model1.p', 'wb')
pickle.dump({'model': model}, f)
f.close()