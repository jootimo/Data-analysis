import read_yale

from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD

import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

IMG_WIDTH = 168
IMG_HEIGHT = 192

#Load data
pic_tuples = read_yale.load_images_croppedyale() #tuple: pics, fnames, lnames
pics = read_yale.images_to_array(pic_tuples[0], pic_tuples[2])

#Reshape to 2D-array
pixels = []
for pic in pics[0]:
    pixels.append(np.reshape(pic, (IMG_WIDTH, IMG_HEIGHT, 1)))

#Get labels as integers
labels = []
unique_labels = []
prev_label = ""
num = -1
for ix, label in enumerate(pics[1]):
    if label != prev_label:
        prev_label = label
        num = num + 1
        unique_labels.append(num)
        
    labels.append(num)

#Keras model wants categorical labels
labels_categ = to_categorical(labels, num_classes=None)

#Split to training and test sets
pixels = np.array(pixels)
x_train, x_test, y_train, y_test = train_test_split(
    pixels, labels_categ, test_size=0.2)

#Create and compile model
model = Sequential()
model.add(Convolution2D(16, (5,5), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(36, (5,5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=38, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_split=0.2, epochs=40, batch_size=64)

#Do predictions on the test set
predictions_probs = model.predict(x_test)
#Get single array of predicted labels
predictions = []
for p in predictions_probs:
    max_prob = -1
    ix_max_prob = -1
    for ix, prob in enumerate(p):
        if prob > max_prob:
            max_prob = prob
            ix_max_prob = ix

    predictions.append(unique_labels[ix_max_prob])

#Get single array of true labels
true_labels = []
for label_vec in y_test:
    for i in range(len(label_vec)):
        if label_vec[i] == 1:
            true_labels.append(i)
            break

#Save confusion matrix
skplt.metrics.plot_confusion_matrix(true_labels, predictions, text_fontsize='small',
    hide_zeros=True, normalize=True, figsize=(15,15))
plt.savefig("confusion.png")
plt.gcf().clear()

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=10)
#Save history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("acc.png")
plt.gcf().clear()

#Save history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("loss.png")

#from keras.utils import plot_model
#plot_model(model, to_file='model.png')
