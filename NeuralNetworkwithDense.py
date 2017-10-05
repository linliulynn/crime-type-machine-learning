from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)

import numpy as np
import keras
import pandas as pd
from keras.layers import Dense, Dropout, Input, Embedding, Flatten, LSTM
from keras.models import Model
from keras.optimizers import SGD
import matplotlib.pylab as plt
from sklearn.metrics import classification_report
import itertools
import collections
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix

crimes = pd.read_csv('selected_features_2012_to_2017.csv',error_bad_lines=False)

crimes.index = pd.DatetimeIndex(crimes.Date)

crimes_2012_2016 = crimes.loc['2012':'2016']
crimes_2017 = crimes.loc['2017']
print('Number of observations in the training data:', len(crimes_2012_2016))
print('Number of observations in the test data:',len(crimes_2017))

# #seperate features into sparse features and non-sparse features
# features_train = crimes_2012_2016[["hour", "Day of Week", "Primary Type in number", "Community Area", "Business Hour",
#                              "Business Day"]]
# part_features_train = crimes_2012_2016[["hour", "Day of Week", "Community Area", "Business Hour",
#                              "Business Day"]]
# one_features_train = crimes_2012_2016["Primary Type in number"]
# y = crimes_2012_2016["Location Description Number"]
# y_train = keras.utils.to_categorical(y, num_classes=4)
#
#
# features_test = crimes_2017[["hour", "Day of Week", "Primary Type in number", "Community Area", "Business Hour",
#                              "Business Day"]]
# part_features_test = crimes_2017[["hour", "Day of Week", "Community Area", "Business Hour",
#                              "Business Day"]]
# one_features_test = crimes_2017["Primary Type in number"]
# y_2017 = crimes_2017["Location Description Number"]
# y_test = keras.utils.to_categorical(y_2017, num_classes=4)
# print(y_2017)
# print(y_test)
#
# time_start = time.clock()
#
# #build neural network model1 with one dense embedding#
# size = crimes_2012_2016["Primary Type in number"].size
# input = Input(shape=(1,), name='input')
# x = Embedding(output_dim=4, input_dim=33, input_length=1)(input)
# x = Flatten()(x)
# #x = Dense(4, activation='relu')(x)
#
# part_input = Input(shape=(5,), name='part_input')
# x = keras.layers.concatenate([x, part_input])
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
# output = Dense(4, activation='softmax')(x)
#
# model = Model(inputs=[input, part_input], outputs=[output])
#
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])
#
# callbacks = [
#     EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=0, verbose=0),
#     ModelCheckpoint(filepath='kfold_weights_path', monitor='val_loss', save_best_only=True, verbose=0),
# ]
# model.fit([np.array(one_features_train), np.array(part_features_train)], [np.array(y_train)],
#           epochs=20, batch_size=128, validation_data=([np.array(one_features_test), np.array(part_features_test)], np.array(y_test)), callbacks=callbacks)
#
# y_pred = model.predict([np.array(one_features_test), np.array(part_features_test)], batch_size=32, verbose=0)
# print(y_pred.shape)
# y_pred = np.argmax(y_pred,axis=1)
# print(y_pred)
# print(y_2017)
# print(collections.Counter(y_pred))
# y_pred = np.array([y_pred])
# y_pred = y_pred.transpose()


# build neural network model2,two dense features
#seperate features into sparse features and non-sparse features
features_train = crimes_2012_2016[["hour", "Day of Week", "Primary Type in number", "Community Area", "Business Hour",
                             "Business Day"]]
part_features_train = crimes_2012_2016[["hour", "Day of Week", "Business Hour",
                             "Business Day"]]
first_feature_train = crimes_2012_2016["Primary Type in number"]
second_feature_train = crimes_2012_2016["Community Area"]

y = crimes_2012_2016["Location Description Number"]
y_train = keras.utils.to_categorical(y, num_classes=4)

features_test = crimes_2017[["hour", "Day of Week", "Primary Type in number", "Community Area", "Business Hour",
                             "Business Day"]]
part_features_test = crimes_2017[["hour", "Day of Week", "Business Hour",
                             "Business Day"]]
first_feature_test = crimes_2017["Primary Type in number"]
second_feature_test = crimes_2017["Community Area"]
y_2017 = crimes_2017["Location Description Number"]
y_test = keras.utils.to_categorical(y_2017, num_classes=4)
print(y_2017)
print(y_test)

time_start = time.clock()

#neural network model#
size = crimes_2012_2016["Primary Type in number"].size
input = Input(shape=(1,), name='input')
x = Embedding(output_dim=4, input_dim=33, input_length=1)(input)
x = Flatten()(x)
#x = Dense(4, activation='relu')(x)

input2 = Input(shape=(1,), name='input2')
x2 = Embedding(output_dim=4, input_dim=79, input_length=1)(input2)
x2 = Flatten()(x2)

# activation function and nodes in every layer
part_input = Input(shape=(4,), name='part_input')
x = keras.layers.concatenate([x, x2, part_input])
x = Dense(64, activation='tanh')(x)
x = Dropout(0.6)(x)
x = Dense(256, activation='tanh')(x)
x = Dropout(0.6)(x)
x = Dense(64, activation='tanh')(x)
x = Dropout(0.6)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=[input, input2, part_input], outputs=[output])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# early stop
callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=0, verbose=0),
    ModelCheckpoint(filepath='kfold_weights_path', monitor='val_loss', save_best_only=True, verbose=0),
]
model.fit([np.array(first_feature_train), np.array(second_feature_train), np.array(part_features_train)], [np.array(y_train)],
          epochs=20, batch_size=128, validation_data=([np.array(first_feature_test), np.array(second_feature_test), np.array(part_features_test)], np.array(y_test)), callbacks=callbacks)

y_pred = model.predict([np.array(first_feature_test), np.array(second_feature_test), np.array(part_features_test)], batch_size=32, verbose=0)
print(y_pred.shape)
y_pred = np.argmax(y_pred,axis=1)
print(y_pred)
print(y_2017)
print(collections.Counter(y_pred))
y_pred = np.array([y_pred])
y_pred = y_pred.transpose()


# compute time to build a model
time_elapsed = (time.clock() - time_start)
print("time to build a model is:", time_elapsed)


class_names = [1,2,3]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_2017, y_pred)
np.set_printoptions(precision=4)

# compute classification report
classificationReport = classification_report(y_2017, y_pred, digits=5)
# classificationReport = classificationReport.to_array()
# np.set_printoptions(precision=2)
print(classificationReport)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# plot_model(model, to_file='model.png')
