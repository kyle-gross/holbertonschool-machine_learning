#!/usr/bin/env python3

import matplotlib.pyplot as plt
import tensorflow.keras as K
import tensorflow as tf
preprocess_data = __import__('0-transfer').preprocess_data

# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase 

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)

# plt.figure(1, figsize=(15, 8))

# plt.subplot(221)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'valid'])

# plt.subplot(222)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model_loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'valid'])

# plt.show()
