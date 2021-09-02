# Keras

## Important concepts
* Keras library
* Activations, regularization, and evaluation using Keras

## Resources
* [Sequential model](https://www.tensorflow.org/guide/keras/sequential_model "Sequential model")
* [Functional API](https://www.tensorflow.org/guide/keras/functional "Functional API")
* [Optimizers](https://keras.io/api/optimizers/ "Optimizers")

## References
* [tf.keras.models.Model](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/models/Model.md "tf.keras.models.Model")
* [tf.keras.layers.Dense](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/layers/Dense.md "tf.keras.layers.Dense")
* [tf.keras.regularizers.l2](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/regularizers/l2.md "tf.keras.regularizers.l2")
* [tf.keras.optimizers.Adam](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/optimizers/Adam.md "tf.keras.optimizers.Adam")
* [tf.keras.callbacks.EarlyStopping](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/callbacks/EarlyStopping.md "tf.keras.callbacks.EarlyStopping")
* [tf.keras.callbacks.LearningRateScheduler](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/callbacks/LearningRateScheduler.md "tf.keras.callbacks.LearningRateScheduler")

## Tasks
### [0. Sequential](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-keras/0-sequential.py "0. Sequential")

Builds a neural network using the Keras library using the Sequential() class.

---
### [1. Input](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-keras/1-input.py "1. Input")

Builds a neural network using the Keras library without using the Sequential() class.

---
### [2. Optimize](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-keras/2-optimize.py "2. Optimize")

Function that sets up Adam optimization for a Keras model with categorical crossentropy loss and accuracy metrics.

---
### [3. One Hot](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-keras/3-one_hot.py "3. One Hot")

Converts a lebel vector into a one-hot matrix.

---
### [4. Train](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-keras/4-train.py "4. Train")

Trains a Keras model using mini-batch gradient descent.

---
### [5. Validate](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-keras/5-train.py "5. Validate")

Updated version of previous task, add analyzation of validation data.

---
### [6. Early Stopping](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-keras/6-train.py "6. Early Stopping")

Updated version of previous task, trains model using early stopping.

---
### [7. Learning Rate Decay](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-keras/7-train.py "7. Learning Rate Decay")

Updated version of previous task, implements learning rate decay.

---
### [8. Save Only the Best](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-keras/8-train.py "8. Save Only the Best")

Updated version of previous task, saves model after the optimal epoch.

---
### [9. Save and Load Model](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-keras/9-model.py "9. Save and Load Model")

Contains two functions, one which saves and one which loads a model.

---
### [10. Save and Load Weights](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-keras/10-weights.py "10. Save and Load Weights")

Contains two functions, one which saves and one which loads a model's weights.

---
### [11. Save and Load Configuration](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-keras/11-config.py "11. Save and Load Configuration")

Contains two functions, one which saves and one which loads a model's configuration. Saves to JSON format.

---
### [12. Test](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-keras/12-test.py "12. Test")

Tests a network loaded from a Keras model.

---
### [13. Predict](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-keras/13-predict.py "13. Predict")

Makes a prediction using a neural network made with Keras.
