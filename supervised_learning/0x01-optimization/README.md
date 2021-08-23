# Optimization

## Important concepts:
* Hyperparameters
* Regularization
* Normaliziation
* Batch normalization
* Learning rate decay
* Mini-batch gradient descent
* Gradient descent optimzation algorithms
    * Momentum
    * RMSprop
    * Adam

## Resources:
* [Hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning) "Hyperparameter")
* [Regularization](https://www.youtube.com/watch?v=6g0t3Phly2M&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=4 "Regularization")
* [Normalizing inputs](https://www.youtube.com/watch?v=FDCfw-YqWTE&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=9 "Normalizing inputs")
* [Batch normalization](https://www.youtube.com/watch?v=tNIpEZLv_eg&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=27 "Batch normalization")
    * [tf.nn.batch_normalization](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/nn/batch_normalization.md "tf.nn.batch_normalization")
    * [tf.nn.moments](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/nn/moments.md "tf.nn.moments")
* [Learning rate decay](https://www.youtube.com/watch?v=QzulmoOg2JE&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=24 "Learning rate decay")
    * [tf.train.inverse_time_decay](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/train/inverse_time_decay.md "tf.train.inverse_time_decay")
* [Mini-batch gradient descent](https://www.youtube.com/watch?v=4qJaSmvhxi8&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=15 "Mini-batch gradient descent")
    * [Understanding mini-batch](https://www.youtube.com/watch?v=-_4Zi8fCZO4&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=16 "Understanding mini-batch")
* [Momentum optimization](https://www.youtube.com/watch?v=k8fTYJPd3_I&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=20 "Momentum optimization")
    * [tf.train.MomentumOptimizer](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/train/MomentumOptimizer.md "tf.train.MomentumOptimizer")
* [RMSprop optimization](https://www.youtube.com/watch?v=_e-LFe_igno&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=21 "RMSprop optimization")
    * [tf.train.RMSPropOptimizer](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/train/RMSPropOptimizer.md "tf.train.RMSPropOptimizer")
* [Adam optimization](https://www.youtube.com/watch?v=JXQT_vxqwIs&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=22 "Adam optimization")
    * [tf.train.AdamOptimizer](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/train/AdamOptimizer.md "tf.train.AdamOptimizer")

## Tasks:
[0. Normalization Constants](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/0-norm_constants.py "0. Normalization Constants")

Calculates normalization constants of a matrix.
* mean / std dev

---
[1. Normalize](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/1-normalize.py "1. Normalize")

Normalizes a matrix.
* (X - mean) / std_dev

---
[2. Shuffle Data](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/2-shuffle_data.py "2. Shuffle Data")

Shuffles data points in two matricies.

---
[3. Mini-Batch](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/3-mini_batch.py "3. Mini-Batch")

Loads then trains a neural network model using mini-batch gradient descent.

---
[4. Moving Average](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/4-moving_average.py "4. Moving Average")

Calculates the exponentially weighted (moving) average of a data set with bias correction.

---
[5. Momentum](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/5-momentum.py "5. Momentum")

Updates a variable using the gradient descent with momentum algorithm.

---
[6. Momentum Upgraded](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/6-momentum.py "6. Momentum Upgraded")

Creates training operation for a neural network in Tensorflow. Uses gradient descent with momentum optimization algorithm.

---
[7. RMSProp](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/7-RMSProp.py "7. RMSProp")

Updates a variable using the RMSProp optimization algorithm.

---
[8. RMSProp Upgraded](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/8-RMSProp.py "8. RMSProp Upgraded")

Creates training operation for a neural network in Tensorflow. Uses RMSProp optimization algorithm.

---
[9. Adam](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/9-Adam.py "9. Adam")

Updates a variable using the Adam optimization algorithm. Adam optimization is a combination of momentum and RMSprop optimization.

---
[10. Adam Upgraded](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/10-Adam.py "10. Adam Upgraded")

Creates training operation for a neural network in Tensorflow. Uses Adam optimization algorithm.

---
[11. Learning Rate Decay](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/11-learning_rate_decay.py "11. Learning Rate Decay")

Updates the learning rate (alpha) using inverse time decay.

---
[12. Learning Rate Decay Upgraded](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/12-learning_rate_decay.py "12. Learning Rate Decay Upgraded")

Creates a learning rate decay operation in Tensorflow using inverse_time_decay().

---
[13. Batch Normalization](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/13-batch_norm.py "13. Batch Normalization")

Normalizes unactivated output of a neural network using batch normalization.

---
[14. Batch Normalization Upgraded](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/14-batch_norm.py "14. Batch Normalization Upgraded")

Creates a batch normalization layer for a neural network using Tensorflow.

---
[15. Put it all together and what do you get?](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-optimization/15-model.py "15. Put it all together and what do you get?")

Builds, trains, and saves a neural network model in tensorflow using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization.
