# Regularization

## Important concepts
* L2 regularization
* Dropout
* Early stopping

## Resources
* [Regularization](https://www.youtube.com/watch?v=6g0t3Phly2M&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=4 "Regularization")
* [Dropout](https://www.youtube.com/watch?v=D8PJAL-MZv8&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=6 "Dropout")
* [Early stopping](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf "Early stopping")

## References:
* [numpy.linalg.norm](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.norm.html "numpy.linalg.norm")
* [numpy.random.binomial](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.binomial.html "numpy.random.binomial")
* [tf.losses.get_regularization_loss](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/losses/get_regularization_losses.md "tf.losses.get_regularization_loss")
* [tf.contrib.layers.l2_regularizer](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/contrib/layers/l2_regularizer.md "tf.contrib.layers.l2_regularizer")
* [tf.layers.Dense](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/layers/Dense.md#kernel_regularizer "tf.layers.Dense")
* [tf.layers.Dropout](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/layers/Dropout.md "tf.layers.Dropout")

## Tasks
### [0. L2 Regularization Cost](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-regularization/0-l2_reg_cost.py "0. L2 Regularization Cost")

Calculates the cost of a neural network with L2 regularization

---
### [1. Gradient Descent with L2 Regularization](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-regularization/1-l2_reg_gradient_descent.py "1. Gradient Descent with L2 Regularization")

Performs gradient descent over a neural network with L2 regularization

---
### [2. L2 Regularization Cost](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-regularization/2-l2_reg_cost.py "2. L2 Regularization Cost")

Calculates cost of a neural network with L2 regularization using Tensorflow.

---
### [3. Create a Layer with L2 Regularization](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-regularization/3-l2_reg_create_layer.py "3. Create a Layer with L2 Regularization")

Creates a Tensorflow layer that includes L2 regularization.

---
### [4. Forward Propagation with Dropout](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-regularization/4-dropout_forward_prop.py "4. Forward Propagation with Dropout")

Conducts forward propogation using dropout.

---
### [5. Gradient Descent with Dropout](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-regularization/5-dropout_gradient_descent.py "5. Gradient Descent with Dropout")

Performs gradient descent over a neural network using Dropout regularization.

---
### [6. Create a Layer with Dropout](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-regularization/6-dropout_create_layer.py "6. Create a Layer with Dropout")

Creates a layer of a neural network in Tensorflow using dropout

---
### [7. Early Stopping](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-regularization/7-early_stopping.py "7. Early Stopping")

Determines if gradient descent should be stopped early.
* Early stopping should occur when the validation cost of the network has not decreased relative to the optimal validation cost by more than the threshold over a specific patience count.
