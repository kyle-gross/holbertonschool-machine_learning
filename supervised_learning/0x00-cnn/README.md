# Convolutional neural networks (CNNs)

## Important concepts
* Convolutional layers
* Pooling layers
* 2D Conv nets

## Resources
* [DeepAI](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF "Deep AI")
    * [One layer of CNN](https://www.youtube.com/watch?v=jPOAS7uCODQ&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=8 "One layer of CNN")
* [CNN explained](https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8 "CNN explained")
* [CNN explained 2](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721 "CNN explained 2")
* [CNN visual explainaition](https://www.youtube.com/watch?v=YRhxdVk_sIs "CNN visual explainaition")
* [Back-prop for convolutional layer](https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509 "Back-prop for convolutional layer")
* [Back-prop for pooling layer](https://lanstonchu.wordpress.com/2018/09/01/convolutional-neural-network-cnn-backward-propagation-of-the-pooling-layers/ "Back-prop for pooling layer")
* [More back-prop](https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199 "More back-prop")

## References
* [numpy.zeros_like](https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html "numpy.zeros_like")
* [tf.layers.Conv2d](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/layers/Conv2D.md "tf.layers.Conv2d")
* [tf.keras.layers.Conv2d](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/layers/Conv2D.md "tf.keras.layers.Conv2d")
* [tf.layers.AveragePooling2D](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/layers/AveragePooling2D.md "tf.layers.AveragePooling2D")
* [tf.keras.layers.AveragePooling2D](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/layers/AveragePooling2D.md "tf.keras.layers.AveragePooling2D")
* [tf.layers.MaxPooling2D](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/layers/MaxPooling2D.md "tf.layers.MaxPooling2D")
* [tf.keras.layers.MaxPooling2D](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/layers/MaxPool2D.md "tf.keras.layers.MaxPooling2D")
* [tf.layers.Flatten](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/layers/Flatten.md "tf.layers.Flatten")
* [tf.keras.layers.Flatten](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/layers/Flatten.md "tf.keras.layers.Flatten")

## Tasks
### [0. Convolutional Forward Prop](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-cnn/0-conv_forward.py "0. Convolutional Forward Prop")

Performs forward propagation over a convolutional layer of a neural network.

---
### [1. Pooling Forward Prop](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-cnn/1-pool_forward.py "1. Pooling Forward Prop")

Performs forward propagation over a pooling layer of a neural network.

---
### [2. Convolutional Back Prop](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-cnn/2-conv_backward.py "2. Convolutional Back Prop")

Performs back propagation over a convolutional layer of a neural network.

---
### [3. Pooling Back Prop](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-cnn/3-pool_backward.py "3. Pooling Back Prop")

Performs back propagation over a pooling layer of a neural network.

---
### [4. LeNet-5 (Tensorflow)](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-cnn/4-lenet5.py "4. LeNet-5 (Tensorflow)")

Builds a modified version of the LeNet-5 architecture using tensorflow.

---
### [5. LeNet-5 (Keras)](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-cnn/5-lenet5.py "5. LeNet-5 (Keras)")

Builds a modified version of the LeNet-5 architecture using keras.
