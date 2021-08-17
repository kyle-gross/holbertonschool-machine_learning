# Tensorflow

## Important topics:
* Tensors
* Graphs
* Sessions
* Variables
* Placeholders
* Save/Restore

## Resources:
* [Intro to Tensorflow](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/low_level_intro.md "Intro to Tensorflow")
* [Tensor](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/Tensor.md "Tensor")
* [Graphs](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/graphs.md "Graphs")
* [Sessions](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/Session.md#run "Sessions")
* [Variables](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/variables.md "Variables")
* [Placeholders](https://databricks.com/tensorflow/placeholders "Placeholders")
* [Save/Restore](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/saved_model.md "Save/Restore")
    * [Import MetaGraph](https://stackoverflow.com/questions/42072234/tensorflow-import-meta-graph-and-use-variables-from-it "Import MetaGraph")
    * [Export/Import MetaGraph](https://docs.w3cub.com/tensorflow~python/meta_graph "Export/Import MetaGraph")

## Tasks:
[0. Placeholders](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-tensorflow/0-create_placeholders.py "0. Placeholders")
* Create two placeholders, `x` and `y`, for the neural network.

[1. Layers](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-tensorflow/1-create_layer.py "1. Layers")
* Creates a single layer for the neural network.

[2. Forward Propagation](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-tensorflow/2-forward_prop.py "2. Forward Propagation")
* Creates the forward propagation graph for the neural network.

[3. Accuracy](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-tensorflow/3-calculate_accuracy.py "3. Accuracy")
* Determines accuracy of neural network.

[4. Loss](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-tensorflow/4-calculate_loss.py "4. Loss")
* Calculates loss of neural network. Uses `tf.losses.softmax_cross_entropy()`.

[5. Train_Op](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-tensorflow/5-create_train_op.py "5. Train Op")
* Creates the training operation for the neural network. Uses gradient descent as optimizer.

[6. Train](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-tensorflow/6-train.py "6. Train")
* Uses previously created functions to create, forward_propagate, and optimize.
* Trains the neural network over `iterations`.
* Saves trained model to `save_path`.

[7. Evaluate](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-tensorflow/7-evaluate.py "7. Evaluate")
* Evaluates the output of the neural network.
