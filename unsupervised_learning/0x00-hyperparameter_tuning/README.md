# Hyperparameter Tuning

## Important Concepts
* Gaussian Processes
* Bayesian Optimization

## Resources
* [Introduction to Gaussian processes](https://www.youtube.com/watch?v=4vGiHC35j9s "Introduction to Gaussian processes")
* [Gaussian processes](https://www.youtube.com/watch?v=MfHKW5z-OOA "Gaussian processes")
* [UBC Gaussian Processes](https://www.cs.ubc.ca/~hutter/EARG.shtml/earg/papers05/rasmussen_gps_in_ml.pdf "UBC Gaussian Processes")
* [Kriging](https://en.wikipedia.org/wiki/Kriging "Kriging")
* [GPy](https://gpy.readthedocs.io/en/latest/ "GPy")

## References
* []( "")

## Tasks
### [0. Initialize Gaussian Process](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-hyperparameter_tuning/0-gp.py "0. Initialize Gaussian Process")

Creates a class that represents a noiseless 1D Gaussian process.

---
### [1. Gaussian Process Prediction](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-hyperparameter_tuning/1-gp.py "1. Gaussian Process Prediction")

Add a public instance attribute, predict(), to the class created in task 0. It predicts the mean and standard deviation of points in a Gaussian Process.

---
### [2. Update Gaussian Process](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-hyperparameter_tuning/2-gp.py "2. Update Gaussian Process")

Add a public instance attribute, update(), to the GaussianProcess class.

---
### [3. Initialize Bayesian Optimization](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-hyperparameter_tuning/3-bayes_opt.py "3. Initialize Bayesian Optimization")

Creates the class BayesianOptimization which performs Bayesian optimization on a noiseless 1D Gaussian process.

---
### [4. Bayesian Optimization - Acquisition](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-hyperparameter_tuning/4-bayes_opt.py "4. Bayesian Optimization - Acquisition")

Add a public instance method, acquisition(), that calculates the next best sample location.

---
### [5. Bayesian Optimization](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-hyperparameter_tuning/5-bayes_opt.py "5. Bayesian Optimization")

Add a public instance method, optimize(), that optimizes the black-box function.
