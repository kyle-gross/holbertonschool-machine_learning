# Error Analysis

## Important concepts:
* Confusion matrices
    * How to read, measure, and calculate
* Bias and variance

## Resources
* [Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix "Confusion matrix")
    * [Confusion matrix equations](https://newbedev.com/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-false-negative "Confusion matrix equations")
* [Bias-variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff "Bias-variance tradeoff")
* [Bias/variance](https://www.youtube.com/watch?v=SjQyLhQIXSM&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=3 "Bias/variance")
    * [Basic recipie for ML](https://www.youtube.com/watch?v=C1N_PDHuJ6Q&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=3 "Basic recipie for ML")
* [Human-level performance](https://www.youtube.com/watch?v=J3HHOwcrkK8&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b&index=9 "Human-level performance")

## Tasks
### [0. Create Confusion](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-error_analysis/0-create_confusion.py "0. Create Confusion")

Create a confusion matrix out of OH numpy.ndarrays. Uses np.matmul()

---
### [1. Sensitivity](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-error_analysis/1-sensitivity.py "1. Sensitivity")

Calculates the sensitivity (recall) of a confusion matrix.

---
### [2. Precision](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-error_analysis/2-precision.py "2. Precision")

Calculates the precision (PPV) for each class in a confusion matrix.

---
### [3. Specificity](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-error_analysis/3-specificity.py "3. Specificity")

Calculates the specificity (TNR, true negative rate) for each class in a confusion matrix.

---
### [4. F1 score](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-error_analysis/4-f1_score.py "4. F1 score")

Calculates the F1 score of a confusion matrix using PPV and TPR.

---
### [5. Dealing with Error](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-error_analysis/5-error_handling "5. Dealing with Error")

This is a text file containing multiple choice answers.
Scenarios:
```
1. High Bias, High Variance
2. High Bias, Low Variance
3. Low Bias, High Variance
4. Low Bias, Low Variance
```
Approaches:
```
A. Train more
B. Try a different architecture
C. Get more data
D. Build a deeper network
E. Use regularization
F. Nothing
```

---
### [6. Compare and Contrast](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-error_analysis/6-compare_and_contrast "6. Compare and Contrast")

Text file containing a single multiple choice answer.