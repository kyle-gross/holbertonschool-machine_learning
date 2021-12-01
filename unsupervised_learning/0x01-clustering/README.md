# Clustering

## Important Concepts
* K-means
* Bimodal Distribution
* Mixture Models
* Expectation Maximum (EM)
* Hierarchical Clustering
* Scikit-learn

## Resources
* [Understanding K-means Clustering in Machine Learning](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1 "Understanding K-means Clustering in Machine Learning")
* [Steps to calculate centroids in cluster using K-means](https://www.datasciencecentral.com/profiles/blogs/steps-to-calculate-centroids-in-cluster-using-k-means-clustering "Steps to calculate centroids in cluster using K-means")
* [K-means clustering: how it works](https://www.youtube.com/watch?v=_aWzGGNrcic "K-means clustering: how it works")
* [K-means in numpy](https://nbviewer.org/github/flothesof/posts/blob/master/20150717_Kmeans.ipynb "K-means in numpy")
* [Clustering: how many clusters?](https://www.youtube.com/watch?v=xNfOheh-res "Clustering: how many clusters?")
* [Gaussian Mixture Model](https://brilliant.org/wiki/gaussian-mixture-model/ "Gaussian Mixture Model")
* [EM algorithm: how it works](https://www.youtube.com/watch?v=REypj2sy_5U "EM algorithm: how it works")
* [EM Algorithm](https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html "EM Algorithm")
* [EM Algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm "EM Algorithm")
* [Beginners Guide to Hierarchical Clustering](https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/ "Beginners Guide to Hierarchical Clustering")

## References
* [`scikit-learn`](https://scikit-learn.org/stable/index.html "scikit-learn")
* [`sklearn.cluster`](https://scikit-learn.org/stable/modules/clustering.html#clustering "sklearn.cluster")
* [`sklearn.cluster.KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans "sklearn.cluster.KMeans")
* [Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html#mixture "Gaussian mixture models")
* [`sklearn.mixture.GaussianMixture`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture "sklearn.mixture.GaussianMixture")
* [SciPy](https://scipy.org/ "SciPy")
* [`scipy.cluster.hierarchy`](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html "scipy.cluster.hierarchy")
    * [`scipy.cluster.hierarchy.linkage`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage "scipy.cluster.hierarchy.linkage")
    * [`scipy.cluster.hierarchy.fcluster`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster "scipy.cluster.hierarchy.fcluster")
    * [`scipy.cluster.hierarchy.dendrogram`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html#scipy.cluster.hierarchy.dendrogram "scipy.cluster.hierarchy.dendrogram")
* [`numpy.random.uniform`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html "numpy.random.uniform")

## Tasks
### [0. Initialize K-means](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x01-clustering/0-initialize.py "0. Initialize K-means")

Initializes cluster centroids for K-means.

---
### [1. K-means](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x01-clustering/1-kmeans.py "1. K-means")

Performs K-means on a dataset.

---
### [2. Variance](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x01-clustering/2-variance.py "2. Variance")

Calculates the total intra-cluster variance for a data set.

---
### [3. Optimize k](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x01-clustering/3-optimum.py "3. Optimize k")

Tests for the optimum number of clusters by variance.

---
### [4. Initialize GMM](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x01-clustering/4-initialize.py "4. Initialize GMM")

Initializes variables for a Gaussian Mixture Model.

---
### [5. PDF](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x01-clustering/5-pdf.py "5. PDF")

Calculates the PDF of a Gaussian distribution.

---
### [6. Expectation](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x01-clustering/6-expectation.py "6. Expectation")

Calculates the expectation step in the EM algorithm for a GMM.

---
### [7. Maximization](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x01-clustering/7-maximization.py "7. Maximization")

Calculates the maximization step in the EM algorithm for a GMM.

---
### [8. EM](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x01-clustering/8-EM.py "8. EM")

Performs the expectation maximum for a GMM.

---
### [9. BIC](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x01-clustering/9-BIC.py "9. BIC")

Finds the best number of clusters for a GMM using the Bayesian Information Criterion.

---
### [10. Hello, sklearn!](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x01-clustering/10-kmeans.py "10. Hello, sklearn!")

Performs K-means on a dataset using sklearn.

---
### [11. GMM](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x01-clustering/11-gmm.py "11. GMM")

Calculates GMM from a dataset using sklearn.

---
### [12. Agglomerative](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x01-clustering/12-agglomerative.py "12. Agglomerative")

Performs agglomerative clustering on a dataset.
