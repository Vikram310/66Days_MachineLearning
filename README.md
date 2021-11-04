# # **Journey of 66DaysOfData in Machine Learning**

[**Day1 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6858272818472329216-FOFx)

**💡 Principal Component Analysis**: 
- PCA is a dimensionality reduction technique that enables you to identify the correlations and patterns in the dataset so that it can be transformed into a dataset of significantly lower dimensions without any loss of important information. It is an unsupervised statistical technique used to examine the interrelations among a set of variables. It is also known as a general factor analysis where regression determines a line of best fit. It works on a condition that while the data in a higher-dimensional space is mapped to data in a lower dimension space, the variance or spread of the data in the lower dimensional space should be maximum.
- PCA is carried out in the following steps

      1. Standardization of Data
      
      2. Computing the covariance matrix
      
      3. Calculation of the eigenvectors and eigenvalues
      
      4. Computing the Principal components
      
      5. Reducing the dimensions of the Data.
    
- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - [**ML from Scratch on Youtube**](https://lnkd.in/gNPM6vW2) 

[**Day2 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6858635634706386944-RoFf)

**💡 Principal Component Analysis using scikit-learn**: 
- PCA projects observations onto the (hopefully fewer) principal components of the feature matrix that retain the most variance. PCA can also be used in the scenario, where we need features to be retained that share maximum variance. PCA is implemented in scikit-learn using the PCA method:

      class sklearn.decomposition.PCA(n_components=None, *, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)

- n_components has two operations, depending on the argument provided. If the argument is greater than 1,n_components will return that many features. If the argument to n_components is between 0 and 1, PCA returns the minimum amount of features that retain that much variance. It is common to use values of 0.95 and 0.99, meaning 95% and 99% of the variance of the original features has been retained.
- whiten =True transforms the values of each principal component so that they have zero mean and unit variance. Whitening will remove some information from the transformed signal but can sometimes improve the predictive accuracy of the downstream estimators.
- svd_solver=" randomized", which implements a stochastic algorithm to find the first principal components in often significantly less time. 

- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - [**Scikit-learn Implementation**](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) 

[**Day3 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6859000073649561600-IFgU)

**💡 RBF Kernel PCA**: 
- Standard PCA uses linear projection to reduce the features. If the data is linearly separable then PCA works well. However, if your data is not linearly separable, then linear transformation will not work as well as expected. In this scenario, Kernel PCA will be useful. Kernel PCA uses a kernel function to project a dataset into a higher dimensional feature space, where it is linearly separable, this is called the Kernel trick. The most commonly used Kernel PCA is Gaussian radial basis function kernel RBF.
- RBF Kernel PCA is carried out in the following steps

      1. Computation of Kernel Matrix: 
            We need to compute kernel matrix for every point i.e., if there are 50 samples in a dataset, this step will result in a 50x50 kernel matrix.
      
      2. Eigen-decomposition of Kernel Matrix:
            To make the kernel matrix centered, we are applying this step and to obtain the eigenvectors of the centered kernel matrix that correspond to the largest eigenvalues.
 
- Reference:
  - [**Machine Learning with Python Cookbook**](https://www.amazon.in/Machine-Learning-Python-Cookbook-Preprocessing/dp/9352137302/ref=sr_1_1?crid=3SWKWJG6II2GK&keywords=machine+learning+with+python+cookbook&qid=1636010115&sprefix=Machine+Learning+with+Python+%2Caps%2C273&sr=8-1)
  
