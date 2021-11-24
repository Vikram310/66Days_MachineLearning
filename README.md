# **Journey of 66DaysOfData in Machine Learning**

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

[**Day4 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_datawithvikram-datascience-careers-activity-6859371819770748929--Etu)

**💡 Linear Discriminant Analysis**: 
-  LDA is a classification that is also a popular technique for dimensionality reduction. In PCA we were only interested in the component axes that maximize the variance in the data, while in LDA we have the additional goal of maximizing the differences between classes. In scikit-learn, It is implemented using LinearDiscriminantAnalysis, which includes a parameter, n_components, indicating the number of features we want to be returned, which can be determined using explained_variance. PCA is unsupervised whereas LDA is supervised.
- LDA is carried out in the following steps

      1. Calculate between-class scatter(S_B)
      2. Calculate in-class scatter(S_W)
      3. Calculate Eigenvalues of (Inverse of in-class scatter)*(between-class scatter)
      4. Sort the Eigenvectors according to their Eigenvalues in decreasing order
      5. Choose first k Eigenvectors and that will be new k dimensions(Linear Discriminants)
      6. Transform the original n-dimensional data points into k dimensions.
    
- Reference:
  - [**Machine Learning with Python Cookbook**](https://www.amazon.in/Machine-Learning-Python-Cookbook-Preprocessing/dp/9352137302/ref=sr_1_1?crid=3SWKWJG6II2GK&keywords=machine+learning+with+python+cookbook&qid=1636010115&sprefix=Machine+Learning+with+Python+%2Caps%2C273&sr=8-1)
  - [**ML from Scratch on Youtube**](https://lnkd.in/gNPM6vW2)

[**Day5 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6859726411431845890-bvhM)

**💡 Reducing Features using Non-Negative Matrix Factorization (NMF)**:

- NMF is an unsupervised technique for linear dimensionality reduction that factorizes the feature matrix into matrices representing the latent relationship between observations and their features. NMF can reduce dimensionality because, in matrix multiplication, the two factors can have significantly fewer dimensions than the product matrix. Feature Matrix cannot contain negative values as the name implies and it does not provide us with the explained variance of the outputted features as PCA and other techniques. The best way to find the optimum value is by trying a range of values and finding the one that produces the best result. 
- In NMF, we have a variable "r" which denotes desired number of features and NMF factorizes the feature matrix such that :

            V = W x H

      V -> Original Input Matrix (Linear combination of W & H)
      W -> Feature Matrix (dimensions m x r)
      H -> Coefficient Matrix (dimensions r x n)
      r -> Low rank approximation of A (r ≤ min(m,n))
 
- Reference:
  - [**Machine Learning with Python Cookbook**](https://www.amazon.in/Machine-Learning-Python-Cookbook-Preprocessing/dp/9352137302/ref=sr_1_1?crid=3SWKWJG6II2GK&keywords=machine+learning+with+python+cookbook&qid=1636010115&sprefix=Machine+Learning+with+Python+%2Caps%2C273&sr=8-1)

[**Day6 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6860085603141328896-wDQN)

**💡 Variance Threshold**: 
-  Feature selection is the process of reducing the number of input variables when developing a predictive model. It is desirable to reduce the number of input variables to both reduce the computational cost of modelling and, in some cases, to improve the performance of the model. Feature selector that removes all low-variance features. This feature selection algorithm looks only at the features (X), not the desired outputs (y), and can thus be used for unsupervised learning.

- Reference:
  - A comprehensive Guide to Machine Learning 

[**Day7 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6860449779521064960-CSja)

**💡 Feature Selection using Pearson Correlation**: 

- Feature selection is also called variable selection or attribute selection. Feature selection methods aid you in your mission to create an accurate predictive model. They help you by choosing features that will give you good or better accuracy whilst requiring less data. While creating any model, the correlation between an independent feature and dependent feature is of high importance, but if two or more independent features are highly correlated, it is of no use, they just act as duplicate features and it is better to remove those independent features so that we can get more accurate results.

- Reference:
  - A comprehensive Guide to Machine Learning

[**Day8 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6860834031316148224-3S86)

**💡 Feature Selection - Information Gain - Mutual Information in Classification**: 

- Feature selection helps to zone in on the relevant variables in a data set, and can also help to eliminate collinear variables. It helps reduce the noise in the data set, and it helps the model pick up the relevant signals. Mutual information (MI) between two random variables is a non-negative value, which measures the dependency between the variables .It is equal to zero if and only if two random variables are independent ,and higher values mean higher dependency. The function relies on non-parametric methods based on entropy estimation from K-Nearest neighbors distances. The mutual information between two random variables X and Y can be stated formally as follows:
                              
      𝐈(𝐗 ; 𝐘) = 𝐇(𝐗) – 𝐇(𝐗 | 𝐘) 𝐖𝐡𝐞𝐫𝐞 𝐈(𝐗 ; 𝐘) 𝐢𝐬 𝐭𝐡𝐞 𝐦𝐮𝐭𝐮𝐚𝐥 𝐢𝐧𝐟𝐨𝐫𝐦𝐚𝐭𝐢𝐨𝐧 𝐟𝐨𝐫 𝐗 𝐚𝐧𝐝 𝐘, 𝐇(𝐗) 𝐢𝐬 𝐭𝐡𝐞 𝐞𝐧𝐭𝐫𝐨𝐩𝐲 𝐟𝐨𝐫 𝐗 𝐚𝐧𝐝 𝐇(𝐗 | 𝐘) 𝐢𝐬 𝐭𝐡𝐞 𝐜𝐨𝐧𝐝𝐢𝐭𝐢𝐨𝐧𝐚𝐥 𝐞𝐧𝐭𝐫𝐨𝐩𝐲 𝐟𝐨𝐫 𝐗 𝐠𝐢𝐯𝐞𝐧 𝐘. 𝐓𝐡𝐞 𝐫𝐞𝐬𝐮𝐥𝐭 𝐡𝐚𝐬 𝐭𝐡𝐞 𝐮𝐧𝐢𝐭𝐬 𝐨𝐟 𝐛𝐢𝐭𝐬.
      
- Reference:
  - A comprehensive Guide to Machine Learning

[**Day9 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6861256326744555520-0E9G)

**💡 Gradient Descent**: 

- Gradient descent is an iterative optimization algorithm that is popular and it is a base for many other optimization techniques, which tries to obtain minimal loss in a model by tuning the weights/parameters in the objective function. There are 3 types of Gradient Descent:

      Types of Gradient Descent:
            1. Batch Gradient Descent
            2. Stochastic Gradient Descent
            3. Mini Batch Gradient Descent
      
- Steps to achieve Minimal Loss:
      
      1. The first stage in gradient descent is to pick a starting value (a starting point) for w1, which is set to 0 by many algorithms.
      2. The gradient descent algorithm then calculates the gradient of the loss curve at the starting point. 
      3. The gradient always points in the direction of steepest increase in the loss function. The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.
      4. To determine the next point along the loss function curve, the gradient descent algorithm adds some fraction of the gradient's magnitude to the starting point and moves forward.
      5. The gradient descent then repeats this process, edging ever closer to the minimum.
      
- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)

[**Day10 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6861602060660568064-P6Zu)

**💡 Feature Selection - Information Gain - Mutual Information in Regression**: 

- Feature selection helps to zone in on the relevant variables in a data set, and can also help to eliminate collinear variables. It helps reduce the noise in the data set, and it helps the model pick up the relevant signals. Mutual information (MI) between two random variables is a non-negative value, which measures the dependency between the variables .It is equal to zero if and only if two random variables are independent ,and higher values mean higher dependency. The function relies on non-parametric methods based on entropy estimation from K-Nearest neighbors distances. The mutual information between two random variables X and Y can be stated formally as follows:
                              
      𝐈(𝐗 ; 𝐘) = 𝐇(𝐗) – 𝐇(𝐗 | 𝐘) 𝐖𝐡𝐞𝐫𝐞 𝐈(𝐗 ; 𝐘) 𝐢𝐬 𝐭𝐡𝐞 𝐦𝐮𝐭𝐮𝐚𝐥 𝐢𝐧𝐟𝐨𝐫𝐦𝐚𝐭𝐢𝐨𝐧 𝐟𝐨𝐫 𝐗 𝐚𝐧𝐝 𝐘, 𝐇(𝐗) 𝐢𝐬 𝐭𝐡𝐞 𝐞𝐧𝐭𝐫𝐨𝐩𝐲 𝐟𝐨𝐫 𝐗 𝐚𝐧𝐝 𝐇(𝐗 | 𝐘) 𝐢𝐬 𝐭𝐡𝐞 𝐜𝐨𝐧𝐝𝐢𝐭𝐢𝐨𝐧𝐚𝐥 𝐞𝐧𝐭𝐫𝐨𝐩𝐲 𝐟𝐨𝐫 𝐗 𝐠𝐢𝐯𝐞𝐧 𝐘. 𝐓𝐡𝐞 𝐫𝐞𝐬𝐮𝐥𝐭 𝐡𝐚𝐬 𝐭𝐡𝐞 𝐮𝐧𝐢𝐭𝐬 𝐨𝐟 𝐛𝐢𝐭𝐬.

**Select Percentile**
- This is a modification to the K-Best feature selection technique where we select the top x percentile of the best scoring features. So in our example, if we say that K is 80%, we want to select the top 80 percentile of the features based on their scores.      
      
- Reference:
  - A Comprehensive Guide to Machine Learning

[**Day11 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6861989776892030976-Imca)

**💡 Cross-Validation**: 

- Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation. Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model. It is a popular method because it is simple to understand and because it generally results in a less biased or less optimistic estimate of the model skill than other methods, such as a simple train/test split.
      
- Procedure for K-Fold Cross Validation:
      
      1. Shuffle the dataset randomly.
      2. Split the dataset into k groups
      3. For each unique group:
            a. Take the group as a holdout or test data set
            b. Take the remaining groups as a training data set
            c. Fit a model on the training set and evaluate it on the test set
            d. Retain the evaluation score and discard the model
      4. Summarize the skill of the model using the sample of model evaluation scores
      
      
- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - [**Machine Learning Mastery**](https://machinelearningmastery.com/)

[**Day12 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6862275048439463936-gXFE)

**💡 Linear Regression**: 

- Linear Regression is a linear approach to modeling the relationships between a scalar response or dependent variable and one or more explanatory variables or independent variables. Linear regression assumes that the relationship between the features and the target vector is approximately linear. That is, the effect of the features on the target vector is constant. 
- In linear regression, the target variable y is assumed to follow a linear function of one or more predictor variables plus some random error. The machine learning task is to estimate the parameters of this equation which can be achieved in two ways:

      The first approach is through the lens of minimizing loss. A common practice in machine learning is to choose a loss function that defines how well a model with a given set of parameters estimates the observed data. The most common loss function for linear regression is squared error loss. 
      
      The second approach is through the lens of maximizing the likelihood. Another common practice in machine learning is to model the target as a random variable whose distribution depends on one or more parameters, and then find the parameters that maximize its likelihood

- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - [**Data Science from Scratch**](https://www.amazon.in/Data-Science-Scratch-Joel-Grus/dp/149190142X)

[**Day13 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6862672742160965632-LexA)

**💡 Regularized Regression**: 

- Regularized regression penalizes the magnitude of the regression coefficients to avoid overfitting, which is particularly helpful for models using a large number of predictors. There are two most common methods for regularized regression: Ridge Regression and Lasso Regression. The only difference between Ridge and Lasso regression is Ridge Regression uses the L2 norm for regularization and Lasso Regression uses the L1 norm for regularization.
- L1 Norm: The basis for penalization is the sum of the absolute value of the weights for the features. It tries to achieve a sparse solution where most of the features have a zero weight. It can have multiple solutions. Essentially, the L1 norm performs feature selection and uses only a few useful features for building prediction models, and completely ignores the rest of the features.
- L2 Norm: The basis for penalization is the squared sum of weights. It tries to reduce the magnitude of weights associated with all features, thereby reducing the effect of each feature on the predicted value. As it involves a squared term, it is not preferred when dealing with outliers. It always has a unique solution and handles complex datasets better than the L1 norm.

- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - A Comprehensive Guide to Machine Learning

[**Day14 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6862984538180718592-BXWt)

**💡 Bayesian Regression**: 

-  Bayesian regression places a prior distribution on the regression coefficients in order to reconcile existing beliefs about these parameters with information gained from new data. To demonstrate Bayesian regression, we’ll follow three typical steps to Bayesian analysis:
            
            1. Writing the likelihood
            2. Writing the prior density
            3. Using Bayes’ Rule to get the posterior density, which is used to calculate the maximum-a-posteriori (MAP)


**💡 Generalized Linear Models (GLM)**: 

-  Generalized linear models (GLMs) expand on ordinary linear regression by changing the assumed error structure and allowing for the expected value of the target variable to be a nonlinear function of the predictors. One example of GLM is Poisson regression.  A GLM can be fit in these four steps:
            
            1. Specify the distribution of Y indexed by its mean parameter μ
            2. Specify the link function η (subscript n)=g(μ(subscript n)).
            3. Identify a loss function. This is typically the negative log-likelihood.
            4. Find the β that minimize that loss function.


- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - A Comprehensive Guide to Machine Learning

[**Day15 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6863376542739898368-1StW)

**💡 Logistic Regression**: 

- Logistic Regression models a function of the target variable as a linear combination of the predictors, then converts this function into a fitted value in the desired range.
- Binary or Binomial Logistic Regression can be understood as the type of Logistic Regression that deals with scenarios wherein the observed outcomes for dependent variables can be only in binary, i.e., it can have only two possible types.
- Multinomial Logistic Regression works in scenarios where the outcome can have more than two possible types – type A vs type B vs type C – that are not in any particular order. 


- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - A Comprehensive Guide to Machine Learning

[**Day16 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6863716039226720256-L9Pp)

**💡 Perceptron Algorithm**: 

- The perceptron algorithm is a simple classification method that plays an important historical role in the development of a much more flexible neural network.The perceptron is a linear binary classifier—linear since it separates the input variable space linearly and binary since it places observations into one of two classes. 
- It consists of a single node or neuron that takes a row of data as input and predicts a class label. This is achieved by calculating the weighted sum of the inputs and a bias (set to 1). The weighted sum of the input of the model is called the activation. The coefficients of the model are referred to as input weights and are trained using the stochastic gradient descent optimization algorithm.

- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - [**Machine Learning Mastery**](https://machinelearningmastery.com/)

[**Day18 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6864445774042951680-7_4y)

**💡 Generative Classifiers**: 

- Generative classifiers view the predictors as being generated according to their class—i.e., they see the predictors as a function of the target, rather than the other way around. They then use Bayes’ rule to turn P( x(n) | Y(n) = k ) into P( Y(n) = k | x(n) ).  Generative models can be broken down into the three following steps:
      
      1. Estimate the density of the predictors conditional on the target belonging to each class.
      2. Estimate the prior probability that a target belongs to any given class. 
      3. Using Bayes’ rule, calculate the posterior probability that the target belongs to any given class.

- This can be achieved using any one of LDA, QDA, or Naive Bayes. Quadratic Discriminant Analysis (QDA) is a classification algorithm and it is used in machine learning and statistics problems. QDA is an extension of Linear Discriminant Analysis (LDA). Unlike LDA, QDA considers each class to have its own variance or covariance matrix rather than to have a common one.

- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - A Comprehensive Guide to Machine Learning

[**Day19 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6864917122414473216-mzcm)

**💡 CART (Classification and Regression Trees)**: 

- A decision tree is an interpretable machine learning method for regression and classification. Trees iteratively split samples of the training data based on the value of a chosen predictor; the goal of each split is to create two sub-samples, or “children,” with greater purity of the target variable than their “parent”.
- For regression tasks, purity means the first child should have observations with high values of the target variable and the second should have observations with low values.
- For classification tasks, purity means the first child should have observations primarily of one class and the second should have observations primarily of another.

- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - A Comprehensive Guide to Machine Learning

[**Day20 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6865279984592936960-2Gi9)

**💡 TREE Ensemble Methods**: 

- Ensemble methods combine the output of multiple simple models, often called “learners”, in order to create a final model with lower variance.Bagging, short for bootstrap aggregating, trains many learners on bootstrapped samples of the training data and aggregates the results into one final model. The process of bagging is very simple yet often quite powerful.
- How exactly we combine the results of the learners into a single fitted value (the second part of the second step) depends on the target variable.

      1. For a continuous target variable, we typically average the learners’ predictions.
      2. For a categorical target variable, we typically use the class that receives the plurality vote.

- A random forest is a slight extension to the bagging approach for decision trees that can further decrease overfitting and improve out-of-sample precision. Unlike bagging, random forests are exclusively designed for decision trees. Like bagging, a random forest combines the predictions of several base learners, each trained on a bootstrapped sample of the original training set. Random forests, however, add one additional regulatory step: at each split within each tree, we only consider splitting a randomly-chosen subset of the predictors.

- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - A Comprehensive Guide to Machine Learning

[**Day21 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6866004656930336768-HfU2)

**💡 Adaboost for Classification**: 

-  Like bagging and random forests, boosting combines multiple weak learners into one improved model. Boosting trains these weak learners sequentially, each one learning from the mistakes of the last. Weak learners in a boosting model learn from the mistakes of previous iterations by increasing the weights of observations that previous learners struggled with. How exactly we fit a weighted learner depends on the type of learner. Fortunately, for classification trees, this can be done with just a slight adjustment to the loss function. We use Discrete Adaboost for binary classification.
-  Discrete Adaboost for Classification is achieved using the following steps:

            1. Initialize the weights with W(1,n) = 1/N
            2. Build a weak learner t using W(t)
            3. Use weak learner to calculate fitted values
            4. Calculate the weighted error for learner t
            5. Calculate the accuracy measure for learner t
            6. Calculate the overall fitted values

- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - A Comprehensive Guide to Machine Learning

[**Day22 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6866366842416586752-g4_4)

**💡 Adaboost for Regression**: 

-  Like AdaBoost, this algorithm uses weights to emphasize observations that previous learners struggled with. Unlike AdaBoost, however, it does not incorporate these weights into the loss function directly. Instead, in every iteration, it draws bootstrap samples from the training data where observations with greater weights are more likely to be drawn.
-  We then fit a weak learner to the bootstrapped sample, calculate the fitted values on the original sample (i.e. not the bootstrapped sample), and use the residuals to assess the quality of the weak learner.
-  In simple words, iteratively fit a weak learner, see where the learner struggles, and emphasize the observations where it failed (where the amount of emphasis depends on the overall strength of the learner).

- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - A Comprehensive Guide to Machine Learning

[**Day23 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_github-vikram31066daysmachinelearning-activity-6866729191661035522-fFTt)

**💡 Neural Networks**: 

-  Neural networks come in a variety of forms intended to accomplish a variety of tasks. Recurrent neural networks are designed to model time series data, convolutional neural networks are designed to model image data and Feed-forward Neural networks can be used for regression or classification tasks. 
-  An activation function is a (typically) nonlinear function that allows the network to learn complex relationships between the predictor(s) and the target variable(s). The two most common activation functions are ReLU (Rectified Linear Unit) and Sigmoid Functions. ReLU is a simple yet extremely common activation function. It acts like a switch, selectively turning channels on and off. 
-  Neural Networks can be constructed in two common ways:

            1. Loop Approach
            2. Matrix Approach

- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - A Comprehensive Guide to Machine Learning

[**Day24 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6867092398095388672-d_5R)

**💡 Construction of FFNN by Loop Approach**: 

-  It loops through the observations and adds the individual gradients
-  Firstly, we build activation functions and then, we construct a class for fitting feed-forward networks by looping through observations. This class conducts gradient descent by calculating the gradients based on one observation at a time, looping through all observations, and summing the gradients before adjusting the weights.
-  Once instantiated, we fit a network, which requires training data, the number of nodes for the hidden layer, an activation function for the first and second layers’ outputs, a loss function, and some parameters for gradient descent. After storing those values, the method randomly instantiates the network’s weights: W1, c1, W2, and c2. It then passes the data through this network to instantiate the output values: h1, z1, h2, and yhat(z2)
-  We then begin conducting gradient descent. Within each iteration of the gradient descent process, we also iterate through the observations. For each observation, we calculate the derivative of the loss for that observation with respect to the network’s weights. We then sum these individual derivatives and adjust the weights accordingly, as is typical in gradient descent.

- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - A Comprehensive Guide to Machine Learning

[**Day25 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6867452400391860224-PGFO)

**💡 Implementation of FFNN**: 

-  Neural networks in Keras can be fit through one of two APIs: The Sequential API and The Functional API.
-   Fitting a network with the Keras sequential API can be broken down into four steps:

            1. Instantiate model
            2. Add layers
            3. Compile model (and summarize)
            4. Fit model

- A Dense layer is one in which each neuron is a function of all the other neurons in the previous layer. We identify the number of neurons in the layer with the units argument and the activation function applied to the layer with the activation argument. For the first layer only, we must also identify the input_shape or the number of neurons in the input layer.
- Fitting models with the Functional API can again be broken into four steps:
   
      1. Define layers
      2. Define model
      3. Compile model (and summarize)
      4. Fit model

- While the sequential approach first defines the model and then adds layers, the functional approach does the opposite. Note that in this approach, we link layers directly.

- Reference:
  - [**Machine Learning From Scratch**](https://dafriedman97.github.io/mlbook/content/introduction.html)
  - A Comprehensive Guide to Machine Learning

[**Day26 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6868180581071646720-HkAR)

**💡 Reducing Variance usign Regularization**:: 

- Regularized regression learners are similar to standard ones, except they attempt to minimize RSS and some penalty for the total size of the coefficient values, called a shrinkage penalty because it attempts to “shrink” the model. There are two common types of regularized learners for linear regression: Ridge regression and Lasso Regression. The only formal difference is the type of shrinkage penalty used.
- In Ridge regression, the shrinkage penalty is a tuning hyperparameter multiplied by the squared sum of all coefficients, whereas in Lasso regression, shrinkage penalty is a tuning hyperparameter multiplied by the sum of the absolute value of all coefficients. As a very general rule of thumb, ridge regression often produces slightly better predictions than lasso, but lasso produces more interpretable models.
- If we want a balance between ridge and lasso’s penalty functions we can use an elastic net, which is simply a regression model with both penalties included.Regardless of which one we use, both ridge and lasso regressions can penalize large or complex models by including coefficient values in the loss function we are trying to minimize

- Reference:
  - Machine Learning with Python Cookbook

[**Day27 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6868547393386885120-ch0P)

**💡 Decision Tree Classifier**:: 

-  In a decision tree, every decision rule occurs at a decision node, with the rule creating branches leading to new nodes. One reason for the popularity of tree-based models is their interpretability. In fact, decision trees can literally be drawn out in their complete form to create a highly intuitive model. Decision tree learners attempt to find a decision rule that produces the greatest decrease in impurity at a node. While there are a number of measurements of impurity, by default Decision Tree Classifier uses Gini impurity.
-  This process of finding the decision rules that create splits to increase impurity is repeated recursively until all leaf nodes are pure or some arbitrary cut-off is reached. One of the advantages of decision tree classifiers is that we can visualize the entire trained model making decision trees one of the most interpretable models in machine learning. While this solution visualized a decision tree classifier, it can just as easily be used to visualize a decision tree regressor.

- Reference:
  - Machine Learning with Python Cookbook

[**Day28 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6868901539273154560-Q7VV)

**💡 Random Forest Classifier**:: 

-  A common problem with decision trees is that they tend to fit the training data too closely (i.e., overfitting). This has motivated the widespread use of an ensemble learning method called random forest. In a random forest, many decision trees are trained, but each tree only receives a bootstrapped sample of observations and each node only considers a subset of features when determining the best split. 
-  However, a random forest model is comprised of tens, hundreds, even thousands of decision trees. This makes a simple, intuitive visualization of a random forest model impractical. But, we can compare the relative importance of each feature. Features with splits that have the greater mean decrease in impurity (e.g. Gini impurity or entropy in classifiers and variance in regressors) are considered more important.However, there are two things to keep in mind regarding feature importance.
-  First, scikit-learn requires that we break up nominal categorical features into multiple binary features. Second, if two features are highly correlated, one feature will claim much of the importance, making the other feature appear to be far less important. The higher the number, the more important the feature. By plotting these values we can add interpretability to our random forest models.

- Reference:
  - Machine Learning with Python Cookbook

[**Day29 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6869264271306563584-TBLP)

**💡 Selecting Important Features using Random Forest**:: 

-  In scikit-learn, we can use a simple two-stage workflow to create a model with reduced features. 

            1. First, we train a random forest model using all features. 
            2. Then, we use this model to identify the most important features. 
            3. Next, we create a new feature matrix that includes only these features. 
- It must be noted that there are two caveats to this approach:
      
      1. Nominal categorical features that have been one-hot encoded will see the feature importance diluted across the binary features.
      2. The feature importance of highly correlated features will be effectively assigned to one feature and not evenly distributed across both features.

- Reference:
  - Machine Learning with Python Cookbook
  - [Research paper on Variable selection using Random Forests](https://hal.archives-ouvertes.fr/file/index/docid/755489/filename/PRLv4.pdf)

