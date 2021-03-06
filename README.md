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

**💡 Reducing Variance usign Regularization**: 

- Regularized regression learners are similar to standard ones, except they attempt to minimize RSS and some penalty for the total size of the coefficient values, called a shrinkage penalty because it attempts to “shrink” the model. There are two common types of regularized learners for linear regression: Ridge regression and Lasso Regression. The only formal difference is the type of shrinkage penalty used.
- In Ridge regression, the shrinkage penalty is a tuning hyperparameter multiplied by the squared sum of all coefficients, whereas in Lasso regression, shrinkage penalty is a tuning hyperparameter multiplied by the sum of the absolute value of all coefficients. As a very general rule of thumb, ridge regression often produces slightly better predictions than lasso, but lasso produces more interpretable models.
- If we want a balance between ridge and lasso’s penalty functions we can use an elastic net, which is simply a regression model with both penalties included.Regardless of which one we use, both ridge and lasso regressions can penalize large or complex models by including coefficient values in the loss function we are trying to minimize

- Reference:
  - Machine Learning with Python Cookbook

[**Day27 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6868547393386885120-ch0P)

**💡 Decision Tree Classifier**:

-  In a decision tree, every decision rule occurs at a decision node, with the rule creating branches leading to new nodes. One reason for the popularity of tree-based models is their interpretability. In fact, decision trees can literally be drawn out in their complete form to create a highly intuitive model. Decision tree learners attempt to find a decision rule that produces the greatest decrease in impurity at a node. While there are a number of measurements of impurity, by default Decision Tree Classifier uses Gini impurity.
-  This process of finding the decision rules that create splits to increase impurity is repeated recursively until all leaf nodes are pure or some arbitrary cut-off is reached. One of the advantages of decision tree classifiers is that we can visualize the entire trained model making decision trees one of the most interpretable models in machine learning. While this solution visualized a decision tree classifier, it can just as easily be used to visualize a decision tree regressor.

- Reference:
  - Machine Learning with Python Cookbook

[**Day28 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6868901539273154560-Q7VV)

**💡 Random Forest Classifier**: 

-  A common problem with decision trees is that they tend to fit the training data too closely (i.e., overfitting). This has motivated the widespread use of an ensemble learning method called random forest. In a random forest, many decision trees are trained, but each tree only receives a bootstrapped sample of observations and each node only considers a subset of features when determining the best split. 
-  However, a random forest model is comprised of tens, hundreds, even thousands of decision trees. This makes a simple, intuitive visualization of a random forest model impractical. But, we can compare the relative importance of each feature. Features with splits that have the greater mean decrease in impurity (e.g. Gini impurity or entropy in classifiers and variance in regressors) are considered more important.However, there are two things to keep in mind regarding feature importance.
-  First, scikit-learn requires that we break up nominal categorical features into multiple binary features. Second, if two features are highly correlated, one feature will claim much of the importance, making the other feature appear to be far less important. The higher the number, the more important the feature. By plotting these values we can add interpretability to our random forest models.

- Reference:
  - Machine Learning with Python Cookbook

[**Day29 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6869264271306563584-TBLP)

**💡 Selecting Important Features using Random Forest**: 

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

[**Day30 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6869628887819808769-W9wZ)

**💡 Handling Imbalanced Classes**: 

-  Imbalanced classes are a common problem when we are doing machine learning in the real world. Left unaddressed, the presence of imbalanced classes can reduce the performance of our model. However, many learning algorithms in scikit-learn come with built-in methods for correcting imbalanced classes.
-   We can set RandomForestClassifier to correct for imbalanced classes using the class_weight parameter. If supplied with a dictionary in the form of class names and respective desired weights, RandomFor estClassifier will weigh the classes accordingly. However, often a more useful argument is balanced, wherein classes are automatically weighted inversely proportional to how frequently they appear in the data as  

            W(j) = n / (k * n(j)) 

- Reference:
  - Machine Learning with Python Cookbook

[**Day31 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6869983202891698176-LMUl)

**💡 Adaboost Classifier**:

- In a random forest, an ensemble (group) of randomized decision trees predicts the target of a vector. An alternative, and often more powerful, approach is called boosting. In Adaboost, we iteratively train a series of weak models, each iteration giving higher priority to observations the previous model predicted incorrectly.
- Steps followed in Adaboost:

      1. Assign every observation, xi, an initial weight value, wi = 1 n, where n is the total number of observations in the data. 
      2. Train a “weak” model on the data. 
      3. For each observation: 
           a. If the weak model predicts xi correctly, wi is increased. 
           b. If the weak model predicts xi incorrectly, wi is decreased. 
      4. Train a new weak model where observations with greater wi are given greater priority. 
      5. Repeat steps 4 and 5 until the data is perfectly predicted or a preset number of weak models has been trained

- The end result is an aggregated model where individual weak models focus on more difficult observations. In scikit-learn, we can implement AdaBoost using AdaBoostClassifier or AdaBoostRegressor. The most important parameters are base_estimator, n_estimators, and learning_rate: 

      1. base_estimator is the learning algorithm to use to train the weak models. 
      2. n_estimators is the number of models to iteratively train.
      3. learning_rate is the contribution of each model to the weights and defaults to 1.
      4. loss is exclusive to AdaBoostRegressor and sets the loss function to use when updating weights. 

- Reference:
  - Machine Learning with Python Cookbook

[**Day32 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6871073558047682560-bfLd)

**💡 Performance measures of classifier**: 

- A good way to evaluate a model's accuracy is to use cross-validation. We use the cross_val_score() function to evaluate the model using K-fold cross-validation. But accuracy is not a preferred metric when classifying in predictive analytics. This is because a simple model may have a high level of accuracy but be too crude to be useful. This is known as Accuracy Paradox 
- A much better way to evaluate the performance of a classifier is to look at the confusion matrix. The general idea is to count the number of times instances of class A are classified as class B. Each row in a confusion matrix represents an actual class, while each column represents a predicted class.  A perfect classifier would have only true positives and true negatives, so its confusion matrix would have nonzero values only on its main diagonal. Confusion matrix gives us a lot of information, two of those are precision and recall. 
- Precision is defined as the fraction of relevant instances among all retrieved instances. Recall, sometimes referred to as ‘sensitivity, is the fraction of retrieved instances among all relevant instances. A perfect classifier has precision and recall both equal to 1.

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day33 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6871435103684435968-9gA1)

**💡 Precision-recall Tradeoff**: 

- The precision-recall trade-off can be an essential tool when precision is more important than recall or vice versa.In the context of machine learning, precision and recall are metrics of performance for classification algorithms. Consider a classification task with two classes. 
- Precision is how many times an accurate prediction of a particular class occurs per a false prediction of that class. Recall is the percentage of the data belonging to a particular class which the model properly predicts as belonging to that class. The Idea behind the precision-recall trade-off is that when the threshold is changed for determining if a class is positive or negative it will tilt the scales. Increasing the threshold will decrease recall and increase precision whereas, Decreasing the threshold will increase recall and decrease precision.

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day34 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6871799471076978688-Vw3m)

**💡 ROC Curve**: 

- The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers. It is very similar to the precision/recall curve. ROC curve plots the true positive rate(TPR) against the false-positive rate(FPR), Instead of plotting precision versus recall. The FPR is the ratio of negative instances that are incorrectly classified as positive.
- There is a tradeoff in this curve too: the higher the recall(TPR), the more false positives (FPR) the classifier produces. A good classifier stays as far away i.e. it is more towards the top-left corner of the plot) One way to compare classifiers is to measure the area under the curve (AUC). A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5.
- As a rule of thumb, one should prefer the PR curve whenever the positive class is rare or when you care more about the false positives than the false negatives, and the ROC curve otherwise. 

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day35 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6872133716102275073-ZcjS)

**💡 K-Nearest Neighbors**: 

- K-nearest neighbors (KNN) algorithm is a type of supervised ML algorithm which can be used for both classification as well as regression predictive problems. The following two properties would define KNN well −

      1. Lazy learning algorithm − KNN is a lazy learning algorithm because it does not have a specialized training phase and uses all the data for training while classification.
      2. Non-parametric learning algorithm − KNN is also a non-parametric learning algorithm because it doesn’t assume anything about the underlying data.

- K-nearest neighbors (KNN) algorithm uses ‘feature similarity’ to predict the values of new data points which further means that the new data point will be assigned a value based on how closely it matches the points in the training set. We calculate the distance between test data and each row of training data with the help of any of the method namely: Euclidean, Manhattan or Hamming distance. The most commonly used method to calculate distance is Euclidean.

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day36 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6872515085559058432-KkPs)

**💡 Support Vector Machine**: 

- A Support Vector Machine (SVM) is a very powerful and versatile Machine Learning model, capable of performing linear or nonlinear classification, regression, and even outlier detection. Support vector machines classify data by finding the hyperplane that maximizes the margin between the classes in the training data.

- SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine. The SVM classifier which fits the widest possible street between the classes. This is called large margin classification.  There are two main issues with hard margin classification. First, it only works if the data is linearly separable, and second it is quite sensitive to outliers.

- To avoid these issues it is preferable to use a more flexible model. The objective is to find a good balance between keeping the street as large as possible and limiting the margin violations This is called soft margin classification

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day37 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6872955098474143744-6a45)

**💡 Logistic Regression**: 

- Logistic regression is a classification model that is very easy to implement but performs very well on linearly separable classes. Logistic Regression is  a linear model for binary classification that can be extended to multiclass classification using the OvR technique (One-vs-Rest)
- Just like a Linear Regression model, a Logistic Regression model computes a weighted sum of the input features (plus a bias term), but instead of outputting the result directly like the Linear Regression model does, it outputs the logistic of this result, which is given by a sigmoid function. 
- Assumptions of Logistic Regression:

      1. Logistic regression assumes that the outcome variable is binary, where the number of outcomes is two. 
      2. Relationship between the Logit of the outcome and each continuous independent variable is linear. 
      3. It assumes that there are no highly influential outlier data points, as they distort the outcome and accuracy of the model.
- Evaluation of Logistic Regression Model:

      1. AIC (Akaike Information Criteria)
      2. Null Deviance and Residual Deviance
      3. Confusion Matrix
      4. ROC Curve 

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day38 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6873252605414510592-mfct)

**💡 Naive Bayes Classifier**: 

- It is a classification technique based on the Bayes Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes Classifier assumes that the presence of a particular feature is unrelated to the presence of any other feature. Bayes theorem provides a way of calculating posterior probability P(c|x) from P(c), P(x) and P(x|c).

- The most common type of naive Bayes classifier is the Gaussian naive Bayes. In Gaussian naive Bayes, we assume that the likelihood of the feature values, x, given observation is of class y, follows a normal distribution. One of the interesting aspects of naive Bayes classifiers is that they allow us to assign a prior belief over the respected target classes, which can be done using the Priors parameter

- Multinomial naive Bayes works similarly to Gaussian naive Bayes, but the features are assumed to be multinomially distributed. In practice, this means that this classifier is commonly used when we have discrete data. The Bernoulli naive Bayes classifier assumes that all our features are binary such that they take only two values. Like its multinomial cousin, Bernoulli naive Bayes is often used in text classification

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day39 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6873604916418633728-oncj)

**💡 K-Means Clustering**: 

- • The goal of clustering is to find a natural grouping in data so that items in the same cluster are more similar to each other than to those from different cluster. The K - Means algorithm belongs to the category of prototype-based clustering, which means that each cluster is represented by a prototype, which is usually either the centroid (average) of similar points with continuous features, or the medoid in the case of categorical features.

- Steps in K-Means:

      1. Randomly pick k centroids from the examples as initial cluster centers.
      2. Assign each example to the nearest centroid.
      3. Move the centroids to the center of the examples that were assigned to them.
      4. Repeat steps 2 and 3 until the cluster assignments do not change or a user-defined tolerance or a maximum number of iterations is reached.

- The Elbow method is one of the most popular ways to find the optimal number of clusters. This method uses the concept of WCSS value. WCSS stands for Within Cluster Sum of Squares, which defines the total variations within a cluster. 
- Silhouette Scores to validate the clusters, which uses the euclidean method for calculating the distance

      1. Silhouette values lie in the range of -1 to +1. The value of +1 is ideal and -1 is the least preferred
      2. Value of +1 indicates the sample is far away from its neighboring cluster and very close to the cluster it is assigned
      3. Value of -1 indicates the sample is close to the neighboring cluster than to the cluster it is assigned
      4. Value of 0 means it's at the boundary of the distance

- Reference:
  - Python Machine Learning

[**Day40 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6873964910234226688-pDUq)

**💡 Hierarchical Clustering**: 

- Hierarchical clustering is another unsupervised machine learning algorithm, which is used to group the unlabeled datasets into a cluster. It is of two types
- 𝐃𝐢𝐯𝐢𝐬𝐢𝐯𝐞 𝐂𝐥𝐮𝐬𝐭𝐞𝐫𝐢𝐧𝐠:
  
      1. Top down clustering method
      2. Assign all of the observations to a single cluster and then partition the cluster to two least similar clusters.
      3. Proceed recursively on each cluster until there is one cluster for each observation
      4. Produce more accurate hierarchies than agglomerative in some circumstances but are conceptually more complex.
- Agglomerative Clustering:
      
      1. Bottom-up clustering method
      2. Assign each observation to its own cluster
      3. Compute the similarity between each of the clusters and join the two most similar clusters
- Methods to compute the distance between clusters/points:

      𝐒𝐢𝐧𝐠𝐥𝐞 𝐋𝐢𝐧𝐤𝐚𝐠𝐞: Distance between two clusters is defined as shortest distance between two points in each cluster
      𝐂𝐨𝐦𝐩𝐥𝐞𝐭𝐞 𝐋𝐢𝐧𝐤𝐚𝐠𝐞: Distance between two clusters is defined as longest distance between two points in each cluster
      𝐀𝐯𝐞𝐫𝐚𝐠𝐞 𝐋𝐢𝐧𝐤𝐚𝐠𝐞: Distance between two clusters is defined a the average distance between each point in one cluster to every point in the other cluster

- We use Dendrograms to compute the optimal number of clusters in Hierarchical Clustering. It is a diagram that shows the hierarchical relationship between objects. It is most commonly created as an output from hierarchical clustering. The main use of dendrogram is to work out the best way to allocate objects to clusters. Whenever the Data is in spherical shape, K-Means is preferred, whereas the Hierarchical clustering is highly  preferred for social networking analysis

- Reference:
  - Python Machine Learning

[**Day41 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6874329215781736448-Q3PA)

**💡 DBSCAN Clustering**: 

- DBSCAN stands for Density-based spatial clustering of applications with noise. It is able to find arbitrary shaped clusters and clusters with noise (i.e. outliers).The main idea behind DBSCAN is that a point belongs to a cluster if it is close to many points from that cluster. It has two main parameters:
- EPS

      It specifies how close points should be to each other to be considered a part of a cluster.
      This value will be considered as a threshold for considering two points as a Neighbor
- minPts:
      
      It is the minimum number of points to form a dense region.
      Generally, the Number of minPts is equal to twice the numbers of columns in dataset
- There are three types of points after the DBSCAN clustering is complete:
      
      1. Core: It is a point that has at least m points within distance n from itself
      2. Border: This is a point that has at least one core point at a distance n
      3. Noise: which is a point that is neither a core nor a border and it has less than m points within distance n from itself.
- Steps in DBSCAN Algorithm:

      1. Find all the neighbor points within eps and identify the core points or visited with more than minPts neighbors
      2. For each core point if it is not assigned to a cluster, create a new cluster
      3. Find recursively all its density connected points and assign them to the same cluster as the core point.
      4. Iterate through the remaining unvisited points in the dataset. Those points that don't belong to any cluster are noise

- Reference:
  - Python Machine Learning

[**Day42 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6874692912924491776-l_pD)

**💡 Bias - Variance TradeOff**: 

- An important theoretical result of statistics and Machine Learning is the fact that a model’s generalization error can be expressed as the sum of three very different errors:
- Bias :

      Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. This part of the generalization error is due to wrong assumptions, such as assuming that the data is linear when it is actually quadratic. A high-bias model is most likely to underfit the training data.
- Variance:

      Variance is the variability of model prediction for a given data point or a value which tells us spread of our data. This part is due to the model’s excessive sensitivity to small variations in the training data. A model with many degrees of freedom (such as a high-degree polynomial model) is likely to have high variance, and thus to overfit the training data.
- Irreducible error:

      This part is due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data (e.g., fix the data sources, such as broken sensors, or detect and remove outliers).
- Increasing a model’s complexity will typically increase its variance and reduce its bias. Conversely, reducing a model’s complexity increases its bias and reduces its variance. This is why it is called a tradeoff.

- Reference:
  - Python Machine Learning

[**Day43 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6875057106718982144-JDip)

**💡 Early Stopping**: 

- A very different way to regularize iterative learning algorithms such as Gradient Descent is to stop training as soon as the validation error reaches a minimum. This is called early stopping. When training the network, a larger number of training epochs is used than may normally be required, to give the network plenty of opportunity to fit, then begin to overfit the training dataset. There are three elements to using early stopping:

      1. Monitoring model performance.
      2. Trigger to stop training.
      3. The choice of model to use.
- Monitoring Performance:

      The performance of the model must be monitored during training. This requires the choice of a dataset that is used to evaluate the model and a metric used to evaluate the model. 
      Performance of the model is evaluated on the validation set at the end of each epoch, which adds an additional computational cost during training.
- Early Stopping Trigger:

      Once a scheme for evaluating the model is selected, a trigger for stopping the training process must be chosen. The trigger will use a monitored performance metric to decide when to stop training. 
      This is often the performance of the model on the holdout dataset, such as the loss.
- Model Choice:

      As such, some consideration may need to be given as to exactly which model is saved. Specifically, the training epoch from which weights in the model that are saved to file. 
      This will depend on the trigger chosen to stop the training process.
- For example, If the trigger is required to observe a decrease in performance over a fixed number of epochs, then the model at the beginning of the trigger period will be preferred.

- Reference:
  - Python Machine Learning

[**Day44 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6875419103482527744-OUwf)

**💡 Gradient Descent**: 

- Gradient Descent is a very generic optimization algorithm capable of finding optimal solutions to a wide range of problems. The general idea of Gradient Descent is to tweak parameters iteratively in order to minimize a cost function. It measures the local gradient of the error function with regards to the parameter vector and it goes in direction of descending gradient. We reach the minimum once it is zero. We start by filling theta value with random values, which is called random initialization and then we improve it gradually taking one baby step at a time. 
- Learning rate determines the size of step which should be taken by Gradient Descent. If the learning rate is too small, the algorithm takes a long time to reach the minimum, and, on the other hand, if it is too large, it may cross the minimum in the first iteration.When using Gradient Descent, all the features must be on the same scale, they need to be standardized or else it will take much longer to converge. 

**💡Batch Gradient Descent**: Unlike Gradient Descent, Batch GD takes whole training data at every step. As a result, it takes a longer time to train when the size of the training set is large.

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day45 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6875841095528542208-zOxt)

**💡 Stochastic Gradient Descent**: 

- The main problem of Batch Gradient Descent is, it is very slow when the training set is large. To overcome this problem, we use Stochastic Gradient Descent.Stochastic GD just picks a random instance in the training set at every step and computes gradients based only on that single instance. Due to its stochastic nature, this is much less regular than Batch GD
- Instead of gently decreasing until it reaches the minimum, the cost function will bounce up and down, decreasing only on average. With this irregularity, stochastic has a better chance of finding a Global minimum compared to a local minimum.The function that determines the learning rate at each iteration is called the learning schedule. When using Stochastic GD, the training instances must be independent and identically distributed to ensure that parameters get pulled towards global minimum on average. A simple way to ensure this is to shuffle the instances during the training.

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day46 of 66DaysOfData!**]()

**💡 Perceptron**: 

- The Perceptron is one of the simplest ANN architectures, based on a slightly different artificial neuron, a threshold logic unit (TLU), or a linear threshold unit (LTU): the inputs and output are now numbers and each input connection is associated with a weight. The TLU computes a weighted sum of its inputs, then applies a step function to that sum and outputs the result. 
- A single TLU can be used for simple linear binary classification. It computes a linear combination of the inputs and if the result exceeds a threshold, it outputs the positive class or else outputs the negative class. A Perceptron is simply composed of a single layer of TLUs,6 with each TLU connected to all the inputs. When all the neurons in a layer are connected to every neuron in the previous layer, it is called a fully connected layer or a dense layer.
- The perceptron is trained on perceptron training algorithm proposed by Frank Rosenblatt based on Hebb's rule: that is, the connection weight between two neurons is increased whenever they have the same output.Perceptron are trained using a variant of this rule that takes into account the error made by the network; it reinforces connections that help reduce the error. More specifically, the Perceptron is fed one training instance at a time, and for each instance it makes its predictions. For every  output neuron that produced a wrong prediction, it reinforces the connection weights from the inputs that would have contributed to the correct prediction.

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day47 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6876502156015472640-7zaN)

**💡 K-Fold Cross Validation**: 

- The validation approach which we use on general basis (building on training data and testing on test data) has two major weaknesses:

      1. First, the performance of the model can be highly dependent on which few observations were selected for the test set. 
      2. Second, the model is not being trained using all the available data, and not being evaluated on all the available data.
- A better strategy, which overcomes these weaknesses, is called k-fold cross-validation (KFCV).  In KFCV, we split the data into k parts called “folds.” The model is then trained using k – 1 folds combined into one training set and then the last fold is used as a test set. We repeat this k times, each time using a different fold as the test set. The performance on the model for each of the k iterations is then averaged to produce an overall measurement.
- There are three important points to consider when we are using KFCV. 

      1. First, KFCV assumes that each observation was created independent from the other (i.e., the data is independent identically distributed [IID]). If the data is IID, it is a good idea to shuffle observations when assigning to folds. 

      2. Second, when we are using KFCV to evaluate a classifier, it is often beneficial to have folds containing roughly the same percentage of observations from each of the different target classes (called stratified k-fold).

      3. Finally, when we are using validation sets or cross-validation, it is important to pre-process data based on the training set and then apply those transformations to both the training and test set. The reason for this is because we are pretending that the test set is unknown data. 

- If we fit both our preprocessors using observations from both training and test sets, some of the information from the test set leaks into our training set. This rule applies for any preprocessing step such as feature selection.
- cross_val_score() comes with three parameters:

      1. cv determines our cross-validation technique. K-fold is the most common
      2. The scoring parameter defines our metric for success, 
      3. n_jobs=-1 tells scikit-learn to use every core available.

- Reference:
  - ML Cookbook

[**Day48 and 49 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6877215989675778048-6dWa)

**💡 Creating a Baseline Regression Model for Model Evaluation**: 

- "DummyRegressor" allows us to create a very simple model that we can use as a baseline to compare against our actual model. This can often be useful to simulate a “naïve” existing prediction process in a product or system."DummyRegressor" uses the strategy parameter to set the method of making predictions, including the mean or median value in the training set. 
- Furthermore, if we set strategy to constant and use the constant parameter, we can set the dummy regressor to predict some constant value for every observation. By default, score returns the coefficient of determination (R-squared, R2) score. The closer R2 is to 1, the more of the variance in the target vector that is explained by the features.

**💡 Creating a Baseline Classification Model for Model Evaluation**: 

- A common measure of a classifier’s performance is how much better it is than random guessing. "DummyClassifier" makes this comparison easy. The strategy parameter gives us a number of options for generating values. There are two particularly useful strategies:

      First, stratified makes predictions that are proportional to the training set’s target vector’s class proportions.
      Second, uniform will generate predictions uniformly at random between the different classes.

- Reference:
  - ML Cookbook

[**Day50 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6877584384191545345-1KIq)

**💡 Back propagation**: 

- Back propagation, in short, it is simply Gradient Descent using an effective technique for computing the gradients automatically: in just two passes through the network (one forward, one backward) the backpropagation algorithm is able to compute the gradient of the network’s error with regards to every single model parameter. Backpropagation is done in following steps:

1. It handles one mini-batch at a time (for example containing 32 instances each), and it goes through the full training set multiple times. Each pass is called an epoch

2. Each mini-batch is passed to the network’s input layer, which just sends it to the first hidden layer. The algorithm then computes the output of all the neurons in this layer. The result is passed on to the
next layer, its output is computed and passed to the next layer, and so on until we get the output of the last layer, the output layer. This is the forward pass.

3. Next, the algorithm measures the network’s output error (i.e., it uses a loss function that compares the desired output and the actual output of the network, and returns some measure of the error).

4. Then it computes how much each output connection contributed to the error. This is done analytically by simply applying the chain rule, which makes this step fast and precise.

5. The algorithm then measures how much of these error contributions came from each connection in the layer below, again using the chain rule and so on until the algorithm reaches the input layer.

6. This reverse pass efficiently measures the error gradient across all the connection weights in the network by propagating the error gradient backward through the network.

7. Finally, the algorithm performs a Gradient Descent step to tweak all the connection weights in the network, using the error gradients it just computed.

- It is important to initialize all the hidden layers connection weights randomly or else training will fail. In other words, despite having hundreds of neurons per layer, your model will act as if it had only one neuron per layer: it won’t be too smart.If instead you randomly initialize the weights, you break the symmetry and allow backpropagation to train a diverse team of neurons.

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day51 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6877961938559741952-B6Sw)

**💡 Activation Function**: 

- Activation functions are mathematical equations that determine the output of a neural network model. Activation functions also have a major effect on the neural network’s ability to converge and the convergence speed, or in some cases, activation functions might prevent neural networks from converging in the first place. Activation function also helps to normalize the output of any input in the range between 1 to -1 or 0 to 1. Activation function must be efficient and it should reduce the computation time because the neural network sometimes trained on millions of data points.
- Types of Activation Functions:

      1. Binary Step Function
      2. Linear Activation Function
      3. Non-linear Activation Function

- Binary Step Function:
      It is basically a threshold base classifier, in this, we decide some threshold value to decide output that neuron should be activated or deactivated.  It thresholds the input values to 1 and 0, if they are greater or less than zero, respectively.
- Linear Activation Function:
      It is a simple straight line activation function where our function is directly proportional to the weighted sum of neurons or input. Linear activation functions are better in giving a wide range of activations and a line of a positive slope may increase the firing rate as the input rate increases.
- Non-Linear Activation Function:
      Modern neural network models use non-linear activation functions. They allow the model to create complex mappings between the network’s inputs and outputs, such as images, video, audio, and data sets that are non-linear or have high dimensionality. 

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day52 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6878313561433735168-jVQ-)

**💡 Sigmoid Function**: 

- Sigmoid Activation function is very simple which takes a real value as input and gives probability that ‘s always between 0 or 1. It looks like ‘S’ shape.It’s non-linear, continuously differentiable, monotonic, and has a fixed output range. Main advantage is simple and good for classifier. But Big disadvantage of the function is that it It gives rise to a problem of “vanishing gradients” because Its output isn’t zero centered. It makes the gradient updates go too far in different directions. 0 < output < 1, and it makes optimization harder. That takes very high computational time in hidden layer of neural network.

**💡 Softmax Function**:

- The softmax function is a function that turns a vector of K real values into a vector of K real values that sum to 1. The softmax transforms the input values into values between 0 and 1, so that they can be interpreted as probabilities. Generally, we use the function at last layer of neural network which calculates the probabilities distribution of the event over ’n’ different events. The main advantage of the function is able to handle multiple classes.

**💡 Tanh Function**:

- Tanh help to solve non zero centered problem of sigmoid function. Tanh squashes a real-valued number to the range [-1, 1]. Tanh has large area under better slope compared to sigmoid, this helps models using Tanh activation to learn better. Derivative function give us almost same as sigmoid derivative function. It solve sigmoid’s drawback but it still can’t remove the vanishing gradient problem completely.

[**Day53 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6878676950211280896-a94E)

**💡 ReLU Function**: 

- ReLU stands for rectified linear activation unit. It is simple yet really better than its predecessor activation functions such as sigmoid or tanh. ReLU function is its derivative both are monotonic. The function returns 0 if it receives any negative input, but for any positive value x, it returns that value back. Thus it gives an output that has a range from 0 to infinity. The model can, therefore, take less time to train or run. One more important property that we consider the advantage of using ReLU activation function is sparsity. Sparsity results in concise models that often have better predictive power and less overfitting/noise. In a sparse network, it’s more likely that neurons are actually processing meaningful aspects of the problem.

**💡 Leaky ReLU Function**:

- Leaky ReLU function is an improved version of the ReLU activation function. As for the ReLU activation function, the gradient is 0 for all the values of inputs that are less than zero, which would deactivate the neurons in that region and may cause dying ReLU problem.Leaky ReLU is defined to address this problem. Instead of defining the ReLU activation function as 0 for negative values of inputs(x), we define it as an extremely small linear component of x.
- This function returns x if it receives any positive input, but for any negative value of x, it returns a really small value which is 0.01 times x. Thus it gives an output for negative values as well. By making this small modification, the gradient of the left side of the graph comes out to be a non zero value. Hence we would no longer encounter dead neurons in that region.

[**Day54 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedin-datawithvikram-activity-6879039241662402561-PKaR)

**💡 Regression MLP**: 

- MLPs can be used for regression tasks. If you want to predict a single value, then you just need a single output neuron: its output is the predicted value. For multivariate regression, you need one output neuron per output dimension. For example, to locate the center of an object on an image, you need to predict 2D coordinates, so you need two output neurons.
- In general, when building an MLP for regression, you do not want to use any activation function for the output neurons, so they are free to output any range of values. However, if you want to  guarantee that the output will always be positive, then you can use the ReLU activation function in the output layer.If we want to guarantee that the predictions will fall within a given range of values, then you can use the logistic function or the hyperbolic tangent, and scale the labels to the appropriate range: 0 to 1 for the logistic function, or –1 to 1 for the hyperbolic tangent.
- The loss function to use during training is typically the mean squared error, but if you have a lot of outliers in the training set, you may prefer to use the mean absolute error instead.  Alternatively, you can use the Huber loss, which is a combination of both.

**💡 Typical Regression MLP Architecture**: 
      
      1. Input neurons: One per input feature 
      2. Hidden layers: Depends on the problem(Typically 1 to 5)
      3. Neurons per hidden layer: Depends on the problem (10 to 100)
      4. Output neurons: 1 per prediction dimension
      5. Hidden activation:  ReLU 
      6. Output activation: None or ReLU/Softplus (if positive outputs) or Logistic/Tanh (if bounded outputs)
      7. Loss function MSE or MAE/Huber (if outliers)

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day55 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6879404504828264449-4wS2)

**💡 Classification MLP**: 

- MLPs can also be used for classification tasks. For a binary classification problem, you just need a single output neuron using the logistic activation function: the output will be a number between 0 and 1. MLPs can also easily handle multilabel binary classification tasks. For example, An email classification system that predicts whether each incoming email is ham or spam, and simultaneously predicts whether it is an urgent or non-urgent email.
-In the above scenario, need two output neurons, both using the logistic activation function: the first would output the probability that the email is spam and the second would output the probability that it is urgent. In general, we can say that we dedicate one output neuron for each positive class. Note that the output probabilities do not necessarily add up to one. This lets the model output any combination of labels.
- If each instance can belong only to a single class, out of 3 or more possible classes, then we need to have one output neuron per class, and we will use the softmax activation function for the whole output layer. The softmax function will ensure that all the estimated probabilities are between 0 and 1 and that they add
up to one (which is required if the classes are exclusive). This is called multiclass classification. Regarding the loss function, since we are predicting probability distributions, the cross-entropy (also called the log loss) is generally a good choice.

**💡 Typical Classification (Multiclass) MLP Architecture**:
1. Input layers: One per input feature
2. Hidden layers: Depends on the problem(Typically 1 to 5)
3. Neurons per hidden layer: Depends on the problem (10 to 100)
4. Output neurons: 1 per class
5. Output activation: Softmax
6. Loss function : Cross-Entropy

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day56 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6879768505911259136-sRRv)

**💡 Vanishing Gradients Problem**: 

- It describes the situation where a deep multilayer feed-forward network or an RNN is unable to propagate useful gradient information from the output end of the model back to the layers near the input end of the model. The result is the general inability of models with many layers to learn on a given dataset, or for models with many layers to prematurely converge to a poor solution.
- Certain activation functions, like the sigmoid function, squishes a large input space into a small input space between 0 and 1. Therefore, a large change in the input of the sigmoid function will cause a small change in the output. Hence, the derivative becomes small.For shallow network with only a few layers that use these activations, this isn’t a big problem. However, when more layers are used, it can cause the gradient to be too small for training to work effectively. 

**💡 Methods proposed to overcome vanishing gradient problem**:
      
      1. Multi-level hierarchy
      2. Long short – term memory
      3. Faster hardware
      4. Residual neural networks (ResNets)
      5. ReLU
- The simplest solution is to use other activation functions, such as ReLU, which doesn’t cause a small derivative. Residual networks are another solution, as they provide residual connections straight to earlier layers. 
- 
- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day57 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6880124258983522304-8EDv)

**💡 Exploding Gradients Problem**: 

- Exploding gradients are a problem when large error gradients accumulate and result in very large updates to neural network model weights during training. Gradients are used during training to update the network weights, but when the typically this process works best when these updates are small and controlled. A gradient calculates the direction and magnitude during the training of a neural network and it is used to teach the network weights in the right direction by the right amount. When there is an error gradient, explosion of components may grow exponentially.
- When the magnitudes of the gradients accumulate, an unstable network is likely to occur, which can cause poor prediction results or even a model that reports nothing useful what so ever. 
- There are some signs which are helpful in determining whether model is suffering from exploding gradients during training of network

      1. The model does not learn much on training data therefore resulting in poor loss.
      2. The model weights go to NaN during training.
      3. During training, the model weights grow exponentially and become very large.
      4. The error gradient values are always above 1.
- There are many approaches to fix exploding gradients but some of the best approaches are:

      1. Use LSTM network
      2. Use Gradient clipping
      3. Use Regularization(like L2 norm)
      4. Redesign the neural network

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day58 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6880491869734871040-rzCL)

**💡 Batch Normalization**: 

- Although using He initialization along with ELU (or any variant of ReLU) can significantly reduce the vanishing/exploding gradients problems at the beginning of training, it doesn’t guarantee that they won’t come back during training. In a 2015 Sergey Ioffe and Christian Szegedy proposed a technique called Batch Normalization (BN) to address the vanishing/exploding gradients problems. The technique consists of adding an operation in the model just before or after the activation function of each hidden layer, simply zero-centering and normalizing each input, then scaling and shifting the result using two new parameter vectors per layer: one for scaling, the other for shifting.
- In other words, this operation lets the model learn the optimal scale and mean of each of the layer’s inputs. In many cases, if you add a BN layer as the very first layer of your neural network, you do not need to standardize your training set : the BN layer will do it for you. In order to zero-center and normalize the inputs, the algorithm needs to estimate each input’s mean and standard deviation. It does so by evaluating the mean and standard deviation of each input over the current mini-batch (hence the name “Batch Normalization”).

**💡 Advantages of Batch Normalization**:

- By Normalizing the hidden layer activation the Batch normalization speeds up the training process
- It solves the problem of internal covariate shift. Through this, we ensure that the input for every layer is distributed around the same mean and standard deviation.
- Batch normalization smoothens the loss function that in turn by optimizing the model parameters improves the training speed of the model.

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day59 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6880855940623679488-m8V2)

**💡 Gradient Clipping**: 

- Another popular technique to lessen the exploding gradients problem is to simply clip the gradients during backpropagation so that they never exceed some threshold. This is called Gradient Clipping. This technique is most often used in recurrent neural networks, as Batch Normalization is tricky to use in RNNs.In Keras, implementing Gradient Clipping is just a matter of setting the clipvalue or clipnorm argument when creating an optimizer.
- This will clip every component of the gradient vector to a value between –1.0 and 1.0. This means that all the partial derivatives of the loss will be clipped between –1.0 and 1.0. The threshold is a hyperparameter you can tune.Note that it may change the orientation of the gradient vector. In practice however, this approach works well. If you want to ensure that Gradient Clipping does not change the direction of the gradient vector, you should clip by norm by setting clipnorm instead of clipvalue. This will clip the whole gradient if its ℓ2 norm is greater than the threshold you picked.
- If you observe that the gradients explode during training (you can track the size of the gradients using TensorBoard), you may want to try both clipping by value and clipping by norm, with different threshold, and see which option performs best on the validation set.

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day60 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6881209281090326528-Rdte)

**💡 Transfer learning**: 

- It is generally not a good idea to train a very large DNN from scratch: instead, you should always try to find an existing neural network that accomplishes a similar task to the one you are trying to tackle, then just reuse the lower layers of this network.This is called transfer learning. It will not only speed up training considerably, but will also require much less training data.
- If the input pictures of your new task don’t have the same size as the ones used in the original task, you will usually have to add a preprocessing step to resize them to the size expected by the original model. More generally, transfer learning will work best when the inputs have similar low-level features.The output layer of the original model should usually be replaced since it is most likely not useful for the current task. Similarly, the upper hidden layers of the original model are less likely to be as useful as the lower layers, since the high-level features that are most useful for the new task may differ significantly from the ones that were most useful for the original task.
- The more similar the tasks are, the more layers you want to reuse. For very similar tasks, you can try keeping all the hidden layers and just replace the output layer.Try freezing all the reused layers first, then train your model and see how it performs. Then try unfreezing one or two of the top hidden layers to let backpropagation tweak them and see if performance improves. It is also useful to reduce the learning rate when you unfreeze reused layers: this will avoid wrecking their fine-tuned weights.

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day61 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6881594129164451840-lXBm)

**💡 Momentum Optimization**: 

- Imagine a bowling ball rolling down a gentle slope on a smooth surface: it will start out slowly, but it will quickly pick up momentum until it eventually reaches terminal velocity. This is the very simple idea behind Momentum optimization. An Adaptive Optimization Algorithm uses exponentially weighted averages of gradients over previous iterations to stabilize the convergence, resulting in quicker optimization.
- Momentum optimization cares a great deal about what previous gradients were: at each iteration, it subtracts the local gradient from the momentum vector m (multiplied by the learning rate η), and it updates the weights by simply adding this momentum vector. Gradient Descent goes down the steep slope quite fast, but then it takes a very long time to go down the valley. In contrast, momentum optimization will roll down the valley faster and faster
until it reaches the bottom (the optimum).
- In deep neural networks that don’t use Batch Normalization, the upper layers will often end up having inputs with very different
scales, so using Momentum optimization helps a lot. It can also help roll past local minima.The one drawback of Momentum optimization is that it adds yet another hyperparameter to tune. However, the momentum value of 0.9 usually works well in practice and almost always goes faster than regular Gradient Descent.

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day62 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6881939481172279296-3-Vo)

**💡 Nesterov Accelerated Gradient**: 

- The idea of Nesterov Momentum optimization, or Nesterov Accelerated Gradient (NAG), is to measure the gradient of the cost function not at the local position but slightly ahead in the direction of the momentum. The only difference from vanilla Momentum optimization is that the gradient is measured at θ + βm rather than at θ.
- This small tweak works because in general the momentum vector will be pointing in the right direction (i.e., toward the optimum), so it will be slightly more accurate to use the gradient measured a bit farther in that direction rather than using the gradient at the original position. After a while, these small improvements add up and NAG ends up being significantly faster than regular Momentum optimization. Moreover, note that when the momentum pushes the weights across a valley, ∇1(Regular Momentum Update) continues to push further across the valley, while ∇2 (Nesterov Update) pushes back toward the bottom of the valley. This helps reduce oscillations and thus converges faster.
- NAG will almost always speed up training compared to regular Momentum optimization. To use it, simply set nesterov=True when creating the SGD optimizer

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day63 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6882305889710305280-j_Yz)

**💡 Adagrad Optimizer**: 

- Consider the elongated bowl problem, Gradient Descent starts by quickly going down the steepest slope, then slowly goes down the bottom of the valley. It would be nice if the algorithm could detect this early on and correct its direction to point a bit more toward the global optimum. The AdaGrad algorithm achieves this early compared to Gradient Descent by scaling down the gradient vector along the steepest dimensions.
- This algorithm decays the learning rate, but it does so faster for steep dimensions than for dimensions with gentler slopes. This is called an adaptive learning rate.It helps point the resulting updates more directly toward the global optimum. One additional benefit is that it requires much less tuning of the learning rate hyperparameter η
- AdaGrad often performs well for simple quadratic problems, but unfortunately it often stops too early when training neural networks. The learning rate gets scaled down so much that the algorithm ends up stopping entirely before reaching the global optimum. So even though Keras has an Adagrad optimizer, you should not use it to train deep neural networks

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day64 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6882664056126369792-Rssr)

**💡 RMSProp**: 

- RMSProp, which stands for Root Mean Square Propagation, is a gradient descent optimization algorithm. RMSProp was developed in order to overcome the short comings of the AdaGrad algorithm. That is, RMSProp does not decay the learning rate too quickly preventing convergence.
- RMSProp makes use of exponential decay in order to manage the size of the vector(S). Unlike AdaGrad, RMSProp decays the contribution of older gradients at each step. This prevents the magnitude of (S) from becoming so large that it prevents learning.As a result of this exponential decay, the accumulated gradients in (S)  are focused on recent gradients as a opposed to all previous gradients.
- The hyperparameter β  is known as the decay rate. This hyperparameter is used to control the focus of the adaptive learning rate on more recent gradients.In almost all cases RMSProp will outperform AdaGrad. As a result of this RMSProp was the preferred optimization algorithm until the Adam optimization algorithm was introduced.

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day65 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6883025023305699328-EMI6)

**💡 Adam Optimization**: 

- Adam, which stands for adaptive moment estimation, combines the ideas of Momentum optimization and RMSProp: just like Momentum optimization it keeps track of an exponentially decaying average of past gradients, and just like RMSProp it keeps track of an exponentially decaying average of past squared gradients.Instead of adapting the parameter learning rates based on the average first moment (the mean) as in RMSProp, Adam also makes use of the average of the second moments of the gradients (the uncentered variance).
- Specifically, the algorithm calculates an exponential moving average of the gradient and the squared gradient, and the parameters beta1 and beta2 control the decay rates of these moving averages. It takes two parameters: momentum decay hyperparameter β1 and scaling decay hyperparameter β2. The momentum decay hyperparameter β1 is typically initialized to 0.9, while the scaling decay hyperparameter β2 is often initialized to 0.999
- Since Adam is an adaptive learning rate algorithm (like AdaGrad and RMSProp), it requires less tuning of the learning rate hyperparameter η. You can often use the default value η = 0.001, making Adam even easier to use than Gradient Descent.
- Nadam Optimization is an important variant of Adam optimization, which is simply Adam optimization plus the Nesterov trick, so it will often converge slightly faster than Adam. In his report, Timothy Dozat compares many different optimizers on various tasks and finds that Nadam generally outperforms Adam, but is sometimes outperformed by RMSProp

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow

[**Day66 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-linkedinhardmode-datawithvikram-activity-6883384573456195584-2L9K)

**💡 Training Sparse models**: 

- All the optimization algorithms just presented produce dense models, meaning that most parameters will be nonzero. If you need a blazingly fast model at runtime, or if you need it to take up less memory, you may prefer to end up with a sparse model instead. One trivial way to achieve this is to train the model as usual, then get rid of the tiny weights (set them to 0). However, this will typically not lead to a very sparse model, and it may degrade the model’s performance.
- A better option is to apply strong ℓ1 regularization during training, as it pushes the optimizer to zero out as many weights as it can. However, in some cases, these techniques may remain insufficient. One last option is to apply Dual Averaging, often called Follow The Regularized Leader (FTRL), a technique proposed by Yurii Nesterov. 20 When used with ℓ1 regularization, this technique often leads to a very sparse model.

**💡 Learning Rate Scheduling**: 

- Finding a good learning rate can be tricky. If you set it way too high, training may actually diverge. If you set it too low, training will eventually converge to the optimum, but it will take a very long time. If you set it slightly too high, it will make progress very quickly at first, but it will end up dancing around the optimum, never really settling down.If you have a limited computing budget, you may have to interrupt training before it has converged properly, yielding a suboptimal solution
- We can do better than a constant learning rate: if you start with a high learning rate and then reduce it once it stops making fast progress, you can reach a good solution faster than with the optimal constant learning rate.There are many different strategies to reduce the learning rate during training. These strategies are called learning schedules. Some of the popular are:

      1. Power Scheduling
      2. Exponential Scheduling
      3. Piecewise Constant Scheduling
      4. Performance Scheduling

- Reference:
  - Hands-On Machine Learning with Scikit-Learn, Keras and Tensor Flow
