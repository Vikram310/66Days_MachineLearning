[**Day1 of 66DaysOfData!**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6921713172592697344-xstF?utm_source=linkedin_share&utm_medium=member_desktop_web)

**💡 Bias/Variance**: 
- If a model is underfitting(such as logistic regression of non linear data), it has "high bias". If the model is overfitting, then it has "high variance".
- We can get to know whether it has high bias or variance or not by checking the error of training, dev and test, for example:
    1. High Variance: Training error (1%) and Dev Error (11%)
    2. High Bias: Training Error (15%) and Dev Error(14%)
    3. High Bias and High Variance: Training Error(15%) and Test Error (30%)

-  If your algorithm has a high bias:
    1. Try to make the Neural network bigger (size of hidden units, number of layers)
    2. Try a different model that is suitable for your data
    3. Try to run it longer
    4. Different/Advanced Optimization Algorithms
- If your algorithm has high variance:
    1. Add more data
    2. Try regularization
    3. Try a different model that is more suitable for the data
    
- Reference:
  - [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)
