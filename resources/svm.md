## Support Vector Machines (SVM) 

### The basic concept  

Rather than modeling a response (sentiment in our case) directly, logistic regression models the probability that this response belongs to a particular category/class. Logistic regression uses the sigmoid curve we can see below. Using a threshold of your choice, any observation falling below this threshold gets classified to class A. Conversely, any observation that falls above this threshold gets classified to class B.

- **Pro:** This method is scalable to large data and is also one of the more computationally efficient models.
- **Con:** Although it could be extended to multi-class problems (as is the case here), logistic regression is ideally used for binary classification.
