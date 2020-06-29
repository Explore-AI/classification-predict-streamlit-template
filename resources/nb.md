## Naive Bayes  

### The basic concept  

A Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. It uses the prior probability, as well as likelihood of an observation to be in a class in calculating the posterior probability of the observation being in that class. The class that has the highest posterior probability in the end, is the class to which the observation is classified to.  

*Posterior probability* - the probability of an observation being in class A, given a certain input (tweets in our case)

- **Pro:** Assumes that all features are independent, which makes this method very fast compared to more complicated methods. *In some cases, speed is actually preferred over higher accuracy.*
- **Con:** The opposite can be argued here. Since the assumption of independence does not usually hold in real life, the method has some of the lowest accuracies of the models.
