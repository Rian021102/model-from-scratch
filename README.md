# Build Model From Scratch : Logistic Regression Model From Scratch
## 1. Logistic Regression, What is it?
Logistic regression is a statistical regression model used to predict the probability of a binary or categorical outcome. It is commonly used in machine learning and statistics for binary classification problems, where the goal is to assign an observation to one of two possible classes.

The key idea behind logistic regression is to model the relationship between the predictors (also known as independent variables or features) and the probability of a particular outcome. Unlike linear regression, which predicts continuous numeric values, logistic regression predicts the probability of an outcome falling into a specific category.

The logistic regression model uses a logistic function, also called the sigmoid function, to map the linear combination of predictors to a probability value between 0 and 1. The sigmoid function has an S-shaped curve and effectively transforms the linear regression output into a probability.

Mathematically, the logistic regression model can be represented as:

p = 1 / (1 + e^(-z))

where p is the probability of the outcome, z is the linear combination of predictors, and e is the base of the natural logarithm. The linear combination of predictors can be written as:

z = b0 + b1*x1 + b2*x2 + ... + bn*xn

where b0, b1, b2, ..., bn are the coefficients of the model, and x1, x2, ..., xn are the values of the predictors.

To estimate the coefficients in logistic regression, a method called maximum likelihood estimation is typically used. The goal is to find the coefficients that maximize the likelihood of the observed outcomes given the predictors.

Once the coefficients are estimated, the logistic regression model can be used to make predictions by calculating the probability of the outcome for new observations based on their predictor values. A common approach is to use a threshold value (e.g., 0.5) to classify the predicted probabilities into the respective categories.

Logistic regression has various extensions, such as multinomial logistic regression for multi-class classification problems and ordinal logistic regression for ordered categorical outcomes. It is a popular and widely used algorithm due to its simplicity, interpretability, and effectiveness in many practical applications.







# sources:
1. https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/what-is-logistic-regression/
