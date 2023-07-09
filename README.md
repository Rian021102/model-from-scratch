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

## 2. Logistic Regression Function
Logistic regression is a statistical model that uses the logistic function, or logit function, in mathematics as the equation between x and y. The logit function maps y as a sigmoid function of x.

![image](https://github.com/Rian021102/model-from-scratch/assets/108880724/e802d0d6-e5fd-41eb-8df4-56db01552cec)

when you plot this function you will have S-curve

![image](https://github.com/Rian021102/model-from-scratch/assets/108880724/ed146ace-3a35-418f-9aa1-ec91fe9f2e2a)

Now if we take how we find y values in linear regression:
y=w.X + b,

where
w=weight
b=bias

then subtitute the linear equation to logistic regression will give us:

![image](https://github.com/Rian021102/model-from-scratch/assets/108880724/a415406b-ecff-4451-88bd-95601b13ce54)

## 3. Loss/Cost Function
In parametric machine learning algorithm, loss/cost function is needed, of which we want to minimize (to find global minimum of) to determine the optimal parameters of w and b, therefore we will have the best predictions. In linear regression, the cost function is Mean Square Error (MSE):


![image](https://github.com/Rian021102/model-from-scratch/assets/108880724/0f0d6db9-915a-4c13-8f18-1ed5afbd9e2a)

which result when plotted:


![image](https://github.com/Rian021102/model-from-scratch/assets/108880724/f5797751-1144-4c1c-af9a-862dabc2f7a6)

However, when we use logistic regression, the cost function will result like this when plotted:

![image](https://github.com/Rian021102/model-from-scratch/assets/108880724/1968f9e4-ba68-4692-9277-0449d62da002)


For binary classification problem, we need to be able for output to being either 0 or 1, hence the cost function for logistic regression becomes:


![image](https://github.com/Rian021102/model-from-scratch/assets/108880724/f707e4d2-5409-4c90-a7bb-31da7af04ef3)

where after subtituting the cost function with log function, the cost function for logistic regression becomes:

![image](https://github.com/Rian021102/model-from-scratch/assets/108880724/e691d667-7b3f-4c27-8950-c6564e60883c)



















                              










# sources:
1. https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/what-is-logistic-regression/
2. https://aws.amazon.com/what-is/logistic-regression/
3. https://www.analyticsvidhya.com/blog/2020/11/binary-cross-entropy-aka-log-loss-the-cost-function-used-in-logistic-regression/
