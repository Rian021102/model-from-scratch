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

### A. Loss/Cost Function
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

### B. Gradient Descent
For optmization logistic regression cost function, we use gradient descent, which updates the parameters based on the gradients of the cost function with respect to the parameters by minimizing cost function, with function as follow:



![image](https://github.com/Rian021102/model-from-scratch/assets/108880724/7adc4ea7-1422-4f0b-9cb0-1b9c709994a7)



## 2. Put all the equations into code


### A. Pseudocode

```

class LogitRegression05:
    initialize:
        learning_rate = 0.001
        iterations = 1000
```
    fit(X, Y):
        randomly select "k" features from total "m" features
        where k << m
        repeat until "l" number of nodes has been reached:
            among the "k" features, calculate the node "d" using the best split point
            split the node into daughter nodes using the best split

        build forest by repeating for "n" number of times:
            randomly select "k" features from total "m" features
            where k << m
            repeat until "l" number of nodes has been reached:
                among the "k" features, calculate the node "d" using the best split point
                split the node into daughter nodes using the best split
          
    predict(X):
        randomly select "k" features from total "m" features
        where k << m
        among the "k" features, calculate the node "d" using the best split point
        return the prediction based on the random forest model




### B. Code in Python

first we initialize the function where:

```

class LogitRegression05:
    initialize:
        learning_rate = 0.001
        iterations = 1000
    
```

This is where:
calculates the cost of the logistic regression model. It first computes the activation values A using the sigmoid function applied to the linear combination of input features (self.X), weights (self.W), and bias (self.b). Then, it computes the cost using the logistic regression cost function formula. It calculates the negative log-likelihood of the predicted probabilities A compared to the true labels self.Y.

The update_weights method performs the weight and bias updates based on the gradients of the cost function. It first computes the activations A using the same sigmoid function. Then, it calculates the difference between the predicted activations and the true labels (tmp). It reshapes tmp to match the shape of the weights. Next, it calculates the gradients of the weights (dW) and bias (db) using matrix multiplication and summation operations. Finally, it updates the weights and bias by subtracting the learning rate (self.learning_rate) multiplied by the gradients.


    def update_weights(self):
        A = 1 / (1 + np.exp(-(self.X.dot(self.W) + self.b)))
        tmp = (A - self.Y.T)
        tmp = np.reshape(tmp, self.m)
        dW = np.dot(self.X.T, tmp) / self.m
        db = np.sum(tmp) / self.m
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

The fit method is used to train the logistic regression model. It takes input features X and corresponding labels Y. Inside the method, the shape of the input data is determined to get the number of training examples (self.m) and the number of features (self.n). The weights self.W and bias self.b are initialized with zeros. The input data X and labels Y are stored in variables self.X and self.Y respectively. self.cost_history is an empty list that will store the cost at each iteration during training.


        for i in range(self.iterations):
            cost = self.compute_cost()
            self.cost_history.append(cost)
            self.update_weights()

The predict method is used to make predictions using the trained model. It takes input features X and computes the activations Z using the sigmoid function. It then sets the predicted labels Y to 1 where the activations are greater than 0.5, and 0 otherwise.
    def predict(self, X):

    
        Z = 1 / (1 + np.exp(-(X.dot(self.W) + self.b)))
        Y = np.where(Z > 0.5, 1, 0)
        return Y

The score method calculates the accuracy of the model's predictions. It takes input features X and true labels y. It uses the predict method to get the predicted labels and compares them with the true labels. It calculates the accuracy as the ratio of the number of correct predictions to the total number of examples.


    def score(self, X, y=None):
        predictions = self.predict(X)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy

## 3. Train Logistic Regression with dataset
For testing the code, we will use simple dataset, diabetes.csv ( . /data/diabetes.csv). The code is tested inside the main.py ( . /logistic_regression/app.py). The accuracy with default parameter (learning rate = 0.001 and iterations = 1000) is 68%. Then, we tuned the model using grid search and randomized search, where accuracy could improve to 69%.

## 4. Summary
In this project we have gone through logistic regression including the logistic regression (sigmoid), and optimization using cost function with optimization of cost function using gradient descent. The last step is to put all these equations into logistic regression code hence we can call the function to train the data and help us to predict the outcome for classification.




# sources:
1. https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/what-is-logistic-regression/
2. https://aws.amazon.com/what-is/logistic-regression/
3. https://www.analyticsvidhya.com/blog/2020/11/binary-cross-entropy-aka-log-loss-the-cost-function-used-in-logistic-regression/
