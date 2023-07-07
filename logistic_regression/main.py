import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from logreg06 import LogitRegression06 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

df=pd.read_csv('/Users/rianrachmanto/pypro/project/model_from_scratch/data/diabetes.csv')
print(df.head())
#print missing values
print(df.isnull().sum())
#set X and y
X=df.drop('Outcome',axis=1)
y=df['Outcome']
#split data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#train model
model=LogitRegression06()
model.fit(X_train,y_train)
#predict
y_pred=model.predict(X_test)
#accuracy
print(accuracy_score(y_test,y_pred))

param_grid = {
    'learning_rate': [0.001, 0.01],
    'iterations': [1000, 2000 ]
}

# RandomizedSearchCV for hyperparameter tuning
randomized_cv = RandomizedSearchCV(model, param_distributions=param_grid, cv=3, n_iter=5)
randomized_cv.fit(X_train, y_train)

print("Best hyperparameters (RandomizedSearchCV):", randomized_cv.best_params_)
print("Best score (RandomizedSearchCV):", randomized_cv.best_score_)


# GridSearchCV for hyperparameter tuning
grid_cv = GridSearchCV(model, param_grid=param_grid, cv=3)
grid_cv.fit(X_train, y_train)

print("Best hyperparameters (GridSearchCV):", grid_cv.best_params_)
print("Best score (GridSearchCV):", grid_cv.best_score_)

# Predict using the best model from GridSearchCV
best_lr_model = grid_cv.best_estimator_
predictions = best_lr_model.predict(X_test)

# Evaluate the model
accuracy = np.sum(predictions == y_test) / len(y_test)
print("Accuracy:", accuracy)