
# XGBoost - Lab

## Introduction

In this lab, we'll install the popular [XGBoost](http://xgboost.readthedocs.io/en/latest/index.html) library and explore how to use this popular boosting model to classify different types of wine using the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) from the UCI Machine Learning Dataset Repository.  

## Objectives

You will be able to:

- Fit, tune, and evaluate an XGBoost algorithm

## Installing XGBoost

The XGBoost model is not currently included in scikit-learn, so we'll have to install it on our own.  To install XGBoost, you'll need to use `conda`. 

To install XGBoost, follow these steps:

1. Open up a new terminal window 
2. Activate your conda environment
3. Run `conda install py-xgboost`. You must use `conda` to install this package -- currently, it cannot be installed using `pip`  
4. Once installation has completed, run the cell below to verify that everything worked 


```python
from xgboost import XGBClassifier
```

Run the cell below to import everything we'll need for this lab. 


```python
import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```

The dataset we'll be using for this lab is currently stored in the file `'winequality-red.csv'`.  

In the cell below, use Pandas to import the dataset into a dataframe, and inspect the `.head()` of the dataframe to ensure everything loaded correctly. 


```python
df = None
```

For this lab, our target column will be `'quality'`.  That makes this a multiclass classification problem. Given the data in the columns from `'fixed_acidity'` through `'alcohol'`, we'll predict the quality of the wine.  

This means that we need to store our target variable separately from the dataset, and then split the data and labels into training and test sets that we can use for cross-validation. 

In the cell below:

- Assign the `'quality'` column to `y` 
- Drop this column (`'quality'`) and assign the resulting DataFrame to `X` 
- Split the data into training and test sets. Set the `random_state` to 42   


```python
y = None
X = None

X_train, X_test, y_train, y_test = None
```

Now that you have prepared the data for modeling, you can use XGBoost to build a model that can accurately classify wine quality based on the features of the wine!

The API for xgboost is purposefully written to mirror the same structure as other models in scikit-learn.  


```python
# Instantiate XGBClassifier
clf = None

# Fit XGBClassifier


# Predict on training and test sets
training_preds = None
test_preds = None

# Accuracy of training and test sets
training_accuracy = None
test_accuracy = None

print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))
print('Validation accuracy: {:.4}%'.format(test_accuracy * 100))
```

## Tuning XGBoost

The model had somewhat lackluster performance on the test set compared to the training set, suggesting the model is beginning to overfit to the training data. Let's tune the model to increase the model performance and prevent overfitting. 

You've already encountered a lot of parameters when working with Decision Trees, Random Forests, and Gradient Boosted Trees.

For a full list of model parameters, see the [XGBoost Documentation](http://xgboost.readthedocs.io/en/latest/parameter.html).

Examine the tunable parameters for XGboost, and then fill in appropriate values for the `param_grid` dictionary in the cell below. 

**_NOTE:_** Remember, `GridSearchCV` finds the optimal combination of parameters through an exhaustive combinatoric search.  If you search through too many parameters, the model will take forever to run! To ensure your code runs in sufficient time, we restricted the number of values the parameters can take.  


```python
param_grid = {
    'learning_rate': [0.1, 0.2],
    'max_depth': [6],
    'min_child_weight': [1, 2],
    'subsample': [0.5, 0.7],
    'n_estimators': [100],
}
```

Now that we have constructed our `params` dictionary, create a `GridSearchCV` object in the cell below and use it to iterate tune our XGBoost model.  

Now, in the cell below:

* Create a `GridSearchCV` object. Pass in the following parameters:
    * `clf`, the classifier
    * `param_grid`, the dictionary of parameters we're going to grid search through
    * `scoring='accuracy'`
    * `cv=None`
    * `n_jobs=1`
* Fit our `grid_clf` object and pass in `X_train` and `y_train`
* Store the best parameter combination found by the grid search in `best_parameters`. You can find these inside the grid search object's `.best_params_` attribute 
* Use `grid_clf` to create predictions for the training and test sets, and store them in separate variables 
* Compute the accuracy score for the training and test predictions  


```python
grid_clf = None
grid_clf.fit(None, None)

best_parameters = None

print('Grid Search found the following optimal parameters: ')
for param_name in sorted(best_parameters.keys()):
    print('%s: %r' % (param_name, best_parameters[param_name]))

training_preds = None
test_preds = None
training_accuracy = None
test_accuracy = None

print('')
print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))
print('Validation accuracy: {:.4}%'.format(test_accuracy * 100))
```

## Summary

Great! You've now successfully made use of one of the most powerful boosting models in data science for modeling.  We've also learned how to tune the model for better performance using the grid search methodology we learned previously. XGBoost is a powerful modeling tool to have in your arsenal. Don't be afraid to experiment with it! 
