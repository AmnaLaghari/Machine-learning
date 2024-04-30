# Machine Learning

Machine learning is a subfield of artificial intelligence that allows computers to learn without being explicitly programmed. This repository will focus on machine learning algorithms (supervised and unsupervised) in python. 

## Setup

The first thing to do is to clone the repository:

```bash 
git clone https://github.com/AmnaLaghari/machine-learning.git
```
Create a virtual environment to install dependencies in and activate it.

```bash
pipenv shell
```
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install packages from requirements.txt

```bash
pip install -r requirements.txt
```

## Supervized machine learning:

Supervised Learning is a machine learning method that needs supervision similar to the student-teacher relationship. In supervised Learning, a machine is trained with well-labeled data, which means some data is already tagged with correct outputs. So, whenever new data is introduced into the system, supervised learning algorithms analyze this sample data and predict correct outputs with the help of labeled data. There are two types of supervised learning

1. __Classification:__ Classification is a type of supervised machine learning where algorithms learn from the data to predict an outcome or event in the future. Classification. There are many machine learning algorithms that can be used for classification tasks. Some of them are:
   * Logistic Regression
   * Decision Tree Classifier
   * K Nearest Neighbor Classifier
   * Random Forest Classifier
   * Neural Networks
2. __Regression:__ Regression is a type of supervised machine learning where algorithms learn from the data to predict continuous values such as sales, salary, weight, or temperature.
There are many machine learning algorithms that can be used for regression tasks. Some of them are:
   * Linear Regression
   * Decision Tree Regressor
   * K Nearest Neighbor Regressor
   * Random Forest Regressor
   * Neural Networks


### Multivariated Regression

Multivariate regression is a technique that helps us assess how different independent variables relate linearly to multiple dependent variables. We use the term "linear" because it reflects the correlations between these variables. After applying multivariate regression to the dataset, we can predict the behavior of a response variable based on its associated predictor variables.

### Logistic Regression

Logistic regression is a classification algorithm used when the target variable is binary or categorical. It models the relationship between the input features and the probability of belonging to a certain class. Logistic regression uses a logistic function (sigmoid function) to map the output to a probability value between 0 and 1. The decision boundary is typically set at a threshold value (e.g., 0.5) to classify instances into different classes.

### Random Forest

Random Forest is a popular and intuitive machine learning algorithm used for both classification and regression tasks; it is a powerful ensemble learning algorithm that combines multiple decision trees to make accurate predictions. It reduces overfitting by training each tree on random subsets of the data and features. Random Forest is robust, handles high-dimensional data well, and provides insights into important features. It finds applications in finance, healthcare, and image recognition, making it a popular algorithm in machine learning.