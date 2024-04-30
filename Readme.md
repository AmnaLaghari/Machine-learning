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

1. **Classification:** Classification is a type of supervised machine learning where algorithms learn from the data to predict an outcome or event in the future. Classification. There are many machine learning algorithms that can be used for classification tasks. Some of them are:
   - Logistic Regression
   - Decision Tree Classifier
   - K Nearest Neighbor Classifier
   - Random Forest Classifier
   - Neural Networks
2. **Regression:** Regression is a type of supervised machine learning where algorithms learn from the data to predict continuous values such as sales, salary, weight, or temperature.
   There are many machine learning algorithms that can be used for regression tasks. Some of them are:
   - Linear Regression
   - Decision Tree Regressor
   - K Nearest Neighbor Regressor
   - Random Forest Regressor
   - Neural Networks

### Multivariated Regression

Multivariate regression is a technique that helps us assess how different independent variables relate linearly to multiple dependent variables. We use the term "linear" because it reflects the correlations between these variables. After applying multivariate regression to the dataset, we can predict the behavior of a response variable based on its associated predictor variables.
