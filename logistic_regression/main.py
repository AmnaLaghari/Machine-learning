import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# You can grab the dataset directly from scikit-learn with load_digits(). It returns a tuple of the inputs and output:
x, y = load_digits(return_X_y=True)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Scale data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

# Create a model and train it
model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',
                           random_state=0)
model.fit(x_train, y_train)

# Evaluate the model
x_test = scaler.transform(x_test)
y_pred = model.predict(x_test)

# print(y_pred)

# You can obtain the accuracy with .score():
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))