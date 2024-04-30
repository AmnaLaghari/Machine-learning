
#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

startup_df=pd.read_csv(r'./multivariated_regresion/data.csv')

# step1: define x and y
x=startup_df.iloc[:,:4]
y=startup_df.iloc[:,4]


# step2: perform ont hot encoding
ohe=OneHotEncoder(sparse_output=False)
xy=ohe.fit_transform(startup_df[['State']])

# step3: change columns using column transfer
col_trans=make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'),['State']),
    remainder='passthrough')
x=col_trans.fit_transform(x)


# step4: split dataset into train set and test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# step5: train the model
linreg=LinearRegression()
linreg.fit(x_train,y_train)

# step6: predict the resulyt
y_pred=linreg.predict(x_test)
print('------------------Predicted Values------------------------------')
print(y_pred)

# step7: evaluate the model
Accuracy=r2_score(y_test,y_pred)*100
print('---------------------Accuracy---------------------------')
print(" Accuracy of the model is %.2f" %Accuracy)


#plot results
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()


pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})
print('--------------------Difference----------------------------')
print(pred_df)