#Linear Regression-One predictor and one feature

import pandas as pd
import matplotlib.pyplot as plt
import sklearn
link="C:\\Users\\Aishwarya\\Downloads\\Various Excel file data for practice\\2_Marks_Data.csv"
df=pd.read_csv(link)
print(df)
print("Shape:",df.shape)
print("columns:",df.columns)

#Performing EDA(Explotatory data analysis) to understand the data
#Scatter plot
plt.scatter(df['Hours'],df['Marks'])
plt.xlabel('Hours')
plt.ylabel('Marks')
plt.title('Scatter plot of Students study hours vs Marks')
plt.show()
#There is positive corelation between X(Study hours) and y(Marks) so this is an example of Linear Regression Model

x=df.iloc[:,:1].values
y=df.iloc[:,1].values

#Splitting data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)

#Model building
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

#Train the data
regressor.fit(x_train,y_train)

#regressor model will leran fro data and will generate m and c
#Equation- y=m*x+c
c=regressor.intercept_
m=regressor.coef_
print("intercept/c:",c)
print("coefficient/m/slop of line:",m)
print(f"Equation of line is: {m} * x + {c}")

#Now if we know X i.e hours of study we can fine marks of that students
#ex-If hours of study is 6 hours- x=6
print("Marks:",m*6+c)

#Evaluate the model
#Predict x_test and then compair it with y_test
outcome=regressor.predict(x_test)
out_df=pd.DataFrame({'Actual':y_test,'Predicted':outcome})
print("Actual vs Predicted:\n",out_df)

#Model Evaluation
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,outcome)
rmse=mse**0.5
mae=metrics.mean_absolute_error(y_test,outcome)
print("Mean Squared Error=",mse)
print("Root Mean Squared Error=",rmse)
print("Mean squared error=",mse)
print("Mean Absolute error=",mae)

#Calculating R-Squared value
# This value can be maximum 1(Good fit) and minimum 0 (Bad fit) [R-squared=1-(MSE REGRESSION-MSE AVERAGE)]-Formula
R_Squared=metrics.r2_score(y_test,outcome)
print("R-squared value=",R_Squared)

























