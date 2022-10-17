'''

IMPORTING PACKAGES
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv("student_scores.csv")
print(data)

'''
PLOTTING THE RAW DATA
'''
plt.title("Raw data- Hours studied vs Marks scored")
plt.xlabel('Number of study hours')
plt.ylabel('Score achieved')
plt.scatter(data.Hours,data.Scores,color='green',label='Data Distribution')
plt.legend(['Data Distribution'])
plt.show()

'''
TRAINING THE DATASET
'''
LR= linear_model.LinearRegression()
LR.fit(data[['Hours']],data.Scores)


m=LR.coef_
print(m,'COEFFICIENT')

b=LR.intercept_
print(b,'INTERCEPT OF LINE')



Predicted_Score= data[['Hours']]* m + b

plt.title("Predictions of Hours studied  vs Marks scored using Linear Regression")
plt.xlabel('Number of study hours')
plt.ylabel('Score achieved')
plt.scatter(data.Hours,data.Scores,color='green',label='Data Distribution')
plt.plot(data.Hours,Predicted_Score,color='red',label='Linear Regression Line')
plt.legend(['Linear Regression Line','Data Distribution'])
plt.show()


'''
TESTING THE DATASET ON THE GIVEN PROBLEM
'''

def LR_Prediction(hour):
    score=m*hour+b
    return score

'''
PROBLEM STATE : WHAT WILL BE THE SCORE OF STUDENT IF HE/SHE STUDIES FOR 9.25 HOURS
'''

Prediction=LR_Prediction(9.25)
print("The problem statement states that the student studies 9.25 hour/day,the prediction of Score is %.2f"%Prediction)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error: ",mean_absolute_error(data.Scores,Predicted_Score))