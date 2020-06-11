import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


cols = ["Radiation","Temperature","TimeSunRise","TimeSunSet"]
data = pd.read_csv('./SolarPrediction.csv')

rad = data["Radiation"]
temp = data["Temperature"]
time = data["Time"]

x = data[cols]

plt.subplot(131)
plt.scatter(data.Time[:200],data.Radiation[:200])
plt.xlabel('Time')
plt.ylabel('Radiation')
plt.title('Time vs Radiation')

plt.subplot(132)
plt.scatter(data.Time[:200],data.Temperature[:200])
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Time vs Temperature')


data.TimeSunRise = pd.to_datetime(data.TimeSunRise)
data.TimeSunSet = pd.to_datetime(data.TimeSunSet)
diff=data.TimeSunSet-data.TimeSunRise
newdiff = diff.drop_duplicates()
redTime=[]
ot=list(newdiff.index)
for item in ot:
    w=data.Time[item]
    redTime.append(w)

redDate=[]
for item in ot:
    w=data.Data[item]
    redDate.append(w)

redTemp=[]
for item in ot:
    w=data.Temperature[item]
    redTemp.append(w)

redRad=[]
for item in ot:
    w=data.Radiation[item]
    redRad.append(w)

redPress=[]
for item in ot:
    w=data.Pressure[item]
    redPress.append(w)

redHum=[]
for item in ot:
    w=data.Humidity[item]
    redHum.append(w)

redWind=[]
for item in ot:
    w=data.WindDirection[item]
    redWind.append(w)

redSpeed=[]
for item in ot:
    w=data.Speed[item]
    redSpeed.append(w)
    
newdiff = newdiff.reset_index(drop=True)
SHI=list(newdiff.index)
T=[]
for i in range(75):
    T.append(int((pd.to_datetime(redTime[i]).value)/10000000000))
SH=[]
for i in range(75):
    SH.append(int((newdiff[i].value)/10000000000))
dct = {"day":SHI,"temp":redTemp,"rad":redRad,"speed":redSpeed,"wind":redWind,"pressure":redPress,"hum":redHum,"solar":SH}
df = pd.DataFrame(dct)
df.to_csv("solorhour.csv")





plt.subplot(133)
plt.plot(SHI,newdiff)
plt.xlabel('Time')
plt.ylabel('Solar Hours')
plt.title('Time vs Solar Hours')
##plt.show()

## Logistic Regressor

from sklearn.linear_model import LogisticRegression
new=pd.read_csv("./solorhour.csv")
array = new.values
X=array[:,1:2]

Y=array[:,8]

clf = LogisticRegression(random_state=0).fit(X, Y)
print(X)
print("predict")
print(clf.predict(X))
print(clf.predict_proba(X))
print("score")
print(clf.score(X, Y))


print("-----------------------------------------------------------------")

##'''DecisionTreeRegressor'''

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor


##### Fit regression model
x = X.astype(int)
y = Y.astype(int)

regr_1 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(x, y)

##### Predict



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
y_1 = regr_1.predict(x_test)
t=[]
for i in range(len(y_test)):
    t.append([y_test[i]])
##### Plot the results
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.figure()
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regr_1.predict(X_grid), color = 'blue')
plt.show()
print(regr_1.score(x, y))

print("-----------------------------------------------------------------")

####'''LinearRegression'''

from sklearn.linear_model import LinearRegression

X = new.iloc[:, 0].values.reshape(-1, 1)

Y = new.iloc[:, 8].values.reshape(-1, 1)

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)
plt.figure()
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()
print(Y_pred)
print(linear_regressor.score(X, Y))




