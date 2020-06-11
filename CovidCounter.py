from pandas import read_csv
import numpy as np

df=read_csv("./datasets_494724_1224506_COVID19_line_list_data.csv",usecols=[2,7])

import matplotlib.pyplot as plt

count_male=0
count_female=0

arr1=[]
arr2=[]

for i in range(1,1085):
    if(df['reporting date'][i]==df['reporting date'][i-1]):
        if(df['gender'][i]=='male'):
            count_male+=1
        elif(df['gender'][i]=='female'):
            count_female+=1
    else:
        arr1.append([count_male+1])
        arr2.append([count_female+1])
        count_male=0
        count_female=0   

fig, ax = plt.subplots(figsize=(20, 10)) 


plt.yticks(np.arange(0, 25, 1))
plt.xticks(np.arange(0,300,10))
plt.plot(arr1,label='male')
plt.plot(arr2,label='female')
plt.ylabel("No. of people/day")
plt.xticks(rotation=90) 
plt.legend()
plt.show()

