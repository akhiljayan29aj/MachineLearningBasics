## Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import *

#### NUMPY

arr = np.array([1,2,3,4])   # creating a 1D array
arr
arr.ndim  # tells the dimension of the array

arr1 = np.array([[1,2,3],[4,5,6],[7,7,8]])   # creating a 2D array 
arr1
arr1.ndim

data = np.array([[1,2,3,4,5,6],[6,5,4,3,2,1]])

plt.plot(data[0],data[1])  # plotting line graph
plt.show()      # shows the plotted graph

arr2 = np.array([45])   # creating a 1D array
arr2.ndim

arr3 = np.array(45)    # array with 0 dimension
arr3.ndim

arr4 = np.array([1.1,2.2,3.3,4.4])

newarr = arr4.astype(str)    # changing the type of the elements of the array into str
newarr
newarr.dtype  # tells us the type, size, byte order

newarr1 = arr4.astype(int)  # changing the type of the elements of the array into int
newarr1.dtype

# Shape and reshaping of narray

arr4.shape  # tells the shape of the narray

data.shape

data.flatten()  # reshape in order of (1,n), where n = number of elements

data.reshape(6,2)  # reshapes in the given order

data.ravel()    #ravel() and flatten() are basically the same


# ARROW PLOT AND STEM PLOT

x=[1,2,3]
y=[4,5,6]

plt.Arrow(5,5,1,1)  #(x, y, dx, dy)  # using arrow plot
plt.show()

plt.stem(x,y)   # using stem plot
plt.show()

plt.bar(x,y)  # using bar plot
plt.show()

y=[20,30,40,50,80,78]

plt.pie(y)  # using pie plot
plt.show()

## Program 1:

a = np.arange(-pi,pi,0.1)
b = []
for i in a:
    b.append(sin(i))
plt.subplot(221)
plt.xticks([-pi,0,pi])
plt.yticks([-1,0,1])
plt.grid()
plt.legend('sine')
plt.title('Graph of Sine')
plt.plot(a,b)
plt.show()

## Program 2:

students = ['A','B','C','D','E','F']
marks = [40,99,45,78,66,90]
plt.bar(students, marks)
plt.show()

#### PANDAS

# Using pandas to read a local CSV

data = pd.read_csv('./iris.csv')
data
