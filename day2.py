## Importing Libraries

import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt


#### PANDAS (cont.)

## Reading a remote csv

# pd.read_csv() returns a remote CSV file in form of a dictionary 
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv')

data.head(5)   # returns the first five dataset
# accessing the dictionary using the keys
x = data['sepal_length']   
y = data['petal_length']
type(x)
plt.plot(x,y)
plt.show()

## DataFrame (This is 2D)(they are similar to dictionaries and elements can be accessed by using keys and indexing)

df = pd.DataFrame([1,2,3])  #creating a DataFrame
df   # accessing the dataframe
df[0]    # accessing the set with key=0
dct = {'name':['akhil','arjun','atishay'],'marks':[80,90,100]}
d2d = pd.DataFrame(dct)   # converting a dictionary into dataframe
plt.bar(d2d['name'],d2d['marks'])
plt.show()

## Series (This is 1D)
s = pd.Series([1,2,3,4])  #creating a Series
s  # accessing the series
s = pd.Series([1,2,3,4],index=[100,101,102,103])  # changing the indices of elements

## Making 3D Dataset
data= {1:df,2:d2d}
data

## Saving a remote CSV file using url onto your system
data1 = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv')
data1.to_csv('iris.csv')

## Plotting using Pandas

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv'
dataset = pd.read_csv(url)

# Box plot
dataset.plot(kind = 'box', subplots = True, layout=(2,2), sharex = False, sharey = False)

# Histogram
dataset.hist()

# Scatter plot
pd.plotting.scatter_matrix(dataset)

plt.show()   #using a single .show() to display all the plots together


#### Web Scraping

# 1. Create a page variable and use HTTP get request 
page = requests.get('https://www.amazon.in/Lenovo-Tab-2GB-32GB-WiFi/dp/B083SMW5H3/ref=lp_21492881031_1_1?s=computers&ie=UTF8&qid=1591168339&sr=1-1')
page.status_code  # This tells the status of the request

## HTTP Statuses
## 200: "OK"
## 403: "FORBIDDEN"
## 404: "NOT FOUND"
## 503: "Service Unavailable"


# Using the header argument in requests method allows us to unblock the site which gives us 503 error
page1 = requests.get('https://www.amazon.in/dp/B0793DBS3X/ref=sspa_dk_detail_2?psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUExRjVRWUtGSlRMRllWJmVuY3J5cHRlZElkPUEwMjcyNjU4MkhLQ0I4Mzg4QUVIUyZlbmNyeXB0ZWRBZElkPUEwNDQwMjUzQUxMVTJZNUNMUUNGJndpZGdldE5hbWU9c3BfZGV0YWlsMiZhY3Rpb249Y2xpY2tSZWRpcmVjdCZkb05vdExvZ0NsaWNrPXRydWU=',headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
    })

# 2. Create a soup variable which contains the parsed source code
soup = BeautifulSoup(page1.content,'html.parser')  #parsing the source code 

print(soup.prettify())    # converting the parsed code into readable format

list(soup.children)     # makes a list of soup's children

# 3. Using hit and trail to access important tags and store them in variable and continue doing so to till we reach the desired tag

list(soup.children)[10]   # accessing the elements of the soup list to check the tag if it is required 
html = list(soup.children)[10]  # this was the <html> parent tag so store it in a variable
type(html)  # checking the type of tag. We can only proceed if it is a bs4.element.Tag

list(html.children)
list(html.children)[3]
body = list(html.children)[3]
type(body)

list(body.children)[5]
div = list(body.children)[5]
type(div)

list(div.children)
list(div.children)[1]
div1 = list(div.children)[1]

list(div1.children)
list(div1.children)[3]
div2 = list(div1.children)[3]

list(div2.children)
div3 = list(div2.children)[1]

list(div3.children)
list(div3.children)[5]

msg=list(div3.children)[5]
type(msg)
msg=msg.getText()  # using getText to extract the text from the tag (can only be done if it is an element tag)

## OR we can use this instead of Step 3

msg = body.findAll('p')   # using findAll to find p tags 
type(msg)  # the type is an element ResultSet so we can't use getText directly
finalMsg = msg[0].getText() # so we using indexing and select the first tag and then use getText
























































