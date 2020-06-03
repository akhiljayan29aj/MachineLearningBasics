import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup


page=requests.get('https://mausam.imd.gov.in/',headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
    })
soup = BeautifulSoup(page.content,'html.parser')
html=list(soup.children)[3]
body=list(html.children)[3]
divs=body.findAll('div')
tem=divs[0].findAll(id="temperature")
hum=divs[0].findAll(id="temperature1")
weather=body.findAll(id="city_weather")
wind=weather[0].findAll('li')
t=weather[0].findAll('small')
air=wind[0].getText()
todayTemp=tem[0].getText()
todayHum=hum[0].getText()
time=t[0].getText()
print("Delhi's Weather")
print("Temperature",todayTemp)
print("Humidity", todayHum)
print("Wind Flow:",air)
print(time)


today=soup.findAll('div',class_='capital')
c=[]
tm=[]
a=[]
h=[]
n_tm=[]
for i in range(0,8):
    cities=today[i].findAll('h3')
    city = cities[0].getText()
    c.append(city)
    temp=today[i].findAll('p',class_="now")
    temps=temp[0].getText()
    tm.append(temps)
    air=today[i].findAll('p',class_="wind")
    airs=air[0].getText()
    a.append(airs)
    humo=today[i].findAll('span',class_="max")
    humos=humo[0].getText()
    h.append(humos)

for i in range(len(tm)):
    x = tm[i]
    x = x.replace('°','')
    x = float(x)
    n_tm.append(x)



plt.bar(c,n_tm)
plt.show()


