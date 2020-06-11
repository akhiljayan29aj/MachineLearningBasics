## sudo pip3 install pdfplumber
## sudo pip3 install pdfminer.six

import pdfplumber
import pandas as pd

with pdfplumber.open(r'./source.pdf') as pdf:
	    first_page = pdf.pages[0]
	    x = first_page.extract_text()
	    data = x.split('\n')

print("**********",data[1],"**********")

for i in range(2,6):
    print(data[i])
print(data[7])

print("******************************************")

for i in range(8,13):
    print(data[i])

print("******************************************")

for i in range(13,18):
    print(data[i])

print("******************************************")

jan = data[34].split('  ')
feb = data[35].split('  ')
mar = data[36].split('  ')
apr = data[37].split('  ')
may = data[38].split('  ')
jun = data[40].split('  ')
jul = data[41].split('  ')
aug = data[42].split('  ')
sep = data[43].split('  ')
oc = data[44].split('  ')
nov = data[45].split('  ')
dec = data[46].split('  ')
year = data[47].split('  ')

months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec","year"]
GIhoriz = []
GIhoriz.append(jan[1])
GIhoriz.append(feb[1])
GIhoriz.append(mar[1])
GIhoriz.append(apr[1])
GIhoriz.append(may[1])
GIhoriz.append(jun[1])
GIhoriz.append(jul[1])
GIhoriz.append(aug[1])
GIhoriz.append(sep[1])
GIhoriz.append(oc[1])
GIhoriz.append(nov[1])
GIhoriz.append(dec[1])
GIhoriz.append(year[1])

horiz = []
horiz.append(jan[2])
horiz.append(feb[2])
horiz.append(mar[2])
horiz.append(apr[2])
horiz.append(may[2])
horiz.append(jun[2])
horiz.append(jul[2])
horiz.append(aug[2])
horiz.append(sep[2])
horiz.append(oc[2])
horiz.append(nov[2])
horiz.append(dec[2])
horiz.append(year[2])

coll = []
coll.append(jan[3])
coll.append(feb[3])
coll.append(mar[3])
coll.append(apr[3])
coll.append(may[3])
coll.append(jun[3])
coll.append(jul[3])
coll.append(aug[3])
coll.append(sep[3])
coll.append(oc[3])
coll.append(nov[3])
coll.append(dec[3])
coll.append(year[3])


sysopd = []
sysopd.append(jan[4])
sysopd.append(feb[4])
sysopd.append(mar[4])
sysopd.append(apr[4])
sysopd.append(may[4])
sysopd.append(jun[4])
sysopd.append(jul[4])
sysopd.append(aug[4])
sysopd.append(sep[4])
sysopd.append(oc[4])
sysopd.append(nov[4])
sysopd.append(dec[4])
sysopd.append(year[4])

syso = []
syso.append(jan[7])
syso.append(feb[7])
syso.append(mar[6])
syso.append(apr[6])
syso.append(may[6])
syso.append(jun[7])
syso.append(jul[7])
syso.append(aug[7])
syso.append(sep[7])
syso.append(oc[6])
syso.append(nov[7])
syso.append(dec[7])
syso.append(year[6])

dct = {"Month":months,"GIHoriz":GIhoriz,"CollPlane":coll,"SysOpd":sysopd,"SysOp":syso}
df = pd.DataFrame(dct)
df.to_csv("pvsystdata.csv")
print(df)
