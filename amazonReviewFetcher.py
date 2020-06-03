from bs4 import BeautifulSoup
import requests

while True:
    url = input("Enter products URL: ")

    sources = requests.get(url,headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',})

    try:
        if sources.status_code == 200:
            sources = requests.get(url,headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',}).text

            soup = BeautifulSoup(sources, 'html5lib')

            rev = soup.findAll('div', id="averageCustomerReviews")

            review = rev[0].findAll('a',class_="a-popover-trigger a-declarative")[0].getText()

            avgCustReview = review.replace('\n','')

            prod = soup.findAll('span', class_="a-list-item a-color-tertiary")[-1].getText()

            prodName = prod.replace('\n','')

            print("Product Name:")
            print(prodName)
            print('Average customer review:')
            print(avgCustReview)

        else:
            print("HTTP Request Error")

    except:
        print("Enter a valid Amazon Product URL")

    finally:
        inp = input("Do you want a review of another product?(Y or N) ")
        if inp == "Y" or inp == "y":
            continue
        elif inp == "N" or inp == "n":
            break
        else:
            print("You pressed a key other than Y or N. So we will restart.")

    
