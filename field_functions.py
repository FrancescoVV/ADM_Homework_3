from bs4 import BeautifulSoup
import requests
import time
import os
import re
import csv
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

#We build this function in order to print like strings the items in the lists
def print_list(l):
    lenght=len(l)
    if lenght==0:
        return ""
    if lenght==1:
        return l[0]
    to_print=""
    if lenght>1:
        for i in range(lenght-1):
            to_print+=l[i]+", "
        to_print+=l[lenght-1]
    return to_print


#In order to compute the score for this field we define a similarity starting on the jaccard distance
class bookTitle():
    def name():
        return "bookTitle"
    def parse(soup):
        bookTitle = soup.find_all('h1')[0].contents[0]
        bookTitle = " ".join(bookTitle.split())
        return (bookTitle)
    def score(book_info, query):
        w1 = set(book_info)
        w2 = set(query)
        return 1 - nltk.jaccard_distance(w1, w2)

#In order to compute the score for this field we define a similarity starting on the jaccard distance
class bookSeries():
    def name():
        return "bookSeries"

    def parse(soup):
        bookSeries = ""
        bookSeries = soup.find('h2', id="bookSeries").text.strip()[1:-1]
        return bookSeries

    def score(book_info, query):
        w1 = set(book_info)
        w2 = set(query)

        return 1 - nltk.jaccard_distance(w1, w2)

#In order to compute the score for this field we define a similarity starting on the jaccard distance
class bookAuthors():
    def name():
        return "bookAuthors"

    def parse(soup):
        bookAuthors = []
        for element in soup.find_all("span", itemprop="name"):
            bookAuthors.append(element.text.strip())
        return print_list(bookAuthors)

    def score(book_info, query):
        w1 = book_info.lower()
        w1 = w1.split(";")

        w2 = query.lower()
        w2 = w2.split(",")
        score = []

        for word_query in w1:
            temp_score = 0
            for element in w2:
                temp_score = max(1 - nltk.jaccard_distance(set(word_query.strip()), set(element.strip())), temp_score)
            score.append(temp_score)

        return sum(score) / len(score)

#In order to compute the score for this field we chose a function that goes from 0 to 1 in a faster way (like the tanh()) and we set the parameters according to our purpose.
#The function custom_tanh()  gives a score really close to 1 if the query matches or goes beyond the parameter, otherwise it gives less point if it's lower.
#Eg: if the query is of 2 points smaller that the real rating Value, the score is 0.5
class ratingValue():
    def name():
        return "ratingValue"

    def parse(soup):
        ratingValue = soup.find_all('span', itemprop="ratingValue")[0].contents[0].split('\n')[1].strip()
        return ratingValue

    def score(book_info, query):
        def custom_tanh(x, parameter):
            value = 3 / 2 * (x - parameter + 2)
            return ((np.tanh(value)) + 1) / 2

        try:
            query = float(query)
        except ValueError:
            print("Warning; failed conversion of", query, "to float")
            return 0

        if query < 0 or query > 5:
            print("Warning: value ouside of the range 0-5")

        return custom_tanh(float(book_info), query)

#In order to compute the score for this field we chose a function that takes the minimun betweem 1 amd the ratio of the real value of the rating count and the one provided in the query.
class ratingCount():
    def name():
        return "ratingCount"

    def parse(soup):
        return str(soup.find("meta", itemprop="ratingCount").get("content"))

    def score(book_info, query):
        try:
            book_info = int(book_info)
            query = int(query)
        except ValueError:
            return 0

        return min(1, book_info / query)

#In order to compute the score for this field we chose a function that takes the minimun betweem 1 amd the ratio of the real value of the review count and the one provided in the query.
class reviewCount():
    def name():
        return "reviewCount"

    def parse(soup):
        return str(soup.find("meta", itemprop="reviewCount").get("content"))

    def score(book_info, query):
        try:
            book_info = int(book_info)
            query = int(query)
        except ValueError:
            return 0

        return min(1, book_info / query)

#We compute the score for this field in the poin 2.2 using the TfIdf and the cosine similarity.
class Plot():
    def name():
        return "Plot"

    def parse(soup):

#With this function we remove the first string in italic if it contains one of this 'forbidden strings'
        def headingToRemove(Plot):
            to_check = Plot.find("i")
            if to_check:
                forbidden_strings = ["isbn", "edition", "librarian's note"]
                for string in forbidden_strings:
                    if string in to_check.text.lower():
                        Plot.find("i").decompose()

        Plot = soup.find("div", id="descriptionContainer").find_all("span")
        if len(Plot) == 2:
            Plot = Plot[1]
            headingToRemove(Plot)
            Plot = Plot.text
            Plot = " ".join(Plot.split())
            Plot = Plot.replace("\\", "")
        elif len(Plot) == 1:
            Plot = Plot[0]
            headingToRemove(Plot)
            Plot = Plot.text
            Plot = " ".join(Plot.split())
            Plot = Plot.replace("\\", "")
        else:
            Plot = ""
        return Plot

    def score(book_info, query):
        print("This should not be call. We compute this with the TfIdf score")
        return 0

#In order to compute the score for this field we chose a Gaussian function with specific parameters.
class NumberOfPages():
    def name():
        return "NumberOfPages"

    def parse(soup):
        N_pages = soup.find_all('span', itemprop="numberOfPages")
        if N_pages:
            return N_pages[0].contents[0].replace('\n', '').strip().split()[0]
        return ""

    def score(book_info, query):

        try:
            n_pages = int(book_info)
        except (ValueError, TypeError) as e:
            return 0

        try:
            query = int(query)

        except (ValueError, TypeError) as e:
            print("This should not be printed")
            return 0

        exponent = -(1 / 60 * (n_pages - query)) ** 2
        return np.exp(exponent)

#In order to compute the score for this field we chose a Gaussian function with an addition of some paratemers to make it so that it gives a good score for a wider range of dates for older books.
# The parameter 2030 is hard coded, we suggest to change it according to the sysdate if a person would like to use this program for years.
class Publishing_Date():
    def name():
        return "Publishing_Date"

    def parse(soup):
        elements = [e for e in soup.find_all("div", class_="row") if re.match(r'Published', e.text.strip())]
        # We first try to get the "first published date"
        if elements:
            date = re.findall(r'(?<=\(first published )(.*?)(?=\))', elements[0].text)
        else:
            return ""
        if date:
            return date[0]
        # We now see if there is a publishing date (but not a first publishing one).
        date = " ".join(elements[0].text.split()).split()
        # Handling the issue that not always the date is in the same format
        if date[1] != "by":
            Publishing_Date = date[1]
            if len(date) > 2 and date[2] != "by":
                Publishing_Date += " " + date[2]
                if len(date) > 3 and date[3] != "by":
                    Publishing_Date += " " + date[3]
            return Publishing_Date
        else:
            return ""

    def score(book_info, query):
        if book_info:
            get_date = book_info.split(" ")[-1]
        try:
            get_date = int(get_date)
        except (ValueError, TypeError, UnboundLocalError) as e:
            return 0

        try:
            query = int(query)

        except (ValueError, TypeError) as e:

            print("This should not be printed")
            return 0

        exponent = -((2 / (2030 - query) ** 0.8) * (get_date - query)) ** 2
        return np.exp(exponent)

#In order to compute the score for this field we define a similarity starting on the jaccard distance
class Characters():
    def name():
        return "Characters"

    def parse(soup):
        Characters = soup.find_all("a", {'href': re.compile(r'^/characters/')})
        characters = []
        for item in Characters:
            characters.append(" ".join(item.text.split()))
        return print_list(characters)

    def score(book_info, query):
        w1 = book_info.lower()
        w1 = w1.split(";")

        w2 = query.lower()
        w2 = w2.split(",")
        score = []

        for word_query in w1:
            temp_score = 0
            for element in w2:
                temp_score = max(1 - nltk.jaccard_distance(set(word_query.strip()), set(element.strip())), temp_score)
            score.append(temp_score)

        return sum(score) / len(score)

#In order to compute the score for this field we define a similarity starting on the jaccard distance
class Setting():
    def name():
        return "Setting"

    def parse(soup):
        Setting_temp = soup.find_all("div", class_="infoBoxRowItem")
        Setting = []
        temp = []
        Setting_places = []
        for element in Setting_temp:
            if element.find("a", {'href': re.compile(r'^/places/')}):
                Setting_places = element
        if Setting_places:
            temp = Setting_places.find_all()
        else:
            Setting = []
        for element in temp:
            if element.name == "a":
                to_insert = element.text.split()
                Setting.append(" ".join(to_insert))
            if element.name == "span":
                to_add = element.text.split()
                Setting[-1] += " " + (" ".join(to_add))
        # The reason wy we use this for loop is because we decide to take the text in parentesis (e.g London(England)) that is harder to pick from the pages,
        # so we find this solution that takes also the words '…more' and '…less' so we delete them in this second step.
        for i in range(len(Setting)):
            Setting[i] = Setting[i].replace("…more", "").replace("…less", "").strip()
        Setting = list(dict.fromkeys([x for x in Setting if x]))
        return print_list(Setting)

    def score(book_info, query):
        w1 = book_info.lower()
        w1 = w1.split(";")

        w2 = query.lower()
        w2 = w2.split(",")
        score = []

        for word_query in w1:
            temp_score = 0
            for element in w2:
                temp_score = max(1 - nltk.jaccard_distance(set(word_query.strip()), set(element.strip())), temp_score)
            score.append(temp_score)

        return sum(score) / len(score)

#For this field we don't define a score
class Url():
    def name():
        return "Url"

    def parse(soup):
        return re.findall(r'(?<=link href=")(.*?)(?=")', str(soup))[0]

    def score(book_info, query):
        print("Warning: a score for Url is not implemented. Returning default value of 1")
        return 0

#to check if the plots are written in a properly way
def custom_detect(string):
    try:
        return detect(string)
    except LangDetectException:
        return "BADLANGUAGE"