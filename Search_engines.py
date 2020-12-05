from field_functions import *
import re
import os
import csv
import nltk
import pickle
import numpy as np
import heapq
from scipy.spatial import distance
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from heapq import heappush
N_doc=26543

''' Data preprocessing '''
''' To create the .pkl files, see the "DataCollection&DataStructure(Point1)" written in the second part. '''
# - keys: index of each word from 0 to 55037
# - values: each book containing the unique word (index number)
with open('inverted_index_1.pkl', 'rb') as handle:
    inverted_index = pickle.load(handle) 

# - keys: all words in the all documents
# - values: index of each word from 0 to 55037
with open('vocabulary.pkl', 'rb') as handle:
    vocabulary = pickle.load(handle)

# - keys: index of each word from 0 to 55037
# - values: number of times a word appear in all books
with open('vocabulary2.pkl', 'rb') as handle:
    vocabulary2 = pickle.load(handle)

# - keys: index of each word from 0 to 55037
# - values: (book, TfIdf) for all the keys 
with open('tfIdf_index.pkl', 'rb') as handle:
    tfIdf_index = pickle.load(handle)

# - keys: book
# - values: (words, TfIdf) for all the words in the plot of the book
with open('BookTokens.pkl', 'rb') as handle:
    BookTokens = pickle.load(handle)

"""SEARCH ENGINE POINT 2.1"""
def SearchEngine2_1():
    tokenizer = RegexpTokenizer(r'[a-z]+') 
    stop_words = set(stopwords.words("english"))
    stemmer= PorterStemmer()

    query = input()
    # Apply the same preprocessing (using tokenizer and stemmer) also on the query string
    query = tokenizer.tokenize(query.lower())
    query_stems = [stemmer.stem(word) for word in query if word not in stop_words]
    query_stem_test = query_stems
    query_stems = []

    # Checking if input stems exists in the vocabulary
    for word in query_stem_test:
        try:
            vocabulary[word]
            query_stems.append(word)
        except KeyError:
            print("Stem", word, "not found. It will be ignored.")

    temp = set()
    if len(query_stems) > 0:
        temp = inverted_index[vocabulary[query_stems[0]]]
        for stem in query_stems:
            temp = temp.intersection(inverted_index[vocabulary[stem]])

    matching_books = list(sorted(temp))

    for i in matching_books:
        with open('articles/article_' + str(i) + '.tsv', 'r', encoding="utf-8") as file:
            temp = csv.DictReader(file, delimiter='\t')
            for row in temp:
                print("BookTitle:", row["bookTitle"])
                print("Plot:")
                print(row["Plot"])
                print("Url:", row["Url"])
                print()

"""SEARCH ENGINE POINT 2.2"""
def SearchEngine2_2():
    tokenizer = RegexpTokenizer(r'[a-z]+') 
    stop_words = set(stopwords.words("english"))
    stemmer= PorterStemmer()

    query = input()
    # Apply the same preprocessing (using tokenizer and stemmer) also on the query string
    query = tokenizer.tokenize(query.lower())
    query_stems = [stemmer.stem(word) for word in query if word not in stop_words]
    query_stem_test = query_stems
    query_stems = []

    # Checking if input stems exists in the vocabulary
    for word in query_stem_test:
        try:
            vocabulary[word]
            query_stems.append(word)
        except KeyError:
            print("Stem", word, "not found. It will be ignored.")

    query_stems = list(dict.fromkeys([x for x in query_stems]))  # Removing possible similarities

    temp = set()
    if len(query_stems) > 0:
        temp = inverted_index[vocabulary[query_stems[0]]]
        for stem in query_stems:
            temp = temp.intersection(inverted_index[vocabulary[stem]])

    matching_books = list(sorted(temp))

    # Calculating tfIdf for the query
    query_tfIdf = []
    for word in query_stems:
        query_tfIdf.append((vocabulary[word], np.log(N_doc / vocabulary2[vocabulary[word]])))
    query_tfIdf.sort()
    query_tfIdf = dict((x, y) for x, y in query_tfIdf)

    # Initializing the heap structure
    BooksWithScore = []

    for book in matching_books:
        doc_vector = []
        query_vector = []
        for word_id in BookTokens[book]:
            doc_vector.append(word_id[1])
            if word_id[0] in query_tfIdf:
                query_vector.append(1)
            else:
                query_vector.append(0)

        doc_vector = np.array(doc_vector)
        query_vector = np.array(query_vector)
        cos_similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))

        # Insert in the heap structure the result
        heappush(BooksWithScore, (cos_similarity, book))
    # take the top k elements from the list, our k is 10
    k = 10
    top_k_books = heapq.nlargest(k, BooksWithScore)

    # Show the results for the first 10 elements, sorted in ascending order
    for book in top_k_books:
        i = book[1]
        with open('articles/article_' + str(i) + '.tsv', 'r', encoding="utf-8") as file:
            temp = csv.DictReader(file, delimiter='\t')
            for row in temp:
                print("BookTitle:", row["bookTitle"])
                print("Plot:")
                print(row["Plot"])
                print("Url:", row["Url"])
                print("Score:", book[0])
                print()

"""SEARCH ENGINE POINT 3"""


def TfIdfScore_plot(query):
    tokenizer = RegexpTokenizer(r'[a-z]+') 
    stop_words = set(stopwords.words("english"))
    stemmer= PorterStemmer()

    query = tokenizer.tokenize(query.lower())
    query_stems = [stemmer.stem(word) for word in query if word not in stop_words]

    query_stem_test = query_stems
    query_stems = []

    # Checking if input stems exists in the vocabulary

    for word in query_stem_test:
        try:
            vocabulary[word]
            query_stems.append(word)
        except KeyError:
            print("Stem", word, "not found. It will be ignored.")

    query_stems = list(dict.fromkeys([x for x in query_stems]))  # Removing possible similarities

    ##########################
    temp = set()

    if len(query_stems) > 0:
        temp = inverted_index[vocabulary[query_stems[0]]]
        for stem in query_stems:
            temp = temp.intersection(inverted_index[vocabulary[stem]])

    matching_books = list(sorted(temp))

    # Calculating tfIdf for the query.
    query_tfIdf = []

    for word in query_stems:
        query_tfIdf.append((vocabulary[word], np.log(N_doc / vocabulary2[vocabulary[word]])))
    query_tfIdf.sort()

    query_tfIdf = dict((x, y) for x, y in query_tfIdf)

    BooksWithScore = []

    for book in matching_books:
        doc_vector = []
        query_vector = []
        for word_id in BookTokens[book]:
            doc_vector.append(word_id[1])
            if word_id[0] in query_tfIdf:
                query_vector.append(1)
            else:
                query_vector.append(0)

        doc_vector = np.array(doc_vector)
        query_vector = np.array(query_vector)
        cos_similarity = 1 - distance.cosine(doc_vector, query_vector)

        heappush(BooksWithScore, (book, cos_similarity))

    BooksWithScore.sort(key=lambda x: -x[1])
    return BooksWithScore


def SearchEngine3(fields_list):
    print("Write the plot keywords")
    plot_input =input()
    print()
    print("Write other parameters, specifing the field separated by a ','. Example: numberofpages 235, title hunger")
    text_input =input()
    text_input =text_input.split(",")
    field_names =[x.name().lower() for x in fields_list]
    query_dictionary ={}
    for input_field in text_input:
        input_field =input_field.split()
        if input_field and input_field[0].lower() in field_names:
            if input_field[0] in query_dictionary:
                print("Warning: field" ,input_field[0] ,"inserted more than once. Only the first value will be used")
                continue

            if len(input_field ) >1:
                query_dictionary[input_field[0].lower()] =" ".join(input_field[1:len(input_field)])
            else:
                print("Warning: the field" ,input_field[0] ,"has no specified value")
        else:
            if input_field:
                print("Warning: the field" ,'" ' +input_field[0] +'"', "does not exist!")
            else:
                print("Warning: empty field name entered")

    print(query_dictionary)

    to_call =[]
    for element in query_dictionary:
        to_call.append(field_names.index(element))

    Book_with_plot_score =(TfIdfScore_plot(plot_input))
    Book_with_plot_normalized =[]

    # We decided to normalize the plot TfIdf score so that the best match has 1 as a score to give more importance to it.
    if Book_with_plot_score:
        max_value =Book_with_plot_score[0][1]
        for book in Book_with_plot_score:
            Book_with_plot_normalized.append((book[0] ,book[1 ] /max_value))

    Book_with_plot_score =Book_with_plot_normalized


    Book_with_full_score =[]  # initializing the heap structure

    for element in Book_with_plot_score:
        book =element[0]
        plot_score =element[1]
        temp_score =0
        with open('articles/article_' + str(book) +'.tsv', 'r', encoding="utf-8") as file:
            temp = csv.DictReader(file, delimiter='\t')
            for row in temp:
                for field in query_dictionary:
                    field_name = fields_list[field_names.index(field)].name()
                    temp_score += fields_list[field_names.index(field)].score(row[field_name], query_dictionary[field])
        score = temp_score + plot_score

        heappush(Book_with_full_score, (score, book))

    # take the top k elements from the list, our k is 10
    k = 10
    top_k_books = heapq.nlargest(k, Book_with_full_score)

    for book in top_k_books:
        i = book[1]
        with open('articles/article_' + str(i) + '.tsv', 'r', encoding="utf-8") as file:
            temp = csv.DictReader(file, delimiter='\t')
            for row in temp:
                print("BookTitle:", row["bookTitle"])
                print("Plot:")
                print(row["Plot"])
                print("Url:", row["Url"])
                print("Score:", book[0])
                print()
