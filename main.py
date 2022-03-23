import csv
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *


def write_political_lean(csv_file):
    with open(f'/home/kirill/PycharmProjects/DataScienceProj/{csv_file}', encoding='utf-8') as r_file:
        file_reader = csv.reader(r_file, delimiter=",")

        for row in file_reader:
            if row[1] == "Liberal":
                f = open('Liberal.txt', 'a')
                f.write(row[0])
            else:
                f = open('Conservative.txt', 'a')
                f.write(row[0])


def tokenize_lean(lean):
    f = open(f"/home/kirill/PycharmProjects/DataScienceProj/{lean}.txt")
    text = f.read()
    punct = string.punctuation + '«»–—'
    en_stopwords = stopwords.words("english")
    dights = string.digits

    clear_text = text.lower()
    clear_text = "".join([ch for ch in clear_text if ch not in punct])
    clear_text = "".join([num for num in clear_text if num not in dights])
    clear_text_tokens = word_tokenize(clear_text)
    clear_text_tokens = [word for word in clear_text_tokens if word not in en_stopwords]
    return clear_text_tokens


def porter_and_lemmatizer(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = []
    normalized_tokens = []

    for word in tokens:
        stemmed_tokens.append(stemmer.stem(word))
    for word in stemmed_tokens:
        lemmas = WordNetLemmatizer()
        normalized_tokens.append(lemmas.lemmatize(word))
    return normalized_tokens


def tf(tokens, name):
    file = open(f"/home/kirill/PycharmProjects/DataScienceProj/frequency_of_{name}.txt", 'w')
    tokens_list = FreqDist(tokens)
    file.write(str([i for i in tokens_list.most_common()]))
    file.close()


def make_word_cloud(normalized_tokens):
    text = " ".join(normalized_tokens)
    wc = WordCloud(background_color="white", repeat=True)
    wc.generate(text)
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


write_political_lean('reddit_comments.csv')
liberal_tokens = tokenize_lean('Liberal')
conservative_tokens = tokenize_lean('Conservative')

lib_lemmas = porter_and_lemmatizer(liberal_tokens)
con_lemmas = porter_and_lemmatizer(conservative_tokens)

tf(lib_lemmas, 'liberal_terms')
tf(con_lemmas, 'conservative_terms')

make_word_cloud(lib_lemmas)
make_word_cloud(con_lemmas)
