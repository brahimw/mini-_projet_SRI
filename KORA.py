import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QTextBrowser, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import requests
from bs4 import BeautifulSoup
import re
import string as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
french_stopwords = set(stopwords.words('french'))

def search_function(query):
    url = 'https://www.lequipe.fr/Football/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = [link.get('href') for link in soup.find_all('a') if
             link.get('href') and link.get('href').startswith('/Football/Actualites')]
    for i in range(len(links)):
        if links[i].startswith('/'):
            links[i] = 'https://www.lequipe.fr/' + links[i][1:]

    docs = []
    for i in links:
        r = requests.get(i)
        soup = BeautifulSoup(r.text, 'html.parser')
        sen = []
        for j in soup.find('div', class_='article__body').find_all('p'):
            sen.append(j.text)
        docs.append(' '.join(sen))

    def remove_punctuations(text):
        return ("".join([ch for ch in text if ch not in st.punctuation]))

    def convert_to_lowercase(text):
        return ("".join([x.lower() for x in text]))

    def remove_small_words(text):
        words = text.split()
        filtered_words = [word for word in words if len(word) > 3]
        filtered_text = " ".join(filtered_words)
        return filtered_text

    def remove_stopwords(text):
        words = word_tokenize(text, language='french')
        filtered_text = " ".join([word for word in words if word.lower() not in french_stopwords])
        return filtered_text

    def lemmatization(text):
        stemmer = SnowballStemmer('french')
        tokens = word_tokenize(st, language='french')
        lemmas = [stemmer.stem(token) for token in tokens]
        return lemmas

    t = docs
    t = remove_punctuations(t)
    t = convert_to_lowercase(t)
    t = remove_small_words(t)
    t = remove_stopwords(t)

    d = [""] * len(docs)
    for i in range(len(docs)):
        d[i] = remove_stopwords(remove_small_words(convert_to_lowercase(remove_punctuations(docs[i]))))

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(d)
    df = pd.DataFrame(X.T.toarray(), index=vectorizer.get_feature_names_out())

    result_list = []

    def get_similar_articles(q, df):
        print("requête:", q)
        print("Résultats de recherche: ")
        q = [q]
        q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0], )
        sim = {}
        for i in range(10):
            sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
        sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
        for k, v in sim_sorted:
            if v != 0.0:
                result_list.append({
                    'similarite': v,
                    'article': docs[k]
                })
        return result_list

    result_list = get_similar_articles(query, df)

    return result_list

class SearchApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def set_logo_pixmap(self, image_filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, image_filename)

        pixmap = QPixmap(image_path)
        self.logo_label.setPixmap(pixmap)
    def init_ui(self):
        self.logo_label = QLabel(self)
        self.set_logo_pixmap("logo.png")
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setFixedSize(1000, 150)

        self.search_input = QLineEdit(self)
        self.search_button = QPushButton('Search', self)
        self.search_results = QTextBrowser(self)
        self.search_results.setFixedHeight(300)

        layout = QVBoxLayout()
        layout.addStretch(1)
        layout.addWidget(self.logo_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.search_input)
        layout.addWidget(self.search_button)
        layout.addWidget(self.search_results)
        layout.addStretch(1)

        self.setLayout(layout)

        self.search_button.clicked.connect(self.perform_search)

        self.setWindowTitle('Moteur de Recherche')
        self.setGeometry(100, 100, 600, 400)
    def perform_search(self):
        query = self.search_input.text()

        search_results = search_function(query)

        self.search_results.clear()
        self.search_results.append(f"Résultat"
                                   f"s de recherche pour '{query}':\n")
        for result in search_results:
            self.search_results.append(f"Similarité: {result['similarite']}\nArticle: {result['article']}\n")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    search_app = SearchApp()
    search_app.show()
    sys.exit(app.exec_())
