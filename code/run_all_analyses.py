#!/usr/bin/env python
# coding: utf-8

import stanza
import string
import os
import os.path
import json
import pandas as pd
import glob
import random
import nltk
import numpy as np
from itertools import combinations
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors, Word2Vec
import gensim.downloader as api
from wordcloud import WordCloud
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords

### helper functions ###

# convert the DE and NL files to dataframe
def load_json_files(path, lang):
    _path = path + lang
    files = os.listdir(_path)
    data = list()
    for filename in files:
        filepath = os.path.join(_path, filename)
        with open(filepath) as file:
            data.append(json.load(file))

    return pd.DataFrame(data)


# MT does not posses any lemmatization and therefor cannot be preprocessed
def preprocess_de(article):
    stopwords = nltk.corpus.stopwords.words('german')
    processed_article = data['de']['nlp'].process(article)
    all_lemmas = []
    for s in processed_article.sentences: 
        clean_lemmas = list()
        for word in s.words:
            if word.text.lower() not in stopwords:
                lemma = word.lemma.lower()
                if lemma not in string.punctuation:
                    clean_lemmas.append(lemma)
        all_lemmas.extend(clean_lemmas)
    return all_lemmas

def preprocess_nl(article):
    stopwords = nltk.corpus.stopwords.words('dutch')
    processed_article = data['nl'].process(article)
    all_lemmas = []
    for s in processed_article.sentences: 
        clean_lemmas = list()
        for word in s.words:
            if word.text.lower() not in stopwords:
                lemma = word.lemma.lower()
                if lemma not in string.punctuation:
                    clean_lemmas.append(lemma)
        all_lemmas.extend(clean_lemmas)
    return all_lemmas


def train_custom_model(dataset):
    articles = dataset['df']['text']
    tokenizer = dataset['nlp']
    tokenized = []
    for article in articles:
        for sent in tokenizer(article).sentences:
            tokenized.append([tok.text.lower() for tok in sent.tokens])

    # Train a Word2Vec model, the min_count parameter indicates the minimum frequency of each word in the corpus
    mymodel = Word2Vec(tokenized, min_count=2)

    # summarise vocabulary
    words = list(mymodel.wv.vocab)

    return mymodel


def visualize_word_vectors(dataset, lang, model, vocab):
    # Apply dimensionality reduction with PCA or T-SNE
    high_dimensional = model[vocab]
    reduction_technique = TSNE(n_components=2)

    two_dimensional = reduction_technique.fit_transform(high_dimensional)

    # Get the indices in the vocabulary for selected terms
    terms = dataset['wv_terms']
    vocab_list = list(vocab.keys())
    term_indices = [vocab_list.index(term) for term in terms]

    # Plot the two-dimensional vectors for the selected terms
    x_values = [two_dimensional[index, 0] for index in term_indices]
    y_values = [two_dimensional[index, 1] for index in term_indices]

    fig, ax = plt.subplots(1, 1, figsize = (15, 10))

    colors = cm.rainbow(np.linspace(0, 1, len(terms)))
    for x, y, c in zip(x_values, y_values, colors):
        ax.plot(x, y, 'o', markersize=12, color=c)

    # Add title and description
    ax.set_title(lang+' terms')

    # Annotate the terms in the plot
    for i, word in enumerate(terms):
        plt.annotate(word, xy=(x_values[i], y_values[i]), fontsize = 16)

    # legend
    plt.legend(dataset['wv_terms_translations'])
    plt.show()


def get_top_tfidf_features(row, terms, top_n=25):
    top_ids = np.argsort(row)[::-1][:top_n]
    top_features = [terms[i] for i in top_ids]
    return top_features


def wordcloud_cluster_byIds(clusterId, clusters, keywords, lang):
    words = []
    for i in range(0, len(clusters)):
        if clusters[i] == clusterId:
            for word in keywords[i]:
                words.append(word)
    # Generate a word cloud based on the frequency of the terms in the cluster
    wordcloud = WordCloud(max_font_size=40, relative_scaling=.8).generate(' '.join(words))
   
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(lang+str(clusterId)+".png")


def cluster_wordcloud(dataset, lang):
    news_content = dataset['df']

    # We filter out empty articles
    news_content = news_content[news_content["text"].str.len() >0 ]
    articles = news_content["text"]

    # You can play around with the ngram range
    if k == 'de':
        vectorizer = TfidfVectorizer(use_idf=True, tokenizer=preprocess_de)
    elif k == 'nl':
        vectorizer = TfidfVectorizer(use_idf=True, tokenizer=preprocess_nl)
    else:
        vectorizer = TfidfVectorizer(use_idf=True)
        
    tf_idf = vectorizer.fit_transform(articles)
    all_terms = vectorizer.get_feature_names()

    # extract the keywords
    num_keywords = 10


    keywords = []
    for i in range(0, tf_idf.shape[0]):
        row = np.squeeze(tf_idf[i].toarray())
        top_terms_for_article= get_top_tfidf_features(row, all_terms, top_n=num_keywords)
        keywords.append(top_terms_for_article)


    # document representations
    all_doc_representations = []

    
    for doc_keywords in keywords:
        doc_representation =[]
        for keyword in doc_keywords:
            keyword_with_capital = keyword[0].upper() + keyword[1:]
            if keyword in dataset['fasttext_model'].vocab:            
                word_representation = dataset['fasttext_model'].get_vector(keyword)
                doc_representation.append(word_representation)
            elif keyword_with_capital in dataset['fasttext_model'].vocab:
                word_representation = dataset['fasttext_model'].get_vector(keyword_with_capital)
                doc_representation.append(word_representation)
            
        # Take the mean over the keywords
        mean_keywords = np.mean(doc_representation, axis=0)
        all_doc_representations.append(mean_keywords)

    # Number of clusters
    num_clusters = 4
    km = KMeans(n_clusters=num_clusters)
    km.fit(all_doc_representations)

    # Output the clusters
    clusters = km.labels_.tolist()
    clustered_articles ={'link': news_content["link"],'website': news_content["website"],'text': news_content["text"], 'cluster': clusters}
    overview = pd.DataFrame(clustered_articles, columns = ['link', 'website', 'text', 'Cluster'])


    # display wordcloud for cluster 3
    for i in range(4):
        wordcloud_cluster_byIds(i, clusters, keywords, lang)


path = input("provide path to data: ")
path_model = input("provide path to language models: ")

# load data
de_df = load_json_files(path, 'de')
nl_df = load_json_files(path, 'nl')
mt_df = pd.read_csv(path+'mt/articles_mt.tsv', sep='\t', header=0)

data = {
    'de': {
        'df': de_df,
        'wv_terms': ["gesetz", "vergewaltigung", "schwangerschaft", "frauen", "abtreibung", "abtreibungen", "leben", "polen", "mehr", "schwangerschaftsabbruch", "sagt", "sei", "kirche"],
        'wv_terms_translations': ["gesetz (law)", "vergewaltigung (rape)", "schwangerschaft (pregnancy)", "frauen (women)", "abtreibung (abortion)", "abtreibungen (abortions)", "leben (life)", "polen (poland)", "mehr (more)", "schwangerschaftsabbruch (abortion)", "sagt (says)", "sei (would be)", "kirche (church)"]
        },
    'nl': {
        'df': nl_df,
        'wv_terms': ["wet", "zwangerschap", "abortus", "vrouwen", "we", "jaar", "nederland", "mensen", "vrouw", "wel", "waar", "weken"],
        'wv_terms_translations': ["wet (law)", "zwangerschap (pregnancy)", "abortus (abortion)", "vrouwen (women)", "we (we)", "jaar (year)", "Nederland (The Netherlands)", "mensen (people)", "vrouw (woman)", "wel (well)", "waar (true / where)", "weken (weeks)"]
        }, 
    'mt': {
        'df': mt_df,
        'wv_terms': ["abort", "stupru", "liġi", "dritt", "tqala", "mewt", "gvern", "nisa"],
        'wv_terms_translations': ["abort (abortion)", "stupru (rape)", "liġi (law)", "dritt (right)", "tqala (pregnancy)", "mewt (death)", "gvern (government)", "nisa (women)"]
        }
    }

# load models
print("loading german model ...", end='')
data['de']['fasttext_model'] = KeyedVectors.load_word2vec_format(path_model+"german.model", binary=True)
print("done")
print("loading dutch model ...", end='')
data['nl']['fasttext_model'] = KeyedVectors.load_word2vec_format(path_model+"model.bin", binary=True)
print("done")
print("loading maltese model ...", end='')
data['mt']['fasttext_model'] = KeyedVectors.load_word2vec_format(path_model+"cc.mt.300.vec")
print("done")

nltk.download('stopwords')


for k, v in data.items():
    print('\n\n### Analysis for '+ k.upper() + ' language ###\n\n')

    # Prepare the nlp pipeline for all languages
    stanza.download(k)
    v['nlp'] = stanza.Pipeline(k, processors='tokenize,pos,lemma')

    # fill missing data with an empty string
    print('filling any missing values (NaNs)...')
    v['df'] = v['df'].fillna('')
    
    # describe (basic statistics)
    print('Describing dataframe...')
    print(v['df'].describe())
    print()

    
    # train custom model and visualize word vectors with that model
    print('training custom model based on data...')
    #model = train_custom_model(v)
    print('creating word vectors based on custom model...')
    #visualize_word_vectors(v, k, model, model.wv.vocab)

    # display a cluster wordcloud for each language
    print('Creating wordcloud based on fasttext model...')
    cluster_wordcloud(v, k)
