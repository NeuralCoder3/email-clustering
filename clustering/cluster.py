import re
import shutil
import string
from wordcloud import WordCloud
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from collections import Counter

# based on https://github.com/dpanagop/ML_and_AI_examples/blob/master/NLP_example_clustering.ipynb


def prepareWord(document, remove_whitespace=False):
    translator_1 = str.maketrans(
        string.punctuation, ' ' * len(string.punctuation))
    document = document.translate(translator_1)
    document = re.sub(r'\d+', ' ', document)
    document = re.sub(r"[^a-zA-Z0-9]+", ' ', document)

    if remove_whitespace:
        document = document.strip()
        document = document.replace(" ", "")
    return document


class LemmaTokenizer(object):
    def __init__(self, stopwords):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = stopwords

    def __call__(self, document):
        document = prepareWord(document)

        lemmas = []
        for token in word_tokenize(document):
            token = token.strip()
            token = self.lemmatizer.lemmatize(token)
            if token not in self.stopwords and len(token) > 2:
                lemmas.append(token)
        return lemmas


def cluster(k, in_dir, out_dir, additional_stopword_files, verbose=False, compute_variance=False):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    true_k = k
    stopword_lists = [
        "english_stopwords.txt",
        "custom_stopwords.txt",
        "german_stopwords.txt"
    ] + additional_stopword_files

    if verbose:
        print("Loading stopword lists...")
    stopwords = []
    for stopword_list in stopword_lists:
        with open(f"{stopword_list}", "r") as file:
            stopwords.extend(file.read().splitlines())

    filenames = []
    texts = []
    for f in os.listdir(in_dir):
        with open(f"{in_dir}/{f}", "r") as file:
            text = file.read()
            texts.append(text)
            filenames.append(f)

    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(stopwords))
    X = vectorizer.fit_transform(texts)

    if compute_variance:
        Sum_of_squared_distances = []
        K = range(2, true_k)
        for k in K:
            if verbose:
                print(f"Computing variance for k={k}")
            km = KMeans(n_clusters=k, max_iter=200, n_init=10)
            km = km.fit(X)
            Sum_of_squared_distances.append(km.inertia_)

        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.savefig(f"{out_dir}/cluster_variance.png")
        return

    model = KMeans(n_clusters=true_k, init='k-means++',
                   max_iter=200, n_init=10)
    model.fit(X)
    labels = model.labels_
    df = pd.DataFrame(list(zip(filenames, labels)),
                      columns=['title', 'cluster'])
    if verbose:
        print(df.sort_values(by=['cluster']))

    stopword_set = set(stopwords)

    result = {'cluster': labels, 'content': texts}
    result = pd.DataFrame(result)
    for k in range(0, true_k):
        s = result[result.cluster == k]
        text = s['content'].str.cat(sep=' ')
        text = text.lower()
        text = ' '.join([word for word in text.split()])
        words = text.split()
        words = [prepareWord(word, True) for word in words]
        words = [word for word in words if word not in stopword_set]
        text = ' '.join(words)
        wordcloud = WordCloud(max_font_size=50, max_words=100,
                              background_color="white").generate(text)
        if verbose:
            print('Cluster: {}'.format(k))
        titles = df[df.cluster == k]['title']
        h = Counter(words)
        h = h.most_common(50)
        if verbose:
            print('Words')
            print(h)
        with open(f"{out_dir}/cluster_{k}.txt", "w") as file:
            for word, count in h:
                file.write(f"{word} {count}\n")
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(f"{out_dir}/cluster_{k}.png")
        out_cluster_dir = f"{out_dir}/cluster_{k}"
        if not os.path.exists(out_cluster_dir):
            os.makedirs(out_cluster_dir)
        for title in titles:
            in_file = f"{in_dir}/{title}"
            out_file = f"{out_cluster_dir}/{title}"
            shutil.copyfile(in_file, out_file)
