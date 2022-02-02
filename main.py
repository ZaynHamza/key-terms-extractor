import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from lxml import etree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
# pip install -r requirements.txt


# preparing data
xml_path = "news.xml"
tree = etree.parse(xml_path)
root = tree.getroot()
corpus = root[0]
# creating exception list
stop_list = list(stopwords.words('english'))
p_list = list(string.punctuation)
stop_list.extend(p_list)
list_to_print = []

# creating dataset
dataset = []

# splitting file into blocks
for block in corpus:
    # taking first five tokens for every header
    list_to_print.append(block[0].text)
    # creating lemmatizer
    lemmatizer = WordNetLemmatizer()
    # tokenizing
    tokens = nltk.tokenize.word_tokenize(block[1].text.lower())
    # lemmatizing
    lemmatized_list = [lemmatizer.lemmatize(token) for token in tokens]
    # getting rid of punctuation and stopwords
    clean_list = [token for token in lemmatized_list if token not in stop_list]
    # creating list of only nouns
    nouns_list = []
    for token in clean_list:
        if nltk.pos_tag([token])[0][1] == 'NN':
            nouns_list.append(nltk.pos_tag([token])[0][0])
    dataset.append(' '.join(nouns_list))

# Vectorizing
stop = list(stopwords.words('english')) + ['ha', 'wa', 'u', 'a']
vectorizer = TfidfVectorizer(stop_words=stop)
tfidf_matrix = vectorizer.fit_transform(dataset).toarray()
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(tfidf_matrix)
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=vectorizer.get_feature_names_out(), columns=["idf_weights"])
count_vector = vectorizer.transform(dataset)
tf_idf_vector = tfidf_transformer.transform(count_vector)
feature_names = vectorizer.get_feature_names_out()
for x in range(0, 10):
    first_document_vector = tf_idf_vector[x]
    df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
    df.index.name = 'word'
    df = df.sort_values(['tfidf', 'word'], ascending=[False, False]).head(5)
    print(list_to_print[x] + ':')
    print(' '.join(list(df.index)), '\n')
