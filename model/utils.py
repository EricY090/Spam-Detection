import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


stopwords = stopwords.words('english')
ps = PorterStemmer()

### Preprocessing the text
def text_preprocess(s):
    temp = re.sub(r'\d+', "", s)        ### Remove numbers
    temp = re.sub(r"http\S+", "", temp)     ### Remove urls
    temp = "".join([char.lower() for char in temp if char not in string.punctuation])       ### Lower case and remove punctuations
    temp = nltk.word_tokenize(temp)     ### Tokenize
    temp = np.array([ps.stem(x) for x in temp if x not in stopwords])     ### Remove stopwords then word stemming the rest
    return temp


### calculate tf
def calculate_tf(x):
    tf = []
    for idx, i in enumerate(x):
        temp = dict()
        for word in i:
            temp[word] = temp[word] +1 if word in temp else 1
        temp[0] = len(i)
        tf.append(temp)
    return tf

### TF-IDF: handles the unseen word by ignoring
def build_tf_idf(tf, vocab, vocab_id, total_docs):
    result = np.zeros(shape=(len(tf), len(vocab_id)), dtype = np.float32)
    for idx, i in enumerate(tf):
        length = i[0]
        for word, freq in i.items():
            if word == 0: continue
            ### Unseen word
            if word not in vocab_id:
                #print(f"Data {idx} has a word {word} that is unseen in training set")
                continue
            id = vocab_id[word]                 ### Id for the word
            result[idx][id] = (freq/length) * math.log2(total_docs/vocab[word])        ### tf-idf
    return result

### Sigmoid function
def expit(x):
    return 1/(1+np.exp(-x))


def evaluate(x, y, w, b):
    y_predict = expit(np.array(np.dot(x, w.T) + b, dtype=np.float32))
    for idx, i in enumerate(y_predict):
        y_predict[idx] = 1 if y_predict[idx] > 0.5 else 0
    y_predict = np.array(y_predict, dtype= int)

    f1 = f1_score(y, y_predict)
    precision = precision_score(y, y_predict)
    recall = recall_score(y, y_predict)
    accuracy = accuracy_score(y, y_predict)
    return f1, precision, recall, accuracy