# src/topic_utils.py
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
STOP = set(stopwords.words('english'))
dictionary = None
lda_model = None
corpus = None

def preprocess_texts(texts):
    out = []
    for t in texts:
        tokens = [w for w in word_tokenize(str(t).lower()) if w.isalpha() and w not in STOP]
        out.append(tokens)
    return out

def build_topics(texts, num_topics=6):
    global dictionary, lda_model, corpus
    tokenized = preprocess_texts(texts)
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(t) for t in tokenized]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    topics = []
    for i in range(num_topics):
        topics.append([word for word,prob in lda_model.show_topic(i, topn=8)])
    return topics

def assign_topic(text):
    if lda_model is None or dictionary is None:
        return "No model (build topics first)"
    bow = dictionary.doc2bow(preprocess_texts([text])[0])
    scores = lda_model.get_document_topics(bow)
    best = sorted(scores, key=lambda x: -x[1])[0]
    return best
