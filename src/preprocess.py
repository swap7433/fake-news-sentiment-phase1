import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ensure resources installed
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'http\S+', '', text)          # remove urls
    text = re.sub(r'[^a-z0-9\s]', ' ', text)    # remove punctuation
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)
