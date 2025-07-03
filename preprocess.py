import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = re.sub(r'http\S+|@\w+|#\w+', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower()
    text = ' '.join([word for word in text.split() if len(word) > 1]) 
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
