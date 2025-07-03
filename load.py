import pandas as pd
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from preprocess import preprocess_text  

nltk.download('stopwords')
nltk.download('wordnet')

cols = ['target', 'id', 'date', 'query', 'user', 'text']
df = pd.read_csv("dataSet.csv", encoding='latin-1', names=cols)
df = df[['text', 'target']]
df['target'] = df['target'].replace(4, 1)
df['cleaned_text'] = df['text'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['target'],
    test_size=0.2,
    random_state=42,
    stratify=df['target']
)
# Vectorize words using TF-IDF (Term Frequency–Inverse Document Frequency)
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=100000,
    sublinear_tf=True,
    min_df=5,
    max_df=0.7,
    stop_words='english'
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)


base_model = LinearSVC(
    C=0.1,
    class_weight='balanced',
    max_iter=2000,
    loss='squared_hinge',
    random_state=42
)
model = CalibratedClassifierCV(base_model)
model.fit(X_train_vec, y_train)


joblib.dump(model, 'sentiment_model.joblib')  
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')  
