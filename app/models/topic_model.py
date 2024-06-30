import joblib

# Load the LDA model and vectorizer
lda_model = joblib.load('D:/mental_health_project/lda_model.joblib')
vectorizer = joblib.load('D:/mental_health_project/vectorizer.joblib')

def extract_topics(text):
    X = vectorizer.transform([text])
    topic_distribution = lda_model.transform(X)
    return topic_distribution
