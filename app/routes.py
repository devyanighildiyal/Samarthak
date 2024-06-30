from flask import Blueprint, request, jsonify, render_template
from .models import sentiment_model, topic_model, ner_model

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.json['user_input']
    sentiment = sentiment_model.analyze_sentiment(user_input)
    topics = topic_model.extract_topics(user_input)
    entities = ner_model.extract_entities(user_input)
    # Add your recommendation logic here based on sentiment, topics, and entities
    recommendations = ["Recommendation 1", "Recommendation 2"] # Placeholder
    return jsonify({
        'sentiment': sentiment,
        'topics': topics.tolist(),
        'entities': entities,
        'recommendations': recommendations
    })
