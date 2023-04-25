from flask import Flask, request, jsonify, current_app as capp
from adjusting_input import Text_Model
import pickle


app = Flask(__name__)
with app.app_context():
    app.model = pickle.load(open('app/models_file/lr_sentiment_classifier.pickle', 'rb'))
    app.tfidf = pickle.load(open('app/models_file/tfidf.pickle', 'rb'))
    app.text_model = Text_Model


@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()['frase']

    # instantiating class that will apply
    # methods to fix text
    text_obj = capp.text_model(req, capp.tfidf)
    text_obj = text_obj.adjusting_text().measuring_relevance()
    vector = text_obj.tfidf_vector

    # prediction
    sentiment = capp.model.predict(vector)[0]
    sentiment = "positivo" if sentiment == 1 else "negativo"

    return jsonify({'sentimento': f'{sentiment}'}), 200


if __name__ == "__main__":
    app.run()
