import re
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
from nltk.corpus import stopwords


app =Flask(__name__)

## data preprocessing function
def preprocess_text(text_data):
    sentence = re.sub(r'[\w\s]',' ',text_data)
    word = [' '.join(token.lower()
                     for token in nltk.word_tokenize(sentence)
                     if token not in stopwords.words('english'))]
    return word

# Load the model and vectorizer
model = pickle.load(open('Voting_model.pkl','rb'))
vectorizer = pickle.load(open('Tfidf_Vectorizer.pkl','rb'))

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    review_text = request.form['review_text']
    preprocessed_text = preprocess_text(review_text)
    transformed_text = vectorizer.transform(preprocessed_text)
    
    prediction = model.predict(transformed_text)
    
    return render_template('result.html', prediction=prediction[0])


if __name__ =="__main__":
    app.run(debug=True)