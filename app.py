from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    return text

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_plagiarism():
    if request.method == 'POST':
        text1 = request.form['text1']
        text2 = request.form['text2']

        text1 = preprocess_text(text1)
        text2 = preprocess_text(text2)
  
        similarity_score = calculate_similarity(text1, text2)
        f="Not plagiarized code"
        if similarity_score>0.8:
            f="Plagiarized code"
        return render_template('result.html', score=similarity_score, fl=f)

if __name__ == '__main__':
    app.run(debug=True)
