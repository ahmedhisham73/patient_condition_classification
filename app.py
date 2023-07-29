from flask import Flask, render_template, request
import spacy
import pickle

app = Flask(__name__)

# Load the English language model
nlp = spacy.blank('en')

def preprocess_text(text):
    # Apply the pipeline to your text
    doc = nlp(text)
    
    # Tokenize, lower case, and lemmatize the text
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    
    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Load the model
with open('ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get the review from the form
        review = request.form['review']

        # Preprocess the review
        preprocessed_review = preprocess_text(review)

        # Make the prediction
        prediction = model.predict([preprocessed_review])[0]

        # You may want to convert the prediction back to the original label here
        # predicted_label = label_encoder.inverse_transform(prediction)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

