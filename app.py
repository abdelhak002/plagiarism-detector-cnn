from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)

def load_trained_model_and_tokenizer(model_path='model/plagiarism_detector_model.keras', tokenizer_path='model/tokenizer.pickle'):
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def predict_plagiarism(model, tokenizer, source_text, suspicious_text, threshold=0.5):
    source_text_str = source_text.tolist() if isinstance(source_text, np.ndarray) else source_text
    suspicious_text_str = suspicious_text.tolist() if isinstance(suspicious_text, np.ndarray) else suspicious_text

    source_sequence = tokenizer.texts_to_sequences([source_text_str])
    suspicious_sequence = tokenizer.texts_to_sequences([suspicious_text_str])

    max_length = 15
    source_padded = pad_sequences(source_sequence, maxlen=max_length, padding='post')
    suspicious_padded = pad_sequences(suspicious_sequence, maxlen=max_length, padding='post')

    predictions = model.predict([source_padded, suspicious_padded])

    print("Model Predictions:", predictions)

    percent_plagiarism = predictions[0][0] * 100

    if predictions[0][0] > threshold:
        return "Plagiarism Detected with {:.2f}% Confidence".format(percent_plagiarism)
    else:
        return "No Plagiarism Detected with {:.2f}% Confidence".format(100 - percent_plagiarism)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        source_text = request.form['source_text']
        suspicious_text = request.form['suspicious_text']

        if not source_text or not suspicious_text:
            result = "Please enter valid source and suspicious texts."
        else:
            model, tokenizer = load_trained_model_and_tokenizer()
            result = predict_plagiarism(model, tokenizer, source_text, suspicious_text)

    return render_template('index.html', result=result)
if __name__ == '__main__':
    app.run(debug=True)
