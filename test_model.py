from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np  # Add this line to import NumPy

def load_trained_model_and_tokenizer(model_path='model/plagiarism_detector_model.keras', tokenizer_path='model/tokenizer.pickle'):
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def predict_plagiarism(model, tokenizer, source_text, suspicious_text, threshold=0.5):
    # Convert texts to strings
    source_text_str = source_text.tolist() if isinstance(source_text, np.ndarray) else source_text
    suspicious_text_str = suspicious_text.tolist() if isinstance(suspicious_text, np.ndarray) else suspicious_text

    # Tokenize the texts
    source_sequence = tokenizer.texts_to_sequences([source_text_str])
    suspicious_sequence = tokenizer.texts_to_sequences([suspicious_text_str])

    # Adjust the length of the sequences to 11
    max_length = 15
    source_padded = pad_sequences(source_sequence, maxlen=max_length, padding='post')
    suspicious_padded = pad_sequences(suspicious_sequence, maxlen=max_length, padding='post')

    predictions = model.predict([source_padded, suspicious_padded])

    if predictions[0][0] > threshold:
        return "Plagiarism Detected"
    else:
        return "No Plagiarism Detected"

if __name__ == "__main__":
    model, tokenizer = load_trained_model_and_tokenizer()
    source_text = "Machine learning involves training models on data."
    suspicious_text = "Data training is part of machine learning."

    result = predict_plagiarism(model, tokenizer, source_text, suspicious_text)
    print(result)

