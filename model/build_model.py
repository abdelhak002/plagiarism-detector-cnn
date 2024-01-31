import pickle
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Concatenate, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
df = pd.read_csv('dataset/plagiarism_data.csv')
source_texts = df['source'].tolist()
suspicious_texts = df['suspicious'].tolist()
labels = df['label'].astype(int).values.reshape(-1, 1)

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(source_texts + suspicious_texts)

# Convert texts to sequences and pad
source_sequences = tokenizer.texts_to_sequences(source_texts)
suspicious_sequences = tokenizer.texts_to_sequences(suspicious_texts)

# Determine the maximum sequence length dynamically
max_length = max(
    max(len(seq) for seq in source_sequences),
    max(len(seq) for seq in suspicious_sequences)
)


# Pad sequences to the dynamically determined maximum length
source_padded = pad_sequences(source_sequences, maxlen=max_length, padding='post')
suspicious_padded = pad_sequences(suspicious_sequences, maxlen=max_length, padding='post')

# Build the model
input_layer_source = Input(shape=(max_length,))
input_layer_suspicious = Input(shape=(max_length,))

embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=300)(input_layer_source)

# TimeDistributed layers
td_layer_1 = TimeDistributed(Dense(300))(embedding_layer)
td_layer_2 = TimeDistributed(Dense(128))(td_layer_1)
td_layer_3 = TimeDistributed(Dense(128))(td_layer_2)
td_layer_4 = TimeDistributed(Dense(128))(td_layer_3)
td_layer_5 = TimeDistributed(Dense(128))(td_layer_4)

# Concatenate
concatenated_layer = Concatenate(axis=-1)([td_layer_1, td_layer_2, td_layer_3, td_layer_4, td_layer_5])

# Dropout and Flatten
dropout_layer_1 = Dropout(0.5)(concatenated_layer)
flatten_layer = Flatten()(dropout_layer_1)

# Second Dropout
dropout_layer_2 = Dropout(0.5)(flatten_layer)

# Dense layer
output_layer = Dense(1, activation='sigmoid')(dropout_layer_2)

model = Model(inputs=[input_layer_source, input_layer_suspicious], outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([source_padded, suspicious_padded], labels, epochs=10, batch_size=32)

# Save the trained model
model.save('model/plagiarism_detector_model.keras')

# Save the tokenizer
with open('model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
