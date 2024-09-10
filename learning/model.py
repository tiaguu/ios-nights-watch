import logging
import argparse
import os
import tempfile
import zipfile
from preprocess import Preprocessor
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
import re

def main():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers= [stream_handler]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("goodware_folder", type=str, help="The folder with goodware files")
    parser.add_argument("malware_folder", type=str, help="The folder with malware files")
    parser.add_argument("ios2vec_folder", type=str, help="The folder with ios2vec model")
    parser.add_argument("weights_folder", type=str, help="The folder where to store the model's weights")

    args = parser.parse_args()

    goodware_folder = args.goodware_folder
    malware_folder = args.malware_folder
    ios2vec_folder = args.ios2vec_folder
    weights_folder = args.weights_folder

    # Load the trained Word2Vec model
    # ios2vec_model = Word2Vec.load(f"{ios2vec_folder}/ios2vec.model")

    # logging.info(f'Loaded ios2vec.model')

    file_paths_and_labels = []

    goodware_dir = os.listdir(goodware_folder)
    goodware_files = sorted(goodware_dir, key=lambda x: os.path.getsize(os.path.join(goodware_folder, x)))[:3]
    for file in goodware_files:
        filepath = os.path.join(goodware_folder, file)
        file_labeled = (filepath, 0)
        file_paths_and_labels.append(file_labeled)

    malware_dir = os.listdir(malware_folder)
    malware_files = sorted(malware_dir, key=lambda x: os.path.getsize(os.path.join(malware_folder, x)))[:3]
    for file in malware_files:
        filepath = os.path.join(malware_folder, file)
        file_labeled = (filepath, 1)
        file_paths_and_labels.append(file_labeled)

    # Train/test split
    train_paths, test_paths = train_test_split(file_paths_and_labels, test_size=0.2, random_state=42)

    logging.info('Separated training and testing')
    logging.info(f'Training: {len(train_paths)}')
    logging.info(f'Testing: {len(test_paths)}')

    # Define LSTM model
    max_length = 5  # Define the maximum length of sequences
    vector_size = 8

    model = Sequential()
    model.add(LSTM(64, input_shape=(None, vector_size), return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

    logging.info(f'Defined the model')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    logging.info(f'Compiled the model')

    # Incremental training setup
    batch_size = 1
    num_epochs = 1

    for epoch in range(num_epochs):
        logging.info(f'Running epoch {epoch + 1}')
        np.random.shuffle(train_paths)  # Shuffle training data each epoch
        for i in range(0, len(train_paths)):
            logging.info(f'Running batch {str(i)}')
            file_path_and_label = train_paths[i]
            X_train, y_train = generate_embeddings_file(file_path_and_label)

            # Iterate over each chunk and train on it
            # for X_train in X_train_chunks:
            #     model.train_on_batch(np.array([X_train]), y_train)
            model.train_on_batch(X_train, y_train)
            
            logging.info(f'Trained on batch')
        logging.info(f'Epoch {epoch + 1} complete')

    # Save model weights
    model.save_weights(f'{weights_folder}/lstm_model.weights.h5')
    logging.info(f'Model weights saved to {weights_folder}/lstm_model.weights.h5')

    # Evaluate the model
    test_model(test_paths, model)

def test_model(test_paths, model):
    accuracies = []
    losses = []
    y_true = []  # Actual labels
    y_pred = []  # Predicted labels

    for test_sample in test_paths:
        X_test, y_test = generate_embeddings_file(test_sample, chunk_size=0)
        
        # Make sure X_test is wrapped in an extra dimension for batch processing
        X_test = np.array([X_test])

        # Get model predictions
        predictions = model.predict(X_test)
        predicted_label = (predictions > 0.5).astype(int)  # Assuming binary classification with threshold 0.5
        
        # Append the actual and predicted labels
        y_true.append(y_test)
        y_pred.append(predicted_label[0][0])

        # Evaluate the model on the single test instance
        loss, accuracy = model.evaluate(X_test, np.array([y_test]), verbose=0)
        
        # Collect the loss and accuracy for later averaging
        accuracies.append(accuracy)
        losses.append(loss)

    # Calculate the average accuracy and loss over all test samples
    average_accuracy = np.mean(accuracies)
    average_loss = np.mean(losses)

    logging.info(f'Average Loss: {average_loss:.4f}')
    logging.info(f'Average Accuracy: {average_accuracy * 100:.2f}%')

    # Build and display the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logging.info(f'Confusion Matrix:\n{cm}')

    return cm  # Return the confusion matrix if needed for further analysis

def get_embeddings_file(file_path_and_label, chunk_size=500):
    labels = []
    file_path, label = file_path_and_label

    logging.info(f"Processing file: {file_path}")

    embeddings = []  # List to store chunks of arrays
    labels = [label]  # List to store chunks of labels
    
    with open(file_path, 'r') as vector_file:
        vector = []
        for line in vector_file:
            cleaned_str = line.replace('[', '').replace(']', '').strip()
            for num in cleaned_str.split():
                vector.append(float(num))

            if len(vector) == 20:
                embeddings.append(vector)
                vector = []

    if chunk_size == 0:
        return np.array(embeddings), np.array(labels)
    else:
        # Split the sequence into smaller chunks of size `chunk_size`
        chunks = split_sequence_into_chunks(embeddings, chunk_size)

        if len(chunks[-1]) < chunk_size:
            for i in range(chunk_size - len(chunks[-1])):
                chunks[-1].append(np.zeros(20))
        
        return np.array(chunks), np.array(labels)
    
def split_sequence_into_chunks(sequence, chunk_size):
    """Splits the input sequence into smaller chunks of size `chunk_size`."""
    return [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]

def generate_embeddings_batch(file_paths, model, max_length):
    X_batch = []
    y_batch = []
    for i in range(0, len(file_paths)):
        logging.info(f'Running batch {str(i)}')
        file_path_and_label = file_paths[i]
        X, y = generate_embeddings_file(file_path_and_label, model, max_length, chunk_size = 0)
        X_batch.append(X)
        y_batch.append(y)

    return np.array(X_batch), np.array(y_batch)

def generate_embeddings_file(file_path_and_label, chunk_size=500):
    labels = []
    file_path, label = file_path_and_label
    vectors = [process_file(file_path)]
    # embeddings = generate_embedding_for_app(app_tokenized_instructions, model, max_length, chunk_size = chunk_size)
    labels.append(label)
    logging.info(f'Embeddings: {np.array(vectors).shape}')
    logging.info(f'Labels: {np.array(labels).shape}')
    return np.array(vectors), np.array(labels)

def generate_embedding_for_app(app_tokenized_instructions, model, max_length=50, chunk_size=500):
    embeddings = []
    
    # Generate embeddings for the instructions
    for instruction in app_tokenized_instructions:
        token = instruction[0]
        if token in model.wv:
            embeddings.append(model.wv[token])
        else:
            embeddings.append(np.zeros(model.vector_size))

    if chunk_size == 0:
        return np.array(embeddings)
    else:
        # Split the sequence into smaller chunks of size `chunk_size`
        chunks = split_sequence_into_chunks(embeddings, chunk_size)

        if len(chunks[-1]) < chunk_size:
            for i in range(chunk_size - len(chunks[-1])):
                chunks[-1].append(np.zeros(model.vector_size))
        
        # Return the padded chunks as the final input
        return np.array(chunks)

def process_file(path):
    application, extension = os.path.splitext(os.path.basename(path))
    if extension == '.zip':
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            for temp_file in os.listdir(temp_dir):
                temp_root, temp_extension = os.path.splitext(temp_file)
                file_path = os.path.join(temp_dir, temp_file)
                if temp_extension == '.txt':
                    return Preprocessor().clean_assembly_file(file_path)

if __name__ == "__main__":
    main()
