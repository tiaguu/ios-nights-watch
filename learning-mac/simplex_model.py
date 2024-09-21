from os import putenv

putenv("PYTORCH_ROCM_ARCH", "gfx1011")
putenv("HSA_OVERRIDE_GFX_VERSION", "10.1.1")

from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import logging
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load environment variables from the .env file
load_dotenv()

# PyTorch model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, lengths):
        # Pack the padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # We only care about the output of the last time step
        out = hidden[-1]  # Take the hidden state of the last LSTM layer
        out = self.fc(out)
        return out

def main():
    # Logging configuration
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[stream_handler])

    # Set up logging
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        logging.info(f"GPU Available. GPU in use: {gpu_name}")
    elif torch.backends.mps.is_available():  # For Apple M1/M2 with Metal Performance Shaders
        logging.info("MPS (Metal Performance Shaders) is available on macOS.")
        logging.info(f"MPS is built: {torch.backends.mps.is_built()}")
    else:
        logging.info("No compatible GPU found. Using CPU.")

    logging.info(f'Torch version: {torch.__version__}')  # Check PyTorch version
    
    files = {}

    goodware_files_dir = 'goodware-opcodes'
    goodware_files = os.listdir(goodware_files_dir)
    sorted_goodware_files = sorted(goodware_files, key=lambda x: os.path.getsize(os.path.join(goodware_files_dir, x)))
    for goodware_file in sorted_goodware_files[0:20]:
        goodware_file_path = f'{goodware_files_dir}/{goodware_file}'
        files[goodware_file_path] = 0

    malware_files_dir = 'malware-opcodes'
    malware_files = os.listdir(malware_files_dir)
    sorted_malware_files = sorted(malware_files, key=lambda x: os.path.getsize(os.path.join(malware_files_dir, x)))
    for malware_file in sorted_malware_files[0:20]:
        malware_file_path = f'{malware_files_dir}/{malware_file}'
        files[malware_file_path] = 1

    # Train/test split
    train_files, test_files = train_test_split(list(files.keys()), test_size=0.2, random_state=42)
    logging.info('Separated training and testing')
    logging.info(f'Training: {len(train_files)}')
    logging.info(f'Testing: {len(test_files)}')

    train_files_data = {}
    for file in train_files:
        train_files_data[file] = files[file]
    logging.info(f'Training: {(train_files_data)}')

    test_files_data = {}
    for file in test_files:
        test_files_data[file] = files[file]
    logging.info(f'Testing: {(test_files_data)}')

    # Model hyperparameters
    input_size = 8  # Each input vector has 8 features
    hidden_size = 128
    output_size = 1  # Binary classification

    # Initialize the model
    model = LSTMModel(input_size, hidden_size, output_size)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logging.info(f'Device: {device}')
    model.to(device)

    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare the data for training
    batch_size = 5
    sorted_train_files = sort_files_by_sequence_length(train_files_data)
    batches = create_batches(sorted_train_files, batch_size)

    # Training loop
    for epoch in range(10):  # Replace with actual number of epochs
        logging.info(f'Starting epoch {epoch + 1}')
        for batch_files, batch_labels in batches:
            logging.info(f'Processing batch of size {len(batch_files)}')

            # Generate embeddings for the batch
            X_train, y_train, lengths = generate_variable_length_embeddings(batch_files)
            X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train = torch.tensor(batch_labels, dtype=torch.float32).to(device)

            # Forward pass
            outputs = model(X_train, lengths)

            # Ensure outputs and y_train have the same shape
            outputs = outputs.view(-1)  # Reshape outputs to [batch_size]
            y_train = y_train.view(-1)  # Reshape y_train to [batch_size]

            # Compute loss
            loss = criterion(outputs, y_train)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(f'Epoch [{epoch+1}], Batch Loss: {loss.item():.4f}')

    logging.info('Training complete')

    # Testing phase - evaluate the model (single-file processing)
    model.eval()  # Set the model to evaluation mode

    y_true = []
    y_pred = []

    for file, label in test_files_data.items():
        logging.info(f'Testing file: {file}')

        # Generate embeddings for the full file
        X_test, _, lengths = generate_variable_length_embeddings([file])
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

        with torch.no_grad():  # Disable gradient calculation during evaluation
            test_outputs = model(X_test, lengths)
            logging.info(test_outputs)
            test_outputs = torch.sigmoid(test_outputs).cpu().numpy()  # Convert to probabilities
            test_prediction = (test_outputs > 0.5).astype(int).squeeze()  # Apply a threshold

            y_true.append(label)
            y_pred.append(test_prediction)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'Precision: {precision:.4f}')
    logging.info(f'Recall: {recall:.4f}')
    logging.info(f'F1 Score: {f1:.4f}')
    logging.info(f'Confusion Matrix:\n{conf_matrix}')

def sort_files_by_sequence_length(files_data):
    """
    Sorts the files by the number of lines (sequence length).
    """
    lengths = {}
    for file in files_data.keys():
        lengths[file] = len(get_vectors_from_file(file))
    
    return sorted(files_data.items(), key=lambda item: lengths[item[0]])

def create_batches(sorted_files, batch_size):
    """
    Groups files of similar lengths into batches.
    """
    batches = []
    current_batch = []
    current_labels = []

    for i, (file, label) in enumerate(sorted_files):
        current_batch.append(file)
        current_labels.append(label)

        if len(current_batch) == batch_size or i == len(sorted_files) - 1:
            batches.append((current_batch, current_labels))
            current_batch = []
            current_labels = []

    return batches

def generate_variable_length_embeddings(files, input_size=8):
    """
    Generates variable-length embeddings for a batch of files.
    """
    data = []
    lengths = []
    labels = []

    for file in files:
        vectors = get_vectors_from_file(file)
        lengths.append(len(vectors))
        data.append(vectors)

    # Pad sequences within the batch to the length of the longest sequence in the batch
    max_len = max(lengths)
    for i in range(len(data)):
        if len(data[i]) < max_len:
            padding = [[0.0] * input_size] * (max_len - len(data[i]))
            data[i].extend(padding)

    return data, labels, lengths

def get_vectors_from_file(file):
    with open(file, 'r') as vector_file:
        vectors = []
        for line in vector_file:
            if len(line) > 0:
                opcode = line.strip()
                key = find_opcode_key(opcode)
                vector = [0.0 for _ in range(8)]
                if key is not None:
                    vector[key] = 1.0
                vectors.append(vector)

    return vectors
    
def find_opcode_key(opcode_value):
    opcodes = {
        0 : ['b', 'bl', 'bx', 'blx'], # Unconditional Branches
        1 : ['b.eq', 'b.ne', 'b.lt', 'b.gt', 'b.le', 'b.ge', 'b.hi', 'b.lo', 'b.pl', 'b.mi', 'b.vs', 'b.vc', 'b.cs', 'b.cc', 'b.al', 'b.nv'], # Conditional Branches
        2 : ['cbz', 'cbnz'], # Compare and Branch
        3 : ['tbz', 'tbnz'], # Test and Branch
        4 : ['ret', 'eret'], # Return Instructions
        5 : ['svc', 'hvc', 'smc', 'brk', 'hlt'], # Exception Generation
        6 : ['br', 'blr', 'braa', 'brab', 'retab'], # Indirect Branching
        7 : ['nop', 'yield', 'wfe', 'wfi', 'sev', 'sevl', 'isb', 'dmb', 'dsb'] # Hints
    }

    for key, values in opcodes.items():
        if opcode_value in values:
            return key
    return None

main()