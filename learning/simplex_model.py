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

# Load environment variables from the .env file
load_dotenv()

# PyTorch model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # For binary classification

    def forward(self, x, lengths):
        # Pack the padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # We only care about the output of the last time step
        out = hidden[-1]  # Take the hidden state of the last LSTM layer
        out = self.fc(out)
        return self.sigmoid(out)

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
    else:
        logging.info("No compatible GPU found. Using CPU.")

    download_url = 'http://95.168.166.236:8000/download/opcodes'
    goodware_urls = [f'{download_url}/goodware/{x}' for x in range(10)]
    malware_urls = [f'{download_url}/malware/{x}' for x in range(10)]
    urls = goodware_urls + malware_urls

    # Train/test split
    train_paths, test_paths = train_test_split(urls, test_size=0.2, random_state=42)
    logging.info('Separated training and testing')
    logging.info(f'Training: {len(train_paths)}')
    logging.info(f'Testing: {len(test_paths)}')

    # Model hyperparameters
    input_size = 8  # Each input vector has 8 features
    hidden_size = 64
    output_size = 1  # Binary classification
    max_length = 5

    # Initialize the model
    model = LSTMModel(input_size, hidden_size, output_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')
    model.to(device)

    # Define loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy training loop (You can replace it with actual data loading)
    for epoch in range(2):  # Replace with actual number of epochs
        logging.info(f'Starting epoch {epoch + 1}')
        # Simulate data loading for variable-length sequences
        X_train, y_train, lengths = generate_variable_length_embeddings(urls_data = train_paths)
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

        # Forward pass
        outputs = model(X_train, lengths)
        loss = criterion(outputs.squeeze(), y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logging.info(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')
    
    logging.info('Training complete')

def generate_variable_length_embeddings(urls_data, batch_size = 16, max_seq_length = 150, input_size = 8):
    data = []
    for url in urls_data:
        get_vectors_from_url(url = url)

    # # Simulating different lengths for each sequence
    # lengths = np.random.randint(1, max_seq_length + 1, size=batch_size)
    # data = [np.random.rand(length, input_size) for length in lengths]

    # # Pad sequences to the same length
    # padded_data = np.zeros((batch_size, max_seq_length, input_size))
    # for i, seq in enumerate(data):
    #     padded_data[i, :len(seq), :] = seq

    # # Dummy binary labels
    # labels = np.random.randint(0, 2, size=batch_size)
    
    # return padded_data, labels, lengths

def get_vectors_from_url(url):
    response = requests.get(url)

    if response.status_code == 200:
        plain_text = response.text
        for line in plain_text.split('\n'):
            if len(line) > 0:
                opcode = line.strip()

                key = find_opcode_key(opcode)
                vector = [0.0 for x in range(8)]

                if key is not None:
                    vector[key] = 1.0
                else:
                    logging.info(vector)
    else:
        logging.info(f"Failed to retrieve the file. Status code: {response.status_code}")
        return []
    
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
