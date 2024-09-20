from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

import logging
import argparse
import os
import tempfile
import zipfile
from preprocess import Preprocessor
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

# PyTorch model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # For binary classification

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Taking the output from the last time step
        return self.sigmoid(out)

def main():
    # Set up logging
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        logging.info(f"CUDA Available. GPU in use: {gpu_name}")
    elif torch.backends.mps.is_available():  # For Apple M1/M2 with Metal Performance Shaders
        logging.info("MPS (Metal Performance Shaders) is available on macOS.")
    elif torch.has_mps:
        logging.info("MPS is supported but no compatible GPU found.")
    else:
        logging.info("No compatible GPU found. Using CPU.")

#     # Logging configuration
#     stream_handler = logging.StreamHandler()
#     stream_handler.setLevel(logging.INFO)
#     logging.basicConfig(level=logging.DEBUG,
#                         format='%(asctime)s - %(levelname)s - %(message)s',
#                         handlers=[stream_handler])

#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()

#     download_url = 'http://95.168.166.236:8000/download/opcodes'
#     goodware_urls = [f'{download_url}/goodware/{x}' for x in range(10)]
#     malware_urls = [f'{download_url}/malware/{x}' for x in range(10)]
#     urls = goodware_urls + malware_urls

#     # Train/test split
#     train_paths, test_paths = train_test_split(urls, test_size=0.2, random_state=42)
#     logging.info('Separated training and testing')
#     logging.info(f'Training: {len(train_paths)}')
#     logging.info(f'Testing: {len(test_paths)}')

#     # Model hyperparameters
#     input_size = 8
#     hidden_size = 64
#     output_size = 1  # Binary classification
#     max_length = 5

#     # Initialize the model
#     model = LSTMModel(input_size, hidden_size, output_size)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)

#     # Define loss and optimizer
#     criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Dummy training loop (You can replace it with actual data loading)
#     for epoch in range(2):  # Replace with actual number of epochs
#         logging.info(f'Starting epoch {epoch + 1}')
#         for i, train_path in enumerate(train_paths):
#             X_train, y_train = generate_embeddings_file(train_path)
#             X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
#             y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

#             # Forward pass
#             outputs = model(X_train)
#             loss = criterion(outputs.squeeze(), y_train)
            
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             logging.info(f'Epoch [{epoch+1}], Batch [{i+1}/{len(train_paths)}], Loss: {loss.item():.4f}')
    
#     logging.info('Training complete')
#     test_model(test_paths, model)

# def test_model(test_paths, model):
#     model.eval()  # Set model to evaluation mode
#     accuracies = []
#     y_true = []
#     y_pred = []

#     with torch.no_grad():
#         for test_sample in test_paths:
#             X_test, y_test = generate_embeddings_file(test_sample)
#             X_test = torch.tensor(X_test, dtype=torch.float32)
#             y_test = torch.tensor(y_test, dtype=torch.float32)

#             outputs = model(X_test)
#             predicted = (outputs > 0.5).float()  # Binary classification threshold
#             y_true.append(y_test.item())
#             y_pred.append(predicted.item())

#     cm = confusion_matrix(y_true, y_pred)
#     logging.info(f'Confusion Matrix:\n{cm}')

# def generate_embeddings_file(file_path_and_label):
#     file_path, label = file_path_and_label
#     # Simulating embedding generation with dummy data for this example
#     vectors = np.random.rand(5, 8)  # Replace with actual data
#     return vectors, label

main()