import logging
import argparse
import os
import tempfile
import zipfile
from gensim.models import Word2Vec
from preprocess import Preprocessor
import numpy as np

class iOSCorpus:
    def __init__(self, goodware_folder, malware_folder):
        self.files = []
        
        goodware_files = os.listdir(goodware_folder)
        for file in goodware_files[:50]:
            self.files.append((file, os.path.join(goodware_folder, file)))

        malware_files = os.listdir(malware_folder)
        for file in malware_files[:50]:
            self.files.append((file, os.path.join(malware_folder, file)))
        

    def __iter__(self):
        for (file, path) in self.files:
            logging.info(f'Processing file: {file}')
            file_vocabulary = process_file(path, file)
            for line in file_vocabulary:
                yield line

# Helper function to process a single file
def process_single_file(file_path, model, vector_folder, application):
    total_instructions = 0
    non_embedded_instructions = 0
    vector_file_path = os.path.join(vector_folder, f"{application}.txt")

    app_tokenized_instructions = process_file(file_path)
    with open(vector_file_path, "w") as vector_file:
        for instruction in app_tokenized_instructions:
            total_instructions += 1
            token = instruction[0]
            if token in model.wv:
                vector_file.write(f"{model.wv[token]}\n")
            else:
                non_embedded_instructions += 1
                vector_file.write(f"{np.zeros(model.vector_size)}\n")
    
    return total_instructions, non_embedded_instructions

# Moved process_zip_file outside
def process_zip_file(file, folder, vector_folder, model):
    logging.info(f"Processing File: {file}")
    path = os.path.join(folder, file)
    application, extension = os.path.splitext(os.path.basename(path))

    if extension == '.zip':
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            for temp_file in os.listdir(temp_dir):
                temp_root, temp_extension = os.path.splitext(temp_file)
                if temp_extension == '.txt':
                    file_path = os.path.join(temp_dir, temp_file)
                    return process_single_file(file_path, model, vector_folder, application)
    
    return 0, 0

# Process a folder of files (parallel)
def process_folder_parallel(folder, vector_folder, model, label):
    files = os.listdir(folder)
    
    total_instructions = 0
    non_embedded_instructions = 0

    for file in files:
        total, non_embedded = process_zip_file(file, folder, vector_folder, model)
        total_instructions += total
        non_embedded_instructions += non_embedded
    
    logging.info(f'{label} Total Instructions: {total_instructions}')
    logging.info(f'{label} Non-Embedded Instructions: {non_embedded_instructions}')
    logging.info(f'{label} Non-Embedded %: {(non_embedded_instructions / total_instructions) * 100:.2f}%')

    return total_instructions, non_embedded_instructions

# Helper function for processing each file
def process_file(path):
    application, extension = os.path.splitext(path)
    if extension == '.txt':
        return Preprocessor().clean_assembly_file(path)

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
    parser.add_argument("goodware_opcodes_folder", type=str, help="The folder where to store goodware opcodes files")
    parser.add_argument("malware_opcodes_folder", type=str, help="The folder where to store malware opcodes files")

    args = parser.parse_args()

    goodware_folder = args.goodware_folder    
    malware_folder = args.malware_folder
    goodware_opcodes_folder = args.goodware_opcodes_folder    
    malware_opcodes_folder = args.malware_opcodes_folder

    if not os.path.isdir(goodware_folder):
        print(f"Error: Path '{goodware_folder}' is not a valid directory.")
        return
    
    if not os.path.isdir(malware_folder):
        print(f"Error: Path '{malware_folder}' is not a valid directory.")
        return
    
    if not os.path.isdir(goodware_opcodes_folder):
        print(f"Error: Path '{goodware_opcodes_folder}' is not a valid directory.")
        return
    
    if not os.path.isdir(malware_opcodes_folder):
        print(f"Error: Path '{malware_opcodes_folder}' is not a valid directory.")
        return
    
    goodware_dir = os.listdir(goodware_folder)
    goodware_files = sorted(goodware_dir, key=lambda x: os.path.getsize(os.path.join(goodware_folder, x)))
    for file in goodware_files:
        logging.info(f'Processing file: {file}')
        filepath = os.path.join(goodware_folder, file)
        application, extension = os.path.splitext(os.path.basename(filepath))

        opcodes_files = os.listdir(goodware_opcodes_folder)
        if f"{application}.txt" not in opcodes_files:

            if extension == '.zip':
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)

                    for temp_file in os.listdir(temp_dir):
                        temp_root, temp_extension = os.path.splitext(temp_file)
                        if temp_extension == '.txt':
                            txt_path = os.path.join(temp_dir, temp_file)
                            opcodes = Preprocessor().get_opcodes_file(txt_path)

                            with open(f"{goodware_opcodes_folder}/{application}.txt", "w") as opcode_file:
                                for opcode in opcodes:
                                    opcode_file.write(f"{opcode}\n")

    malware_dir = os.listdir(malware_folder)
    malware_files = sorted(malware_dir, key=lambda x: os.path.getsize(os.path.join(malware_folder, x)))
    for file in malware_files:
        logging.info(f'Processing file: {file}')
        filepath = os.path.join(malware_folder, file)
        application, extension = os.path.splitext(os.path.basename(filepath))

        opcodes_files = os.listdir(malware_opcodes_folder)
        if f"{application}.txt" not in opcodes_files:

            if extension == '.zip':
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)

                    for temp_file in os.listdir(temp_dir):
                        temp_root, temp_extension = os.path.splitext(temp_file)
                        if temp_extension == '.txt':
                            txt_path = os.path.join(temp_dir, temp_file)
                            opcodes = Preprocessor().get_opcodes_file(txt_path)

                            with open(f"{malware_opcodes_folder}/{application}.txt", "w") as opcode_file:
                                for opcode in opcodes:
                                    opcode_file.write(f"{opcode}\n")

if __name__ == "__main__":
    main()
