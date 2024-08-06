import logging
import argparse
import os
import tempfile
import zipfile
from gensim.models import Word2Vec
from preprocess import Preprocessor

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

    args = parser.parse_args()

    goodware_folder = args.goodware_folder    
    malware_folder = args.malware_folder

    if not os.path.isdir(goodware_folder):
        print(f"Error: Path '{goodware_folder}' is not a valid directory.")
        return
    
    if not os.path.isdir(malware_folder):
        print(f"Error: Path '{malware_folder}' is not a valid directory.")
        return
    
    vocabulary = []

    goodware_files = os.listdir(goodware_folder)
    for file in goodware_files:
        logging.info(f'Processing file: {file}')
        file_vocabulary = process_file(goodware_folder, file)
        vocabulary.extend(file_vocabulary)

    malware_files = os.listdir(malware_folder)
    for file in malware_files:
        logging.info(f'Processing file: {file}')
        file_vocabulary = process_file(malware_folder, file)
        vocabulary.extend(file_vocabulary)

    word2vec_model = Word2Vec(sentences=vocabulary, vector_size=100, window=10, min_count=1, workers=8) # For workers on linux use: nproc
    word2vec_model.save("word2vec_disassembly.model")    
    

def process_file(folder, file):
    application, extension = os.path.splitext(file)
    zip_path = os.path.join(folder, file)
    if extension == '.zip':
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            for temp_file in os.listdir(temp_dir):
                temp_root, temp_extension = os.path.splitext(temp_file)
                file_path = os.path.join(temp_dir, temp_file)
                if temp_extension == '.txt':
                    return Preprocessor().clean_assembly_file(file_path)

if __name__ == "__main__":
    main()