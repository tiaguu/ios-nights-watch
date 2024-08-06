import logging
import argparse
import os
import tempfile
import zipfile
from gensim.models import Word2Vec
from preprocess import Preprocessor

class iOSCorpus:
    def __init__(self, goodware_folder, malware_folder):
        self.files = []
        
        goodware_files = os.listdir(goodware_folder)
        for file in goodware_files:
            self.files.append((file, os.path.join(goodware_folder, file)))

        malware_files = os.listdir(malware_folder)
        for file in malware_files:
            self.files.append((file, os.path.join(malware_folder, file)))
        

    def __iter__(self):
        for (file, path) in self.files:
            logging.info(f'Processing file: {file}')
            file_vocabulary = process_file(path, file)
            for line in file_vocabulary:
                yield line

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

    corpus = iOSCorpus(goodware_folder = goodware_folder, malware_folder = malware_folder)
    word2vec_model = Word2Vec(corpus, vector_size=100, window=10, min_count=1, workers=32) # For workers on linux use: nproc

    word2vec_model.save("ios2vec.model")    
    

def process_file(path, file):
    application, extension = os.path.splitext(file)
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