import logging
import argparse
import os
import tempfile
import zipfile
from gensim.models import Word2Vec
from preprocess import Preprocessor
import numpy as np
from concurrent.futures import ProcessPoolExecutor

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
    parser.add_argument("model_folder", type=str, help="The folder with model files")
    parser.add_argument("goodware_vector_folder", type=str, help="The folder where to store goodware vector files")
    parser.add_argument("malware_vector_folder", type=str, help="The folder where to store malware vector files")

    args = parser.parse_args()

    goodware_folder = args.goodware_folder    
    malware_folder = args.malware_folder
    model_folder = args.model_folder
    goodware_vector_folder = args.goodware_vector_folder    
    malware_vector_folder = args.malware_vector_folder

    if not os.path.isdir(goodware_folder):
        print(f"Error: Path '{goodware_folder}' is not a valid directory.")
        return
    
    if not os.path.isdir(malware_folder):
        print(f"Error: Path '{malware_folder}' is not a valid directory.")
        return
    
    if not os.path.isdir(model_folder):
        print(f"Error: Path '{model_folder}' is not a valid directory.")
        return
    
    if not os.path.isdir(goodware_vector_folder):
        print(f"Error: Path '{goodware_vector_folder}' is not a valid directory.")
        return
    
    if not os.path.isdir(malware_vector_folder):
        print(f"Error: Path '{malware_vector_folder}' is not a valid directory.")
        return

    model = Word2Vec.load(f"{model_folder}/ios2vec.model")

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

    # Process a folder of files (parallel)
    def process_folder_parallel(folder, vector_folder, model, label):
        files = os.listdir(folder)
        vector_files = os.listdir(vector_folder)
        
        total_instructions = 0
        non_embedded_instructions = 0

        def process_zip_file(file):
            path = os.path.join(folder, file)
            application, extension = os.path.splitext(os.path.basename(path))
            
            if f"{application}.txt" not in vector_files and extension == '.zip':
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)

                    for temp_file in os.listdir(temp_dir):
                        temp_root, temp_extension = os.path.splitext(temp_file)
                        if temp_extension == '.txt':
                            file_path = os.path.join(temp_dir, temp_file)
                            return process_single_file(file_path, model, vector_folder, application)
            
            return 0, 0

        with ProcessPoolExecutor() as executor:
            results = executor.map(process_zip_file, files)

        for total, non_embedded in results:
            total_instructions += total
            non_embedded_instructions += non_embedded
        
        logging.info(f'{label} Total Instructions: {total_instructions}')
        logging.info(f'{label} Non-Embedded Instructions: {non_embedded_instructions}')
        logging.info(f'{label} Non-Embedded %: {(non_embedded_instructions / total_instructions) * 100:.2f}%')

        return total_instructions, non_embedded_instructions

    # Process both goodware and malware folders
    malware_total, malware_non_embedded = process_folder_parallel(
        malware_folder, malware_vector_folder, model, 'Malware'
    )
    
    goodware_total, goodware_non_embedded = process_folder_parallel(
        goodware_folder, goodware_vector_folder, model, 'Goodware'
    )

    # Summarize results
    total_instructions = goodware_total + malware_total
    non_embedded_instructions = goodware_non_embedded + malware_non_embedded
    logging.info(f'Total Instructions: {total_instructions}')
    logging.info(f'Total Non-Embedded Instructions: {non_embedded_instructions}')
    logging.info(f'Total Non-Embedded %: {(non_embedded_instructions / total_instructions) * 100:.2f}%')

    # corpus = iOSCorpus(goodware_folder = goodware_folder, malware_folder = malware_folder)
    # word2vec_model = Word2Vec(corpus, vector_size=20, window=3, min_count=1, sample=0.0, workers=32, epochs=1) # For workers on linux use: nproc

    # word2vec_model.save(os.path.join(model_folder, "ios2vec.model"))    

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
