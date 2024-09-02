import logging
import argparse
import os
import tempfile
import zipfile
from preprocess import Preprocessor
import tiktoken

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

    goodware_files = os.listdir(goodware_folder)
    for file in goodware_files:
        filepath = os.path.join(goodware_folder, file)
        processed_file = process_file(filepath)
        lines = []
        word_count = 0
        for line in processed_file:
            word_count += len(line)
            line_text = ' '.join(line)
            lines.append(line_text)
        file_text = ('\n').join(lines)
        num_tokens = num_tokens_from_string(file_text, "cl100k_base")

        logging.info('-----------------')
        logging.info(file)
        logging.info(f'Number of lines: {len(lines)}')
        logging.info(f'Number of words: {word_count}')
        logging.info(f'Number of tokens: {num_tokens}')

    malware_files = os.listdir(malware_folder)
    for file in malware_files:
        filepath = os.path.join(malware_folder, file)
        processed_file = process_file(filepath)
        lines = []
        word_count = 0
        for line in processed_file:
            word_count += len(line)
            line_text = ' '.join(line)
            lines.append(line_text)
        file_text = ('\n').join(lines)
        num_tokens = num_tokens_from_string(file_text, "cl100k_base")

        logging.info('-----------------')
        logging.info(file)
        logging.info(f'Number of lines: {len(lines)}')
        logging.info(f'Number of words: {word_count}')
        logging.info(f'Number of tokens: {num_tokens}')
        

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
                
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

if __name__ == "__main__":
    main()
