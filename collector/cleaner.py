import os
import argparse
import logging
import zipfile
import tempfile

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers= [stream_handler]
)

# Create the parser
parser = argparse.ArgumentParser(description='Clean up instances that were not disassembled correctly.')

# Add arguments
parser.add_argument("folder", type=str, help="The folder where instances are stored")

# Parse the arguments
args = parser.parse_args()
dirt_folder = args.folder

logging.info(f"Running cleaner on folder {dirt_folder}")

for unchecked_file in os.listdir(dirt_folder):
    application, extension = os.path.splitext(unchecked_file)
    zip_path = os.path.join(dirt_folder, unchecked_file)
    if extension == '.zip':
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            for temp_file in os.listdir(temp_dir):
                temp_root, temp_extension = os.path.splitext(temp_file)
                file_path = os.path.join(temp_dir, temp_file)
                if temp_extension == '.txt':
                    with open(file_path, 'r') as file:
                        content = file.read()
                        lines = content.splitlines()
                        if len(lines) < 300:
                            logging.info(f"––––––––––––––––––––––––––––––––––")
                            logging.info(f"Application: {application}")
                            logging.info(f"Lines: {len(lines)}")
                            logging.info(f"")
                            logging.info(content)
                            
                            

    