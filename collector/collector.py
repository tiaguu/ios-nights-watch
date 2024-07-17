import subprocess
import os
import argparse
import pandas as pd
import logging
from disassemble import Disassembler
import zipfile
import signal
import requests

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers= [stream_handler]
)

# Create the parser
parser = argparse.ArgumentParser(description='Collect decrypted .ipas from an iOS device and upload them.')

# Add arguments
parser.add_argument('--host', type=str)
parser.add_argument('--port', type=str)

# Parse the arguments
args = parser.parse_args()

# Access the arguments
host = args.host
port = args.port

logging.info(f"Running collector on Host {host} and Port {port}")

with open('./dataset/applications_dataset.txt', 'r') as dataset:
    applications = [line.strip() for line in dataset.readlines()]

logging.info(f"Loaded applications from dataset")

for application in applications:
    try:
        output_name = application.replace(".", "_")

        logging.info(f"\n\n–––––––––––––––––––––––– {application} ––––––––––––––––––––––––\n\n")

        # Path to your bash script and arguments
        bash_script_path = './get-ipa/get-ipa'
        arguments = [host, port, application, output_name]

        # Run the bash script with arguments and wait for it to complete
        process = subprocess.run([bash_script_path] + arguments, check=True)

        # Check the return code to determine if the script ran successfully
        if process.returncode == 0:
            logging.info(f"{application} retrieved successfully")
        else:
            logging.info(f"Failed to retrieve {application}")

        def timeout_handler(signum, frame):
            raise "Function execution exceeded time limit"

        # Register the signal function handler
        signal.signal(signal.SIGALRM, timeout_handler)

        # Set the alarm for 600 seconds (10 minutes)
        signal.alarm(600)

        ipa_file = f"./decrypted/{output_name}.ipa"
        disassembly_file = f"./decrypted/{output_name}.txt"
        Disassembler().extract_disassembly(input_file = ipa_file, output_file = disassembly_file)

        # Reset the alarm
        signal.alarm(0)

        # Join disassembly and decrypted .ipa in zip
        output_zip = f"./decrypted/{output_name}.zip"
        with zipfile.ZipFile(output_zip, 'w') as zipf:
            zipf.write(ipa_file, arcname=os.path.basename(ipa_file))
            zipf.write(disassembly_file, arcname=os.path.basename(disassembly_file))

        os.remove(ipa_file)
        os.remove(disassembly_file)

        # Open the file in binary mode and send the POST request
        with open(output_zip, 'rb') as file:
            files = {'file': file}
            response = requests.post('http://95.168.166.236:8000/upload', files=files)

        # Print the response from the server
        print(response.status_code)
        print(response.text)
    except:
        logging.info(f"FAILED: Could not get application: {application}")