import argparse
import os
import zipfile
import subprocess
from disassemble import Disassembler
import signal
import logging
import requests
import shutil

def main():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers= [stream_handler]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("server_ip", type=str, help="The server IP address")
    parser.add_argument("dylibs_folder", type=str, help="Path to the dylibs folder")
    parser.add_argument("malware_folder", type=str, help="Path where injected files will be placed")
    
    args = parser.parse_args()

    server_ip = args.server_ip    
    dylibs_folder = args.dylibs_folder
    malware_folder = args.malware_folder

    if not os.path.isdir(dylibs_folder):
        print(f"Error: Path '{dylibs_folder}' is not a valid directory.")
        return
    
    if not os.path.exists(malware_folder):
        os.makedirs(malware_folder)

    dylib_files = []
    for dylib_file in os.listdir(dylibs_folder):
        root, extension = os.path.splitext(dylib_file)
        if extension == '.dylib':
            dylib_files.append(dylib_file)

    malware_instances_number = 136
    for number in range(malware_instances_number):
        logging.info(f"––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")
        logging.info(f"–––––––––––––––––––––––––––––––––––– {number} –––––––––––––––––––––––––––––––––––––––")
        logging.info(f"––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")

        logging.info("Downloading goodware")
        download_url = f'http://{server_ip}:8000/download/{number}'
        with requests.get(download_url, stream=True) as response:
            goodware_zip_path = os.path.join(malware_folder, f'{number}.zip')
            with open(goodware_zip_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        logging.info("Downloaded goodware with success")

        logging.info("Obtaining goodware .ipa")

        assigned_dylib_index = number % len(dylib_files) if number != 0 else 0
        dylib = dylib_files[assigned_dylib_index]
        dylib_path = os.path.join(dylibs_folder, dylib)

        goodware_unzip = os.path.join(malware_folder, str(number))

        with zipfile.ZipFile(goodware_zip_path, 'r') as zip_ref:
            zip_ref.extractall(goodware_unzip)

            for goodware_unzipped_file in os.listdir(goodware_unzip):
                root, extension = os.path.splitext(goodware_unzipped_file)
                if extension == '.txt':
                    goodware_txt_path = os.path.join(goodware_unzip, goodware_unzipped_file)
                    os.remove(goodware_txt_path)
                if extension == '.ipa':
                    goodware_name = root
                    goodware_ipa_path = os.path.join(goodware_unzip, goodware_unzipped_file)

            dylib_name, extension = os.path.splitext(dylib)

            malware_name = f'{goodware_name}_{dylib_name}'
            malware_unzip = os.path.join(malware_folder, malware_name)
            new_goodware_ipa_path = os.path.join(malware_unzip, f'{goodware_name}.ipa')
            malware_ipa_path = os.path.join(malware_unzip, f'{malware_name}.ipa')

            os.remove(goodware_zip_path)
            os.rename(goodware_unzip, malware_unzip)
            os.rename(new_goodware_ipa_path, malware_ipa_path)
        
            logging.info("Obtained goodware .ipa")

            logging.info("Injecting malware .dylib into goodware .ipa")

            subprocess.check_output(['./inject/inject', malware_ipa_path, '-d', dylib_path, '--ipa'])

            logging.info("Injected with success")

            def timeout_handler(signum, frame):
                raise "Function execution exceeded time limit"

            # Register the signal function handler
            signal.signal(signal.SIGALRM, timeout_handler)

            # Set the alarm for 600 seconds (10 minutes)
            signal.alarm(600)

            logging.info(f"Disassembling {malware_name}")

            malware_txt_path = os.path.join(malware_unzip, f'{malware_name}.txt')

            Disassembler().extract_disassembly(input_file = malware_ipa_path, output_file = malware_txt_path)

            logging.info(f"{malware_name} disassembled with success")

            # Reset the alarm
            signal.alarm(0)

            # Join disassembly and decrypted .ipa in zip
            malware_zip = f'{malware_unzip}.zip'
            with zipfile.ZipFile(malware_zip, 'w') as zipf:
                zipf.write(malware_ipa_path, arcname=os.path.basename(malware_ipa_path))
                zipf.write(malware_txt_path, arcname=os.path.basename(malware_txt_path))

            shutil.rmtree(malware_unzip)

            logging.info(f"Uploading {malware_name}")

            filename = os.path.basename(malware_zip)
            with open(malware_zip, mode='rb') as file:
                data = {'file': (filename, file, 'application/zip')}

                response = requests.post('http://{server_ip}:8000/upload/malware', files=data)

                if response.status_code == 200:
                    logging.info(f"{malware_name} uploaded with success")    

            os.remove(malware_zip)        
            logging.info("Cleaned up")


if __name__ == "__main__":
    main()