import argparse
import os
import zipfile
import subprocess
from disassemble import Disassembler
import signal
import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("goodware_folder", type=str, help="Path to the goodware folder")
    parser.add_argument("dylibs_folder", type=str, help="Path to the dylibs folder")
    parser.add_argument("malware_folder", type=str, help="Path where injected files will be placed")
    
    args = parser.parse_args()
    
    goodware_folder = args.goodware_folder
    dylibs_folder = args.dylibs_folder
    malware_folder = args.malware_folder

    if not os.path.isdir(goodware_folder):
        print(f"Error: Path '{goodware_folder}' is not a valid directory.")
        return 

    if not os.path.isdir(dylibs_folder):
        print(f"Error: Path '{dylibs_folder}' is not a valid directory.")
        return
    
    if not os.path.exists(malware_folder):
        os.makedirs(malware_folder)

    goodware_files = []
    for goodware_file in os.listdir(goodware_folder):
        if len(goodware_files) < 136:
            root, extension = os.path.splitext(goodware_file)
            if extension == '.zip':
                goodware_files.append(goodware_file)

    dylib_files = []
    for dylib_file in os.listdir(dylibs_folder):
        root, extension = os.path.splitext(dylib_file)
        if extension == '.dylib':
            dylib_files.append(dylib_file)

    for index in range(len(goodware_files)):
        assigned_dylib_index = index % len(dylib_files)
        dylib = dylib_files[assigned_dylib_index]
        dylib_path = os.path.join(dylibs_folder, dylib)

        goodware_zip = goodware_files[index]
        goodware_zip_path = os.path.join(goodware_folder, goodware_zip)

        goodware_name, extension = os.path.splitext(goodware_zip)
        goodware_unzip = os.path.join(malware_folder, goodware_name)

        dylib_name, extension = os.path.splitext(dylib)

        malware_name = f'{goodware_name}_{dylib_name}'
        malware_unzip = os.path.join(malware_folder, malware_name)

        with zipfile.ZipFile(goodware_zip_path, 'r') as zip_ref:
            zip_ref.extractall(goodware_unzip)

            for goodware_unzipped_file in os.listdir(goodware_unzip):
                root, extension = os.path.splitext(goodware_unzipped_file)
                if extension == '.txt':
                    goodware_txt_path = os.path.join(goodware_unzip, goodware_unzipped_file)
                    os.remove(goodware_txt_path)
                if extension == '.ipa':
                    goodware_ipa_path = os.path.join(goodware_unzip, goodware_unzipped_file)
                    malware_ipa_path = os.path.join(goodware_unzip, f'{malware_name}.ipa')

            subprocess.check_output(['./inject/inject', goodware_ipa_path, '-d', dylib_path, '--ipa'])

            # Set the alarm for 600 seconds (10 minutes)
            signal.alarm(600)

            logging.info(f"Disassembling {malware_name}")

            malware_txt_path = os.path.join(goodware_unzip, f'{malware_name}.txt')

            Disassembler().extract_disassembly(input_file = malware_ipa_path, output_file = malware_txt_path)

            logging.info(f"{malware_name} disassembled with success")

            # Reset the alarm
            signal.alarm(0)

            # Join disassembly and decrypted .ipa in zip
            malware_zip = f'{malware_unzip}.zip'
            with zipfile.ZipFile(malware_zip, 'w') as zipf:
                zipf.write(malware_ipa_path, arcname=os.path.basename(malware_ipa_path))
                zipf.write(malware_txt_path, arcname=os.path.basename(malware_txt_path))

            os.remove(malware_ipa_path)
            os.remove(malware_txt_path)

            


# ./inject testiOS/app.ipa -d  @executable_path/testiOS/libinjectiOS.dylib --ipa


if __name__ == "__main__":
    main()