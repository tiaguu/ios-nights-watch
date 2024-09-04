import os
import subprocess
import zipfile
import shutil

class Disassembler():
    def extract_disassembly(self, input_file, output_file):
        extracted_path = os.path.splitext(input_file)[0]
        os.makedirs(extracted_path, exist_ok = True)
        with zipfile.ZipFile(input_file, 'r') as ipa_zip:
            ipa_zip.extractall(extracted_path)

        binary_path, library_paths = self.find_binary_and_libraries(extracted_path = extracted_path)
        disassembly = self.disassemble_with_otool(binary_path = binary_path)
        
        for library_path in library_paths:
            disassembled_library = self.disassemble_with_otool(binary_path = library_path)
            disassembly += "\n" + disassembled_library 

        with open(output_file, 'w') as file:
            file.write(disassembly)

        # clean up extracted path
        if os.path.exists(extracted_path):
            shutil.rmtree(extracted_path)

        return disassembly
    
    def find_binary_and_libraries(self, extracted_path):
        binary_path = None
        library_paths = []

        for root, dirs, files in os.walk(extracted_path):
            for file in files:
                if file.endswith('.dylib'):
                    library_paths.append(os.path.join(root, file))
            for directory in dirs:
                if directory.endswith('.app'):
                    app_dir = os.path.join(root, directory)
                    for file in os.listdir(app_dir):
                        file_path = os.path.join(app_dir, file)
                        if file == directory[:-4]:  # Check if file is executable
                            binary_path = file_path

        return binary_path, library_paths

    def disassemble_with_otool(self, binary_path):
        # Execute otool command to disassemble the binary file
        try:
            output = subprocess.check_output(['otool', '-tV', binary_path])
            disassembly = output.decode('utf-8')
            return disassembly
        except subprocess.CalledProcessError as e:
            print("Error:", e)
            return []    