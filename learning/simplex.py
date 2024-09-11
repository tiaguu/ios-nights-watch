import logging
import argparse
import os
import json

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
    parser.add_argument("goodware_opcodes_folder", type=str, help="The folder where goodware opcodes files are stored")
    parser.add_argument("malware_opcodes_folder", type=str, help="The folder where goodware opcodes files are stored")

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
    
    all_opcodes = [
        # Unconditional Branches
        'b', 'bl', 'bx', 'blx',
        
        # Conditional Branches
        'b.eq', 'b.ne', 'b.lt', 'b.gt', 'b.le', 'b.ge', 'b.hi', 'b.lo', 'b.pl', 
        'b.mi', 'b.vs', 'b.vc', 'b.cs', 'b.cc', 'b.al', 'b.nv',
        
        # Compare and Branch
        'cbz', 'cbnz',
        
        # Test and Branch
        'tbz', 'tbnz',
        
        # Return Instructions
        'ret', 'eret',
        
        # Exception Generation
        'svc', 'hvc', 'smc', 'brk', 'hlt',
        
        # Indirect Branching
        'br', 'blr', 'braa', 'brab', 'retab',
        
        # Hints
        'nop', 'yield', 'wfe', 'wfi', 'sev', 'sevl', 'isb', 'dmb', 'dsb'
    ]
    
    goodware_opcodes_dir = os.listdir(goodware_opcodes_folder)
    goodware_opcodes_files = sorted(goodware_opcodes_dir, key=lambda x: os.path.getsize(os.path.join(goodware_opcodes_folder, x)))
    for file in goodware_opcodes_files:
        logging.info(f'Processing file: {file}')
        opcodes_count = {}
        for opcode in all_opcodes:
            opcodes_count[opcode] = 0

        application, extension = os.path.splitext(os.path.basename(file))
        logging.info(f'Application name: {application}')

        nr_lines = get_number_lines_file(application)
        logging.info(f'Number of lines: {nr_lines}')

        with open(f'{goodware_opcodes_folder}/{file}', 'r') as file:
            content = file.read()
            for line in content:
                opcode = line.strip()
                if opcode in all_opcodes:
                    opcodes_count[opcode] += 1

        # for key in opcodes_count.keys():
        #     opcodes_count[key] = round(opcodes_count[key] / nr_lines, 2)

        logging.info(opcodes_count)


def get_number_lines_file(application):
    with open('file_lines.json', 'r') as file_stats:
        stats = json.load(file_stats)
        return stats[application]['lines']
    
if __name__ == "__main__":
    main()