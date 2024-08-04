import re

class Preprocessor():
    def clean_assembly_file(self, input_file):
        with open(input_file, 'r') as file:
            content = file.read()
            return self.clean_assembly(input = content)

    def clean_assembly(self, input):
        final = ''
        lines = input.split('\n')
        for line in lines:
            long_address_pattern = r'^[0-9a-fA-F]{16}$'
            small_address_pattern = r'^[0-9a-fA-F]{8}$'
            half_address_pattern = r'^\s{4}[0-9A-Fa-f]{4}$'
            
            if re.match(long_address_pattern, line.split('\t')[0]): 
                instruction = line.split('\t')
                address = instruction[0]
                operation = instruction[1]
                if len(instruction) > 2:
                    arguments = instruction[2].split(';')[0].rstrip()
            elif re.match(small_address_pattern, line.split('\t')[0]):
                instruction = line.split('\t')
                if re.match(small_address_pattern, instruction[1]) or re.match(half_address_pattern, instruction[1]):
                    first_address = instruction[0]
                    second_address = instruction[1]
                    operation = instruction[2]
                    if len(instruction) > 3:
                        arguments = instruction[3].split('@')[0].split(';')[0].rstrip()
                    print(operation, arguments)
                # if len(instruction) > 3:
                #     first_address = instruction[0]
                #     second_address = instruction[1]
                #     operation = instruction[2]
                #     arguments = instruction[3].split('@')[0].rstrip()
                # else:
                #     if len(instruction[1]) == 8:
                #         if '.' in instruction[1]:
                #             pass
                #         else:
                #             print(instruction)
            else:
                pass

if __name__ == "__main__":
    Preprocessor().clean_assembly_file(input_file = './QuanticApps_Quran.txt')