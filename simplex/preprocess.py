import re
from pyparsing import Word, alphanums, alphas, oneOf, Group, delimitedList, nestedExpr
import string
import random
import logging

class Preprocessor():
    def is_address(self, string):
        long_address_pattern = r'^[0-9a-fA-F]{16}$'
        small_address_pattern = r'^[0-9a-fA-F]{8}$'
        half_address_pattern = r'^\s{4}[0-9A-Fa-f]{4}$'
        quarter_address_pattern = r'^\s{6}[0-9A-Fa-f]{2}$'

        return re.match(long_address_pattern, string) \
                or re.match(small_address_pattern, string) \
                or re.match(half_address_pattern, string) \
                or re.match(quarter_address_pattern, string)
    
    def is_subset(self, block1, block2):
        return block2[0] <= block1[0] and block1[1] <= block2[1]

    def assign_priority(self, blocks):
        n = len(blocks)
        priority = [0] * n

        for i in range(n):
            for j in range(n):
                if i != j and self.is_subset(blocks[i][1], blocks[j][1]):
                    priority[i] += 1

        return priority

    def sort_by_priority(self, blocks):
        priority = self.assign_priority(blocks)
        blocks_with_priority = sorted(zip(blocks, priority), key=lambda x: x[1], reverse=True)
        sorted_blocks = [block for block, _ in blocks_with_priority]
        return sorted_blocks
    
    def find_key_by_value(self, my_dict, value):
        for key, val in my_dict.items():
            if val == value:
                return key
        return None 

    def split_ignoring_parentheses(self, instruction):
        if any(char in instruction for char in ['{', '[', '(']):
            blocks = {}
            for character_index in range(len(instruction)):
                character = instruction[character_index]

                if character == '{':
                    remaining_instruction = instruction[character_index + 1:]

                    has_found_bloack = False
                    for second_index in range(len(remaining_instruction)):
                        second_character = remaining_instruction[second_index]

                        if second_character == '}' and not has_found_bloack:
                            block = instruction[character_index:character_index+second_index+2]
                            block_indexes = (character_index, character_index+second_index+1)
                            block_info = (block, block_indexes)
                            blocks[self.generate_random_string(length = second_index + 2)] = block_info
                            has_found_bloack = True

                if character == '[':
                    remaining_instruction = instruction[character_index + 1:]

                    has_found_bloack = False
                    for second_index in range(len(remaining_instruction)):
                        second_character = remaining_instruction[second_index]

                        if second_character == ']' and not has_found_bloack:
                            block = instruction[character_index:character_index+second_index+2]
                            block_indexes = (character_index, character_index+second_index+1)
                            block_info = (block, block_indexes)
                            blocks[self.generate_random_string(length = second_index + 2)] = block_info
                            has_found_bloack = True

                if character == '(':
                    remaining_instruction = instruction[character_index + 1:]

                    has_found_bloack = False
                    for second_index in range(len(remaining_instruction)):
                        second_character = remaining_instruction[second_index]

                        if second_character == ')' and not has_found_bloack:
                            block = instruction[character_index:character_index+second_index+2]
                            block_indexes = (character_index, character_index+second_index+1)
                            block_info = (block, block_indexes)
                            blocks[self.generate_random_string(length = second_index + 2)] = block_info
                            has_found_bloack = True
            
            priority_blocks = self.sort_by_priority(blocks = list(blocks.values()))
            hashed_instruction = instruction
            put_back_blocks = []
            for prioritized_block in priority_blocks:
                key = self.find_key_by_value(blocks, prioritized_block)

                put_back_blocks.append((key, prioritized_block[0]))
                hashed_instruction = hashed_instruction[:prioritized_block[1][0]] + key + hashed_instruction[prioritized_block[1][1]+1:]
            
            split_hashed_instruction = hashed_instruction.split(',')
            for put_back_block in put_back_blocks:
                for hashed_argument_index in range(len(split_hashed_instruction)):
                    if put_back_block[0] in split_hashed_instruction[hashed_argument_index]:
                        split_hashed_instruction[hashed_argument_index] = split_hashed_instruction[hashed_argument_index].replace(put_back_block[0], put_back_block[1])

            for argument_index in range(len(split_hashed_instruction)):
                split_hashed_instruction[argument_index] = split_hashed_instruction[argument_index].strip()

            return split_hashed_instruction
        else:
            return [argument.strip() for argument in instruction.split(',')]
        
    def generate_random_string(self, length):
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choice(characters) for _ in range(length))
        return random_string

    def get_opcodes_file(self, input_file):
        with open(input_file, 'r') as file:
            content = file.read()
            return self.get_opcodes(input = content)

    def get_opcodes(self, input):
        lines = input.split('\n')

        # control_flow_opcodes
        include_opcodes = [
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

        include_opcodes_tuple = tuple(include_opcodes)
        opcodes = []
        total_lines = 0
        kept_lines = 0
        for line in lines:
            total_lines += 1

            instruction = line.split('\t')
            if self.is_address(instruction[0]):
                if len(instruction) > 1:
                    if self.is_address(instruction[1]):
                        first_address = instruction[0]
                        second_address = instruction[1]
                        operation = instruction[2]
                        if len(instruction) > 3:
                            arguments = instruction[3].split('@')[0].split(';')[0].rstrip()
                            arguments = self.split_ignoring_parentheses(instruction = arguments)
                        else:
                            arguments = []
                    else:
                        address = instruction[0]
                        operation = instruction[1]
                        if len(instruction) > 2:
                            arguments = instruction[2].split('@')[0].split(';')[0].rstrip()
                            arguments = self.split_ignoring_parentheses(instruction = arguments)
                        else:
                            arguments = []

                    # Use str.startswith() with the tuple of ignore opcodes
                    if operation in include_opcodes_tuple:
                        opcodes.append(operation)                            
                        kept_lines += 1
            else:
                pass

        logging.info(f'Kept {int((kept_lines / total_lines) * 100)}% of total lines')

        return opcodes