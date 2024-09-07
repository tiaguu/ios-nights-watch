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

    def clean_assembly_file(self, input_file):
        with open(input_file, 'r') as file:
            content = file.read()
            return self.clean_assembly(input = content)

    def clean_assembly(self, input):
        final = []
        lines = input.split('\n')
        operation_list = []
        ignore_opcodes = {'.long', '.short', '.byte', 'nop', 'adr', 'adrp', 'tbl', 'tbx', 'dup', 'zip1', 
                          'zip2', 'trn1', 'trn2', 'uzp1', 'uzp2', 'ld1', 'ld2', 'ld3', 
                          'ld4', 'st1', 'st2', 'st3', 'st4', 'rbit', 'rev', 'rev16', 
                          'rev32', 'rev64', 'fmov', 'fcvt', 'fcvtas', 'fcvtn', 'fcvtms', 
                          'fcvtpu', 'fcvtzu', 'fabs', 'fneg', 'sqxtn', 'sqxtn2', 'uqxtn', 
                          'uqxtn2', 'sqxtun', 'sqxtun2', 'addp', 'faddp', 'cmle', 'cmge', 
                          'cmeq', 'cmgt', 'cmhi', 'cmhs', 'cnt', 'aesd', 'aese', 'aesimc',
                          'msr', 'mrs', 'sys', 'tlbi', 'dcps1', 'dcps2', 'dcps3', 'hlt',
                          'ldr', 'str', 'ldp', 'stp', 'ldur', 'stur', 'ldrb', 'strb', 
                          'ldrh', 'strh', 'add', 'sub', 'mul', 'fmul', 'fadd', 'fmla',
                          'fmls', 'fcmla', 'sqdmulh', 'fmin', 'fmax', 'svc', 'brk', 'pacia', 
                          'pacib', 'abs', 'ngc', 'ngcs', 'madd', 'mls', 'movi', 'lsl', 
                          'lsr', 'asr', 'ror', 'clz', 'bic', 'eor', 'orn', 'and', 'orr', 
                          'neg', 'negs', 'ldrsb', 'ldrsw', 'ldxp', 'stxp', 'stllr', 'ldclr', 
                          'ldset', 'fsub', 'fcmp', 'fsqrt', 'fmadd', 'fmsub', 'fnmadd', 
                          'fnmsub', 'frecpe', 'frsqrte', 'ushl', 'ushr', 'uqrshl', 'uqrshrn', 
                          'uqadd', 'uqsub', 'pmull', 'pmull2', 'facgt', 'facge', 'fcsel', 
                          'fccmp', 'sqdmull', 'uqshl', 'uqshrn', 'sqrshl', 'sqrshrn', 'sqshlu',
                          'prfm', 'prfum', 'dmb', 'dsb', 'isb', 'sha256h', 'sha512h', 'sha1c', 
                          'sha1su0', 'crc32'}
        
        # include_opcodes = [
        #     'bl', 'b', 'cbnz', 'cbz', 'tbz', 'tbnz',  # Control flow
        #     # 'ldr', 'str', 'ldp', 'stp', 'ldur', 'stur',  # Memory operations
        #     # 'add', 'sub', 'mul', 'madd', 'msub',  # Arithmetic
        #     # 'and', 'orr', 'eor', 'bic', 'lsl', 'lsr',  # Logical and shifts
        #     # 'mrs', 'msr', 'sys', 'svc',  # System instructions
        #     # 'fmul', 'fadd', 'fsub',  # Optional SIMD
        #     # 'sha256h', 'crc32b'  # Cryptographic instructions (if needed)
        # ]

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

        # include_opcodes = [
        #     'msr', 'mul', 'adc', 'teq', 'ldm', 'orr', 'sbc',
        #     'and', 'mvn', 'stc', 'stm', 'tst', 'bx', 'cmn',
        #     'sub', 'cmp', 'str', 'mla', 'ldr', 'eor', 'b', 'mov'
        # ]
        
        # Prepare a tuple for faster lookup with str.startswith()
        ignore_opcodes_tuple = tuple(ignore_opcodes)
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
                    if operation.startswith(include_opcodes_tuple):
                        instruction_tokenized = []
                        instruction_tokenized.append(operation)
                        if operation not in operation_list:
                            operation_list.append(operation)

                            if operation not in opcodes:
                                opcodes.append(operation)

                        for argument in arguments:
                            instruction_tokenized.append(argument)
                            
                        kept_lines += 1
                        final.append([' '.join(instruction_tokenized)])
            else:
                pass

        return final
