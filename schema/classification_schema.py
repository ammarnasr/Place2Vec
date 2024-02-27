import json
import os

data_dir = './schema/'

def extract_first_level_codes(schema):
    lines = schema.split('\n')
    codes = []
    indices = []
    for i in range(len(lines)-2):
        line = lines[i]
        if line == '':
            target_line = lines[i+1]
            after_target_line = lines[i+2]
            if target_line[0:2].isdigit() and target_line[2] ==  ' ':
                if after_target_line[0:2].isdigit() and after_target_line[2] == ' ':
                    if target_line not in codes:
                        codes.append(target_line)
                        indices.append(i+1)
    return codes, indices

def extract_second_level_codes(schema):
    lines = schema.split('\n')
    codes = []
    indices = []
    for i in range(len(lines)-1):
        line = lines[i]
        if line[0:2].isdigit() and line[2] ==  ' ':
            target_line = lines[i]
            after_target_line = lines[i+1]
            if after_target_line[0:4].isdigit() and after_target_line[4] == ' ':
                if target_line not in codes:
                    codes.append(target_line)
                    indices.append(i)
    return codes, indices

def extract_third_level_codes(schema):
    lines = schema.split('\n')
    codes = []
    for i in range(len(lines)):
        line = lines[i]
        if line[0:4].isdigit() and line[4] ==  ' ':
            digits_in_line = [d for d in line if d.isdigit()]
            if len(digits_in_line) == 4:
                codes.append(line)
            if len(digits_in_line) == 8:
                seconde_code_digits = ''.join(digits_in_line[4:8])
                first_code = line.split(seconde_code_digits)[0]
                second_code = seconde_code_digits + line.split(seconde_code_digits)[1]
                codes.append(first_code)
                codes.append(second_code)
    return list(set(codes))

def extract_codes(schema):
    lines = schema.split('\n')
    codes = {}
    first_level_codes, indices = extract_first_level_codes(schema)
    indices.append(len(lines))
    for i in range(len(indices)-1):
        start, end = indices[i], indices[i+1]
        first_level_code = lines[start]
        second_level_lines = lines[start:end]
        second_level_codes, indices2 = extract_second_level_codes('\n'.join(second_level_lines))
        indices2 = [start + index for index in indices2]
        indices2.append(end)
        second_level_codes = {}
        for j in range(len(indices2)-1):
            start2, end2 = indices2[j], indices2[j+1]
            second_level_code = lines[start2]
            third_level_lines = lines[start2:end2]
            third_level_codes = extract_third_level_codes('\n'.join(third_level_lines))
            second_level_codes[second_level_code] = third_level_codes
        codes[first_level_code] = second_level_codes
    return codes


def get_classification_schema(schema_path = f'{data_dir}pois_schema.txt'):
    if os.path.exists(f'{data_dir}classification_schema.json'):
        with open(f'{data_dir}classification_schema.json', 'r') as file:
            codes = json.load(file)
        print('Classification schema loaded from file: classification_schema.json')
    else:
        with open(schema_path, 'r') as file:
            schema_txt = file.read()
        codes = extract_codes(schema_txt)
        print(f'Classification schema extracted from file: {schema_path}')
        with open(f'{data_dir}classification_schema.json', 'w') as file:
            json.dump(codes, file, indent=4)
        print(f'Classification schema saved to file: classification_schema.json')

    first_level_count = len(codes)
    second_level_count = sum([len(value) for value in codes.values()])
    third_level_count = sum([len(value2) for value in codes.values() for value2 in value.values()])

    print(f'First level count: {first_level_count}')
    print(f'Second level count: {second_level_count}')
    print(f'Third level count: {third_level_count}')

    return codes

if __name__ == '__main__':
    get_classification_schema()