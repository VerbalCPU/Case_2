import sys
import re

'''
Note: This script cleans up ill-formatted columns in the 'page_{month}.csv' files.
Run this script from the location of the corrupted file.
'''
file_name = sys.argv[1]
fixed_dir_path = '../data_fixed/'
fixed_file_name = fixed_dir_path + file_name.split('.')[0] + '_fixed' + '.' + file_name.split('.')[1]

new_lines = []
counter = 0
sep_correct_amount = 10
with open(file_name, 'r', encoding='iso-8859-1') as f:
    for line in f:
        sep_count = line.count(',')
        if sep_count > 10:
            for i in range(sep_count - sep_correct_amount):
                line = re.sub(r'^(.*?(,.*?){2}),', r'\1', line)

        new_lines.append(line)
        if counter % 10000 == 0:
            print('Progress: {0} lines processed'.format(counter))
        counter = counter + 1

with open(fixed_file_name, 'w') as f:
    print('Writing to fixed file...')
    for line in new_lines:
        f.write(line)
