import os
import re
import pandas as pd
import numpy as np
import jieba

# read all the txt files in the folder and merge them into one txt file
def merge_txt(path):
    files = os.listdir(path)
    files.sort()
    with open('全文.txt', 'w', encoding='utf-8') as f:
        for file in files:
            if not os.path.isdir(file) and os.path.splitext(file)[1] == '.txt':
                print('merge ' + file + ' to merge.txt')
                with open(path + '/' + file, 'r', encoding='utf-8') as f1:
                    for line in f1.readlines():
                        f.writelines(line)
                f.write('\n')
    print('merge done!')



merge_txt('.')