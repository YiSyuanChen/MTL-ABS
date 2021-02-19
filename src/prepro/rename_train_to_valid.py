""" Rename file for meta validation """
import os
import argparse
from os.path import join as pjoin
from utils import str2bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_path", type=str, required=True)
    args = parser.parse_args()

    for dirPath, dirNames, fileNames in os.walk(args.input_path):
        for f in fileNames:
            print(pjoin(dirPath, f))
            if ('train' in f):
                os.rename(pjoin(dirPath, f),
                          pjoin(dirPath, f.replace('train', 'valid')))
