import csv
import argparse
import pandas as pd
import os
from os.path import join
import re



# from helpers import process


def process(q):
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    
    return q 

def save_to_csv(list1, list2, list3, file_name):
    # Zip the lists together
    combined_list = list(zip(list1, list2, list3))

    # Open a new file to write to
    with open(file_name, "w") as f:
        writer = csv.writer(f)
        # Write the combined list to the file
        writer.writerows(combined_list)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', default='data/train.csv', help='Path to input file')
    # args = parser.parse_args()

    # # Split the input path into a directory, filename, and extension
    # dirname, filename = os.path.split(args.data_dir)
    # name, ext = os.path.splitext(filename)

    # # Check if the filename is "train" or "test"
    # if name == 'train':

    #     data = pd.read_csv(args.data_dir)
    #     print(data.head)
    #     data = data.dropna()

    #     data['question1'] = data['question1'].apply(process)
    #     data['question2'] = data['question2'].apply(process)
    #     label = data['is_duplicate']

    #     # Create the output path using the input filename
    #     output_path = os.path.join(dirname, name + '_processed' + ext)

    #     save_to_csv(data['question1'], data['question2'], label, output_path)
    # elif name == 'test':
    #     data = pd.read_csv(args.data_dir)
    #     print(data.shape)
    #     data = data.dropna()

    #     data['question1'] = data['question1'].apply(process)
    #     data['question2'] = data['question2'].apply(process)
    #     label = data['is_duplicate']

    #     # Create the output path using the input filename
    #     output_path = os.path.join(dirname, name + '_processed' + ext)

    #     save_to_csv(data['question1'], data['question2'], label, output_path)
        print("somethin")


