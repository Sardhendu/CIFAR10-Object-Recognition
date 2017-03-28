"""
About Me: The Input unzipped folder for CIFAR10 has the images of all the labels together.
This module just segregates them and puts them into seperate folder.
So ultimately the output of this module will be folders with images each representing one label of the CIFAR10 data set.
"""

import os
import csv
import shutil
from os import listdir
from os.path import isfile, join

if __name__=='__main__':
    # directory containing images folder and labels file
    # working_dir_path = os.path.dirname(__file__)
    working_dir_path = 'E:/anil/IIT Sop/Term02/CSP571/Project/data/' # change this to where the images are stored

    images_folder_name = 'train'
    labels_file_name = 'trainLabels.csv'
    target_labels = {'cat', 'airplane'}

    images_path = os.path.join(working_dir_path, images_folder_name)
    files_in_images_path = set([f for f in listdir(images_path) if isfile(join(images_path, f))])

    # group file ids by label
    files_by_label = {}
    with open(os.path.join(working_dir_path, labels_file_name)) as fh:
        csv_reader = csv.reader(fh)
        for (file_id, label) in csv_reader:
            files_by_label.setdefault(label,[]).append(file_id+'.png')

    for label, file_names in files_by_label.iteritems():
        if label in target_labels:
            destination_folder = join(working_dir_path, images_folder_name + '_' + label) # change folder if required
            try:
                os.stat(destination_folder)
            except:
                os.mkdir(destination_folder)

            for file_name in files_in_images_path.intersection(set(file_names)):
                # shutil.copy(join(images_path, file_name), destination_folder)
                shutil.move(join(images_path, file_name), destination_folder)
