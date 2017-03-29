"""
About Me: The Input unzipped folder for CIFAR10 has the images of all the labels together.
This module just segregates them and puts them into seperate folder.
So ultimately the output of this module will be folders with images each representing one label of the CIFAR10 data set.
"""

import csv
from shutil import copy, move
from os import listdir, mkdir
from os.path import isfile, join, dirname, exists
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] : %(levelname)s : %(message)s")
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # directory containing images folder and labels file
    working_dir_path = join(dirname(__file__), 'data')  # change this to where the images and labels are stored
    # working_dir_path = 'E:/anil/IIT Sop/Term02/CSP571/Project/data/'  # change this to where the images are stored

    images_folder_name = 'train'
    labels_file_name = 'trainLabels.csv'
    target_labels = {'cat', 'airplane'}
    images_count_limit = 500

    # get all files in the image folder
    images_path = join(working_dir_path, images_folder_name)
    files_in_images_path = set([f for f in listdir(images_path) if isfile(join(images_path, f))])
    logger.info('%s files found in image source folder' % (len(files_in_images_path)))

    # group file ids by label
    files_by_label = {}
    with open(join(working_dir_path, labels_file_name)) as fh:
        csv_reader = csv.reader(fh)
        for (file_id, label) in csv_reader:
            files_by_label.setdefault(label, set()).add(file_id + '.png')
    logger.info('Found %s file id and label pairs in labels file' % (len(files_by_label)))

    image_counter = {}
    destination_folders = {}

    # copy or move the files to respective label folder
    for label, file_names in files_by_label.iteritems():
        if label in target_labels:

            if label not in destination_folders:
                destination_folders[label] = join(working_dir_path,
                                                  images_folder_name + '_' + label)  # change destination folder if required
                if not exists(destination_folders[label]):
                    mkdir(destination_folders[label])

            for file_name in files_in_images_path & file_names:
                try:
                    image_counter[label] = image_counter.get(label, 0) + 1
                    copy(join(images_path, file_name), destination_folders[label])
                    # move(join(images_path, file_name), destination_folders[label])
                except:
                    pass

                if image_counter.get(label, 0) >= images_count_limit:
                    break
                if (sum(image_counter.values()) % (images_count_limit*len(target_labels) / 5)) == 0:
                    logger.info('%s files copied to label folders' % (sum(image_counter.values())))
    logger.info('%s files copied to label folders' % (sum(image_counter.values())))