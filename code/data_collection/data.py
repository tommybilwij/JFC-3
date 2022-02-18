import os
import shutil
import zipfile
import argparse
from pathlib2 import Path
import wget

import splitfolders

# Check whether the target directory exists
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return Path(path).resolve(strict=False)

# Download zip file from url link, and unzip it to pvc
def download(source, target, force_clear=False):
    if force_clear and os.path.exists(target):
        print('Removing {}...'.format(target))
        shutil.rmtree(target)

    check_dir(target)

    targt_file = str(Path(target).joinpath('data1.zip'))
    if os.path.exists(targt_file) and not force_clear:
        print('data already exists, skipping download')
        return

    if source.startswith('http'):
        print("Downloading from {} to {}".format(source, target))
        wget.download(source, targt_file)
        print("Done!")
    else:
        print("Copying from {} to {}".format(source, target))
        shutil.copyfile(source, targt_file)

    print('Unzipping {}'.format(targt_file))
    zipr = zipfile.ZipFile(targt_file)
    zipr.extractall(target)
    zipr.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='data cleaning for binary image task')
    parser.add_argument('-b', '--base_path',
                        help='directory to base data', default='../../data')
    parser.add_argument(
        '-z', '--zfile', help='source data zip file', default='../../data1.zip')  # noqa: E501
    parser.add_argument('-f', '--force',
                        help='force clear all data', default=False, action='store_true')  # noqa: E501
    args = parser.parse_args()
    print(args)


    # Initialise path for dataset
    base_path = Path(args.base_path).resolve(strict=False)
    print('Base Path:  {}'.format(base_path))

    # Obtain labelled and nonlabelled dataset zip file from Google Drive Link ID
    print('Acquiring labelled and nonlabelled data...')
    # This will output new_labelled and new_nonlabelled folders in ../data/ directory
    download(args.zfile, str(base_path), args.force)

    # --------------- GROUP 2 CODE (with slight modification on data path START ---------------
    # Load the dataset
    labeled_input_folder = str(base_path)+'/new_labelled' #5-10% of labeled dataset, is stored in /mnt/azure/data/
    output = str(base_path)+'/processed_data' #contains train, test folders for the labeled dataset, will stored in /mnt/azure/data/

    # Create train and test folders inside the processed_data folder. 
    # The train and test folders will contain images and the number of images in the test and train folders are determined by the ratio. 
    # Inside the train and test folders, all the images are seperated and stored into the various categorical folders. 
    splitfolders.ratio(labeled_input_folder, output, seed=15, ratio=(.8, .0, .2)) 
    print('Data Collection component is complete')
    # --------------- GROUP 2 CODE (with slight modification on data path END ---------------
