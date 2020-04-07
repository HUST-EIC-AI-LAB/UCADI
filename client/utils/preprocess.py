import numpy as np 
import SimpleITK as sitk 
import nibabel as nib 
import os 
import argparse
from multiprocessing import Pool

def write_image(image, path):
    image_array = sitk.GetImageFromArray(image)
    sitk.WriteImage(image_array, path)


def read_dicom(path):
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(names)

    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    return reader


def read_image(path):
    reader = read_dicom(path)   
    image = reader.Execute()
    # z, y, x
    image_array = sitk.GetArrayFromImage(image)
    return image_array


def preprocess(data_list):
    save_root = "/mnt/data/zxy/dataset/"
    image_name, image_path = data_list
    try:
        image = read_image(image_path)
        z, y, x = image.shape
        if z > 15 and y == 512 and x == 512:
            print("z {}, y {}, z {}".format(z, y, x))
            write_image(image, save_root + "{}.nii.gz".format(str(image_name)))
            print("{} finished".format(image_name))
    except:
        with open("failed_pre_norm.txt","a") as f:
            f.write(image_name + "\n")
        print("{} failed".format(image_name) + "\n")

    
def gen_path(data_dir):   
    image_list = []
    len_thresh = len(data_dir.split("/")) + 4
    for root, dirs, files in os.walk(data_dir):
        len_split = len(root.split("/"))
        print(len_split, root)
        if len_split == len_thresh:
            dir_name = root.split("/")[-4]
            if dir_name == "DICOM" or dir_name == "DICOMA" or dir_name == "DICOMDIS":
                image_name = root.split("/")[-5] + "_" + root.split("/")[-3] + "_"  + root.split("/")[-1] 
                print(image_name, root)
                image_list.append([image_name, root])    
    return image_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ncp')
    parser.add_argument('--input', default='/mnt/data/zxy/ct_data/', type=str, metavar='SAVE',
                        help='directory to save dicom data (default: none)')
    parser.add_argument('--output', default='/mnt/data/zxy/dataset/', type=str, metavar='SAVE',
                        help='directory to save nii.gz data (default: none)')
    
    global args
    args = parser.parse_args()
    save_root = args.output
    if os.path.exists(save_root) == False:
        os.makedirs(save_root)
    raw_data_dir = args.input
    data_lists = gen_path(raw_data_dir)

    if os.path.exists("failed_pre_norm.txt"): os.remove("failed_pre_norm.txt")

    p = Pool(6)
    p.map(preprocess,data_lists)
    p.close()
    p.join()
