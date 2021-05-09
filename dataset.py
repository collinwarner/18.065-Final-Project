import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import read_image
import requests
import shutil
import json

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        print(self.img_labels['breed'])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 8]), str(self.img_labels.iloc[idx, 1]) +".jpg")
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 8]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample

def parse_to_dictionary(i):
    list_of_string_dicts_improper = dataset.img_labels.iloc[i, 9].strip("[]").replace("'", '"').split("},")
    list_of_string_dicts_improper[0] = list_of_string_dicts_improper[0] + "}"
    #list_of_string_dicts_improper[1] = list_of_string_dicts_improper[1].strip()+"}"
    dict1 = json.loads(list_of_string_dicts_improper[0])
    #dict2 = json.loads(list_of_string_dicts_improper[1])
    return dict1

def check_which_images_are_missing(i, dataset):
    """Checks which images still have not been dowloaded to the appropriate folder

    Args:
        i (int): index into csv file
        dataset (CustomDataset): CustomDataSet object
    """
    filename = "c:\\Users\\Collin\\Documents\\MIT\\Years\\Junior\\Spring\\18.065\\Final Project\\data\\images\\"+str(dataset.img_labels.iloc[i, 8]) +"\\" + str(dataset.img_labels.iloc[i, 1]) + ".jpg"
    if not os.path.exists(filename):
        print("This cat is still missing: " + str(dataset.img_labels.iloc[i, 8]) +"\\" + str(dataset.img_labels.iloc[i, 1]) + ".jpg")
    


def download_image(i, dataset):
    """
        downloads the images from the links in the csv and stores in appropriate folders
        will not download already found images
    """
    filename = "c:\\Users\\Collin\\Documents\\MIT\\Years\\Junior\\Spring\\18.065\\Final Project\\data\\images\\"+str(dataset.img_labels.iloc[i, 8]) +"\\" + str(dataset.img_labels.iloc[i, 1]) + ".jpg"
    # Check if the image was retrieved successfully
    if not os.path.exists(filename):
        
        image_url_list = dataset.img_labels.iloc[i, 10].replace("'", '').strip('][').split(', ')
        found_image = False
        for image_url in image_url_list:
            r = requests.get(image_url, stream = True)

            if r.status_code == 200:
                # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
                r.raw.decode_content = True

                # Open a local file with wb ( write binary ) permission.
                with open(filename,'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                
                print('Image sucessfully Downloaded: ',filename)
                found_image = True
                break
        if not found_image:
            print('Image Couldn\'t be retreived', filename)
        


if __name__=="__main__":
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    print(THIS_FOLDER)
    my_file = os.path.join(THIS_FOLDER, 'data\\data\\cats.csv')
    annotations_file = "c:\\Users\\Collin\\Documents\\MIT\\Years\\Junior\\Spring\\18.065\\Final Project\\data\\data\\cats.csv"
    img_dir = "c:\\Users\\Collin\\Documents\\MIT\\Years\\Junior\\Spring\\18.065\\Final Project\\data\\images"
    dataset = CustomImageDataset(annotations_file, img_dir)
    #print(str(dataset.img_labels.iloc[6752, 1]))
    #remaining = [i for i in range(len(dataset))]
    #list(map(lambda i : download_image(i, dataset), remaining))
    #list(map(lambda i : check_which_images_are_missing(i, dataset), remaining))
    cat_counts = {}
    for sample in dataset:
        label = sample["label"]
        cat_counts.setdefault(label, 0)
        cat_counts[label] += 1
    
    total = 0
    for cat in cat_counts:
        total += cat_counts[cat]
    print(cat_counts)
    print(total)
    
        
    