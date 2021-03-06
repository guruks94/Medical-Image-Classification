import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import torch
import random
import copy
import csv
from PIL import Image, ImageFile
import pydicom as dicom
ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    # Initialize isvindrcxr to False
    isvindrcxr = False

    if normalize.lower() == "imagenet":
        normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
        normalize = transforms.Normalize(
            [0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "vindr-cxr":
        isvindrcxr = True  # Set isvindrcxr to True for VinDr-CXR dataset
        normalize = transforms.Normalize(
            [0.4831, 0.4542, 0.4044], [0.2281, 0.2231, 0.2241])
    elif normalize.lower() == "none":
        normalize = None
    else:
        print("mean and std for [{}] dataset do not exist!".format(normalize))
        exit(-1)
    if mode == "train":
        transformations_list.append(transforms.RandomResizedCrop(crop_size))
        transformations_list.append(transforms.RandomHorizontalFlip())
        transformations_list.append(transforms.RandomRotation(7))
        if isvindrcxr:
            transformations_list.append(
                transforms.ColorJitter(brightness=0.5, contrast=0.5))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
            transformations_list.append(normalize)
    elif mode == "valid":
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
            transformations_list.append(normalize)
    elif mode == "test":
        if test_augment:
            transformations_list.append(transforms.Resize((resize, resize)))
            transformations_list.append(transforms.TenCrop(crop_size))
            transformations_list.append(
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            if normalize is not None:
                transformations_list.append(transforms.Lambda(
                    lambda crops: torch.stack([normalize(crop) for crop in crops])))
        else:
            transformations_list.append(transforms.Resize((resize, resize)))
            transformations_list.append(transforms.CenterCrop(crop_size))
            transformations_list.append(transforms.ToTensor())
            if normalize is not None:
                transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence


class ChestXray14Dataset(Dataset):

    def __init__(self, images_path, file_path, augment, num_class=14, annotaion_percent=100):

        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as fileDescriptor:
            line = True

            while line:
                line = fileDescriptor.readline()

                if line:
                    lineItems = line.split()

                    imagePath = os.path.join(images_path, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [int(i) for i in imageLabel]

                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        indexes = np.arange(len(self.img_list))
        if annotaion_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotaion_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list, _img_label = copy.deepcopy(
                self.img_list), copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []

            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def __getitem__(self, index):

        imagePath = self.img_list[index]

        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])

        if self.augment != None:
            imageData = self.augment(imageData)

        return imageData, imageLabel

    def __len__(self):

        return len(self.img_list)

# ---------------------------------------------PadChest-----------------------------------------------------


class PadChestDataset(Dataset):

    def __init__(self, images_path, file_path, augment, num_class=19, annotaion_percent=100):

        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as fileDescriptor:
            line = True

            while line:
                line = fileDescriptor.readline()

                if line:
                    lineItems = line.split(',')

                    imagePath = os.path.join(
                        images_path, lineItems[0], lineItems[1])
                    imageLabel = lineItems[2:num_class + 2]
                    imageLabel = [int(i) for i in imageLabel]

                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        indexes = np.arange(len(self.img_list))
        if annotaion_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotaion_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list, _img_label = copy.deepcopy(
                self.img_list), copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []

            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def __getitem__(self, index):

        imagePath = self.img_list[index]

        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])

        if self.augment != None:
            imageData = self.augment(imageData)

        return imageData, imageLabel

    def __len__(self):

        return len(self.img_list)

# ---------------------------------------------VinDR CXR----------------------------------------------------


class VinDrCXR(Dataset):

    def __init__(self, images_path, file_path, augment, num_class=6, annotaion_percent=100):

        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as fileDescriptor:
            line = True

            while line:
                line = fileDescriptor.readline()

                if line:
                    lineItems = line.split(',')

                    imagePath = os.path.join(images_path, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [int(i) for i in imageLabel]

                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        indexes = np.arange(len(self.img_list))
        if annotaion_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotaion_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list, _img_label = copy.deepcopy(
                self.img_list), copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []

            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def load_dcm_image(self, imagePath):
        # Read dicom file.
        dcm_file = dicom.dcmread(imagePath)

        # Bits Stored' value should match the sample bit depth of the JPEG2000 pixel (16 bit) data in order to get the correct pixel data.

        if dcm_file.BitsStored in (10, 12):
            dcm_file.BitsStored = 16

        raw_image = dcm_file.pixel_array

        # Normalize pixels to be in [0, 255].
        rescaled_image = cv2.convertScaleAbs(dcm_file.pixel_array,
                                             alpha=(255.0/dcm_file.pixel_array.max()))

        # Correct image inversion.
        if dcm_file.PhotometricInterpretation == "MONOCHROME1":
            rescaled_image = cv2.bitwise_not(rescaled_image)

        # Perform histogram equalization if the input is original dicom file.
        if dcm_file.pixel_array.max() > 255:
            adjusted_image = cv2.equalizeHist(rescaled_image)
        else:
            adjusted_image = rescaled_image

        image = Image.fromarray(adjusted_image)
        imageData = image.convert('RGB')

        return imageData

    def __getitem__(self, index):

        imagePath = self.img_list[index]
        #imageData = Image.open(imagePath).convert('RGB')
        imageData = self.load_dcm_image(imagePath)
        imageLabel = torch.FloatTensor(self.img_label[index])

        if self.augment != None:
            imageData = self.augment(imageData)

        return imageData, imageLabel

    def __len__(self):

        return len(self.img_list)

# ---------------------------------------------Downstream CheXpert------------------------------------------


class CheXpertDataset(Dataset):

    def __init__(self, images_path, file_path, augment, num_class=14,
                 uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100):

        self.img_list = []
        self.img_label = []
        self.augment = augment
        assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
        self.uncertain_label = uncertain_label

        with open(file_path, "r") as fileDescriptor:
            csvReader = csv.reader(fileDescriptor)
            next(csvReader, None)
            for line in csvReader:
                imagePath = os.path.join(images_path, line[0])
                label = line[5:]
                for i in range(num_class):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == 0:
                            label[i] = 0
                        elif a == -1:  # uncertain label
                            label[i] = -1
                    else:
                        label[i] = unknown_label  # unknown label

                self.img_list.append(imagePath)
                imageLabel = [int(i) for i in label]
                self.img_label.append(imageLabel)

        indexes = np.arange(len(self.img_list))
        if annotation_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotation_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list, _img_label = copy.deepcopy(
                self.img_list), copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []

            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def __getitem__(self, index):

        imagePath = self.img_list[index]

        imageData = Image.open(imagePath).convert('RGB')

        label = []
        for l in self.img_label[index]:
            if l == -1:
                if self.uncertain_label == "Ones":
                    label.append(1)
                elif self.uncertain_label == "Zeros":
                    label.append(0)
                elif self.uncertain_label == "LSR-Ones":
                    label.append(random.uniform(0.55, 0.85))
                elif self.uncertain_label == "LSR-Zeros":
                    label.append(random.uniform(0, 0.3))
            else:
                label.append(l)
        imageLabel = torch.FloatTensor(label)

        if self.augment != None:
            imageData = self.augment(imageData)

        return imageData, imageLabel

    def __len__(self):

        return len(self.img_list)
