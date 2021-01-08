from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, utils
from PIL import Image
import pandas as pd
import numpy as np
import os

class MyDataset(Dataset):
    def __init__(self, path_dir, _dir):
        self._dir = _dir
        if _dir == "train_QB":
            self.path_dir = os.path.join(path_dir, _dir)

            self.ms_dir = os.path.join(self.path_dir, "MS")
            self.nir_dir = os.path.join(self.path_dir, "NIR")
            self.pan_dir = os.path.join(self.path_dir, "PAN")
            self.nir_gt_dir = os.path.join(self.path_dir, "NIR_GT")
            self.ms_gt_dir = os.path.join(self.path_dir, "MS_GT")

            self.ms_images = os.listdir(self.ms_dir)
            self.nir_images = os.listdir(self.nir_dir)
            self.pan_images = os.listdir(self.pan_dir)
            self.nir_gt_images = os.listdir(self.nir_gt_dir)
            self.ms_gt_images = os.listdir(self.ms_gt_dir)

        if _dir == "test_QB":
            self.path_dir = os.path.join(path_dir, _dir)

            self.ms_dir = os.path.join(self.path_dir, "MS_test")
            self.nir_dir = os.path.join(self.path_dir, "NIR_test")
            self.pan_dir = os.path.join(self.path_dir, "PAN_test")
            self.nir_gt_dir = os.path.join(self.path_dir, "NIR_test_GT")
            self.ms_gt_dir = os.path.join(self.path_dir, "MS_test_GT")

            self.ms_images = os.listdir(self.ms_dir)
            self.nir_images = os.listdir(self.nir_dir)
            self.pan_images = os.listdir(self.pan_dir)
            self.nir_gt_images = os.listdir(self.nir_gt_dir)
            self.ms_gt_images = os.listdir(self.ms_gt_dir)


    def __getitem__(self, index):
        if self._dir == "train_QB":
            ms_image_index = self.ms_images[index]
            nir_image_index = self.nir_images[index]
            pan_image_index = self.pan_images[index]
            nir_gt_index = self.nir_gt_images[index]
            ms_gt_index = self.ms_gt_images[index]

            ms_img_path = os.path.join(self.ms_dir, ms_image_index)
            nir_img_path = os.path.join(self.nir_dir, nir_image_index)
            pan_img_path = os.path.join(self.pan_dir, pan_image_index)
            nir_gt_path = os.path.join(self.nir_gt_dir, nir_gt_index)
            ms_gt_path = os.path.join(self.ms_gt_dir, ms_gt_index)

            ms_img = transforms.ToTensor()(Image.open(ms_img_path))
            nir_img = transforms.ToTensor()(Image.open(nir_img_path))
            pan_img = transforms.ToTensor()(Image.open(pan_img_path))
            nir_gt_img = transforms.ToTensor()(Image.open(nir_gt_path))
            ms_gt_img = transforms.ToTensor()(Image.open(ms_gt_path))

            ms_nir = torch.cat([ms_img, nir_img])
            ms_nir_gt = torch.cat([ms_gt_img, nir_gt_img])

            return pan_img, ms_nir, ms_nir_gt

        if self._dir == "test_QB":
            ms_image_index = self.ms_images[index]
            nir_image_index = self.nir_images[index]
            pan_image_index = self.pan_images[index]
            nir_gt_index = self.nir_gt_images[index]
            ms_gt_index = self.ms_gt_images[index]

            ms_img_path = os.path.join(self.ms_dir, ms_image_index)
            nir_img_path = os.path.join(self.nir_dir, nir_image_index)
            pan_img_path = os.path.join(self.pan_dir, pan_image_index)
            nir_gt_path = os.path.join(self.nir_gt_dir, nir_gt_index)
            ms_gt_path = os.path.join(self.ms_gt_dir, ms_gt_index)

            ms_img = transforms.ToTensor()(Image.open(ms_img_path))
            nir_img = transforms.ToTensor()(Image.open(nir_img_path))
            pan_img = transforms.ToTensor()(Image.open(pan_img_path))
            nir_gt_img = transforms.ToTensor()(Image.open(nir_gt_path))
            ms_gt_img = transforms.ToTensor()(Image.open(ms_gt_path))

            ms_nir = torch.cat([ms_img, nir_img])
            ms_nir_gt = torch.cat([ms_gt_img, nir_gt_img])

            return pan_img, ms_nir, pan_image_index, ms_nir_gt

    def __len__(self):
        return len(self.pan_images)
