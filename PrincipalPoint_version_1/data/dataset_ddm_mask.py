"""Defines the fisheye dataset for directional marking point detection."""
import json

import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch

from PrincipalPoint_version_1.util.DDM import gpu_calculate_ddm


# from util.tool import pinhole_direction, get_target, region_of_interest, canny_, exact_lane

def generate_heatmap(size, sigma, cx, cy):
    heatmap = np.zeros((size, size))
    heatmap[cx][cy] = 1
    heatmap = cv.GaussianBlur(heatmap, (sigma, sigma), 0)
    am = np.amax(heatmap)
    heatmap /= am
    # plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    # plt.show()
    return heatmap


class CameraPoseDataset(Dataset):
    """fisheye dataset."""

    def __init__(self, root, data_len):
        super(CameraPoseDataset, self).__init__()
        self.root = root
        # self.temp_len = data_len  # all pictures in BFLR in order
        self.temp_names = []
        self.file_name = []
        self.image_transform = ToTensor()
        for image_index in range(data_len):
            self.temp_names.append((str(image_index).zfill(7)))
            # self.temp_names.append((str(4) + str(image_index).zfill(6)))
            # self.temp_names.append((str(2) + str(image_index).zfill(6)))
            # self.temp_names.append((str(2) + str(image_index).zfill(6)))

    def __getitem__(self, index):
        json_name = 'parameter/' + self.temp_names[index]
        # with open(self.root + json_name + '.json') as file:
        #     image_name = json.load(file)['image_name']
        with open(self.root + json_name + '.json') as file:
            camera_matrix = json.load(file)['center_point']
        image_name = self.root + 'img/' + self.temp_names[index] + '.jpg'
        mask_name = self.root + 'mask/' + self.temp_names[index] + '.jpg'
        origin_image = cv.imread(image_name)
        origin_mask = cv.imread(mask_name)
        input_size = 320
        image = self.image_transform(cv.resize(origin_image, (input_size, input_size)))
        image_mask = self.image_transform(cv.resize(origin_mask, (input_size, input_size)))
        # images.append(self.image_transform(image))
        center_point = [0, camera_matrix[0] / origin_image.shape[1], camera_matrix[1] / origin_image.shape[0], 1, 1]
        center_point = torch.tensor(center_point)
        heatmap_h = gpu_calculate_ddm(input_size / 2, input_size / 2, int(center_point[1] * 160),
                                      int(center_point[2] * 160))
        heatmap_l = gpu_calculate_ddm(input_size / 4, input_size / 4, int(0.5 * 80), int(0.5 * 80))
        # heatmap_h = generate_heatmap(160, 9, int(center_point[1] * 160), int(center_point[2] * 160))
        # heatmap_l = generate_heatmap(80, 9, int(center_point[1] * 80), int(center_point[2] * 80))
        mask = np.zeros((int(input_size / 2), int(input_size / 2)))
        noobj_mask = np.ones((int(input_size / 2), int(input_size / 2)))
        tx = np.zeros((int(input_size / 2), int(input_size / 2)))
        ty = np.zeros((int(input_size / 2), int(input_size / 2)))
        # Convert to position relative to box
        gx = center_point[1] * (input_size / 2)
        gy = center_point[2] * (input_size / 2)

        # Get grid box indices
        gi = int(gx)
        gj = int(gy)

        noobj_mask[gj, gi] = 0
        # Masks
        mask[gj, gi] = 1
        # Coordinates
        tx[gj, gi] = gx - gi
        ty[gj, gi] = gy - gj
        mask = torch.tensor(mask)
        noobj_mask = torch.tensor(noobj_mask)
        tx = torch.tensor(tx)
        ty = torch.tensor(ty)
        # heatmap_h = np.load(self.root + 'RPM_h/' + self.temp_names[index] + '.npy')
        # heatmap_l = np.load(self.root + 'RPM_l/' + self.temp_names[index] + '.npy')
        heatmap_h = torch.Tensor(heatmap_h).unsqueeze(0)
        heatmap_l = torch.Tensor(heatmap_l).unsqueeze(0)

        return image, image_mask, center_point, heatmap_l, heatmap_h, mask, noobj_mask, tx, ty

    def __len__(self):
        return len(self.temp_names)
