import json
import random
from math import cos, sin

import cv2
import torch
import cv2
import numpy as np

root = "D:/ZJC/data/instrinsic_test_dataset_v1/"
# root = ""
for index in range(0, 1):
    for i in range(0, 1):
        json_name = str(i) + str(index).zfill(6)
        # json_name = "1"
        with open(root + json_name + '.json') as file:
            image_name = json.load(file)['image_name']
        with open(root + json_name + '.json') as file:
            camera_matrix = json.load(file)['camera_matrix']
        with open(root + json_name + '.json') as file:
            distortion = json.load(file)['distortion']
        # with open(root + json_name + '.json') as file:
        #     image_size = json.load(file)['image_size']
        image = cv2.imread(image_name)
        cv2.imwrite("origin_fisheye.jpg", image)
        scale_x, scale_y = image.shape[1] / 1280, image.shape[0] / 720
        # scale_x, scale_y = 1., 1.
        K = np.zeros((3, 3))
        K[0][0], K[1][1], K[0][2], K[1][2], K[2][2] = camera_matrix[0], camera_matrix[4], \
            camera_matrix[2], camera_matrix[5], 1.0
        theta1 = (-30. / 180. * np.pi)
        T_1 = np.array([[1,0,0],
                        [0,cos(theta1),-sin(theta1)],
                        [0,sin(theta1),cos(theta1)], ])
        # offset_min, offset_max = -50, 50
        # offset_x = random.uniform(offset_min, offset_max)
        # offset_y = random.uniform(offset_min, offset_max)
        # Create a single-channel mask image

        # xc = int(K[0][2] + offset_x)
        # yc = int(K[1][2] + offset_y)
        # new_K = K.copy()
        # new_K[0][2], new_K[1][2] = xc, yc
        D = np.array(distortion)
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (image.shape[1], image.shape[0]),
                                                         cv2.CV_16SC2)
        origin_undistorted_img = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite('origin_undistorted_img.jpg', origin_undistorted_img)

        rows, cols = image.shape[:2]
        image_distorted = np.zeros((1, rows * cols, 2), dtype=np.float32)
        image_H_undistorted = np.zeros((rows, cols, 3), dtype=np.uint8)
        origin_distorted = [(u, v) for u in range(rows) for v in range(cols)]
        for u in range(rows):
            for v in range(cols):
                image_distorted[0][u * cols + v][0] = v
                image_distorted[0][u * cols + v][1] = u

        origin_undistorted_points = cv2.fisheye.undistortPoints(image_distorted, K, D, P=K)
        point = origin_undistorted_points[0]
        origin_undistorted_points = origin_undistorted_points[0]
        # Add homography
        H = K @ T_1 @ np.linalg.inv(K)
        H_points_matrix = np.zeros((len(point), 3), dtype=np.float64)
        for k in range(len(point)):
            H_points_matrix[k] = [point[k][0], point[k][1], 1]

        homography_points_matrix = np.linalg.inv(H) @ H_points_matrix.T

        K_tr = np.linalg.inv(K)
        homography_points_matrix = K_tr @ homography_points_matrix
        normalized_undistorted_points = np.zeros((1, rows * cols, 2), dtype=np.float32)
        for t in range(len(point)):
            normalized_undistorted_points[0][t][0] = homography_points_matrix[0, t] / homography_points_matrix[2, t]
            normalized_undistorted_points[0][t][1] = homography_points_matrix[1, t] / homography_points_matrix[2, t]

        # Distort points back
        H_distorted_points = cv2.fisheye.distortPoints(normalized_undistorted_points, K, D)[0]
        image_H_distorted = np.zeros((rows, cols, 3), dtype=np.uint8)
        for t in range(len(H_distorted_points)):
            pt_x, pt_y = int(H_distorted_points[t][0]), int(H_distorted_points[t][1])
            if 0 <= pt_y < rows and 0 <= pt_x < cols:
                image_H_distorted[t // cols, t % cols] = image[pt_y, pt_x]
        # cv2.imwrite('new_undistorted_img.jpg', image_H_undistorted)
        cv2.imwrite('new_fisheye.jpg', image_H_distorted)
        # dst = np.zeros(image.shape, dtype=np.uint8)
        # mask = np.zeros(image.shape[:2], dtype=np.uint8)
        #
        # # Calculate the center and radius for the minimum enclosing circle
        # # circleCenter = (mask.shape[1] // 2, mask.shape[0] // 2)
        # circleCenter = (xc, yc)
        # # radius = min(mask.shape[1]-x0, mask.shape[0]) // 2
        # radius = min(mask.shape[1] - xc, mask.shape[0] - yc, xc, yc)
        #
        # # Draw a filled white circle on the mask
        # cv2.circle(mask, circleCenter, radius, 255, -1)
        # # cv2.circle(mask, (cols // 2, rows // 2), 600, (255, 255, 255), -1)
        #
        # # Apply the mask to the source image
        # cropped_img = cv2.bitwise_and(image_H_distorted, image_H_distorted, dst, mask=mask)
        # cv2.imwrite('new_fisheye.jpg', cropped_img)
        # mapx, mapy = cv2.fisheye.initUndistortRectifyMap(new_K, D, np.eye(3), new_K,
        #                                                  (cropped_img.shape[1], cropped_img.shape[0]),
        #                                                  cv2.CV_16SC2)
        # new_undistorted_img = cv2.remap(cropped_img, mapx, mapy, interpolation=cv2.INTER_LINEAR,
        #                                 borderMode=cv2.BORDER_CONSTANT)
        # cv2.imwrite('new_undistorted_img.jpg', new_undistorted_img)
        #
        # # 裁切后畸变矫正
        # cropped_img = cropped_img[yc - radius:yc + radius, xc - radius:xc + radius]
        # cv2.imwrite('crop_new_fisheye.jpg', cropped_img)
        # print(new_K)
        #
        # scale_x, scale_y = cropped_img.shape[1] / 1920, cropped_img.shape[0] / 1080
        # new_K[0][0], new_K[1][1], new_K[0][2], new_K[1][2], new_K[2][2] = new_K[0][0], new_K[1][1], \
        #                                               new_K[0][2] * scale_x, new_K[1][2] * scale_y, 1.0
        #
        # mapx, mapy = cv2.fisheye.initUndistortRectifyMap(new_K, D, np.eye(3), new_K, (cropped_img.shape[1], cropped_img.shape[0]),
        #                                                  cv2.CV_16SC2)
        # new_undistorted_img = cv2.remap(cropped_img, mapx, mapy, interpolation=cv2.INTER_LINEAR,
        #                                    borderMode=cv2.BORDER_CONSTANT)
        # cv2.imwrite('crop_new_undistorted_img.jpg', new_undistorted_img)
        # print(K)
        # print(new_K)

        # result_json = {
        #     "image_name": "result2_new.jpg",
        #     "camera_matrix": K.tolist(),
        #     "distortion": distortion,
        # }
        # with open("1.json", 'w', encoding='utf-8') as json_file:
        #     json.dump(result_json, json_file)
