import json

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def polar_image(image, cx, cy):
    # Create a black image of the same size as the source image
    dst = np.zeros(image.shape, dtype=np.uint8)

    # Create a single-channel mask image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Calculate the center and radius for the minimum enclosing circle
    # circleCenter = (mask.shape[1] // 2, mask.shape[0] // 2)
    circleCenter = (int(cx), int(cy))
    # radius = min(mask.shape[1]-x0, mask.shape[0]) // 2
    radius = 40

    # Draw a filled white circle on the mask
    cv2.circle(mask, circleCenter, radius, 255, -1, lineType=cv2.LINE_AA)
    # Apply the mask to the source image
    cropped_img = cv2.bitwise_and(image, image, dst, mask=mask)
    cropped_img = cv2.blur(cropped_img, (7, 7))

    # 线性极坐标变换
    out = cv2.linearPolar(cropped_img, (int(cx), int(cy)), int(cy), cv2.INTER_LINEAR)
    print(out.shape)
    dst = cv2.Sobel(out, cv2.CV_64F, 0, 1, ksize=3)
    # 这条代码将正负取绝对值
    dst1 = cv2.convertScaleAbs(dst)
    # a = np.sum(dst1/255)
    # 显示原图和输出图像
    # cv2.imshow("Image", cropped_img)
    # cv2.imshow("out", out)
    # cv2.imshow("sobel", dst)
    # cv2.imshow("abs_sobel", dst1)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dst1 / 255


def calculate_ddm(height, width, x_c, y_c, k1, k2, k3, k4):
    ddm = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            r2 = (x - x_c) ** 2 + (y - y_c) ** 2
            ddm[y, x] = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3 + k4 * r2 ** 4
    return ddm


def generate_heatmap(size, sigma, x0, y0):
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    # The gaussian is not normalized, we want the center value to equal 1
    heatmap = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return heatmap


def gpu_calculate_ddm(height, sigma, x_c, y_c):
    # Create a grid of coordinates
    # x_coords = np.arange(width, dtype=np.float32)
    # y_coords = np.arange(height, dtype=np.float32)
    # y_grid, x_grid = np.meshgrid(y_coords, x_coords)
    x = np.arange(0, height, 1, np.float32)
    y = x[:, np.newaxis]
    # Calculate squared distances from the center
    r2 = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
    ddm = np.exp(-((r2 ** 0.25) / (2 * sigma ** 2)))
    # Compute the distortion
    # ddm = 1 + 0.1 * r2 + 0.01 * r2 ** 2 + 0.001 * r2 ** 3 + 0.0001 * r2 ** 4
    # ddm = 1 + 1 * r2**0.25
    normalized = cv2.normalize(ddm, None, 0, 1, cv2.NORM_MINMAX)
    # ddm = r2

    return normalized


def ddm_color_map(height, width, x_c, y_c, distortion):
    x_coords = np.arange(width, dtype=np.float32)
    y_coords = np.arange(height, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(y_coords, x_coords)

    # Calculate squared distances from the center
    r2 = np.sqrt((x_grid - x_c) ** 2 + (y_grid - y_c) ** 2)

    # Compute the distortion
    ddm = 1 + distortion[0] * r2 + distortion[1] * r2 ** 2 + distortion[2] * r2 ** 3 + distortion[3] * r2 ** 4
    # ddm = 1 + 1 * (r2) ** 0.5

    length_normalized = cv2.normalize(ddm, None, 0, 255, cv2.NORM_MINMAX)
    # hsv_image = np.zeros((320, 320, 3), dtype=np.uint8)
    # hsv_image[..., 0] = 255 - length_normalized  # 色调表示方向
    # hsv_image[..., 1] = 255  # 饱和度设置为最大
    # hsv_image[..., 2] = 255  # 亮度表示长度
    # color_map = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    # color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2GRAY)
    return 1 - length_normalized


def main():
    root = "D:/ZJC/data/instrinsic_test_dataset/"
    for index in range(1):
        for i in range(1):
            json_name = str(i) + str(index).zfill(6)
            with open(root + json_name + '.json') as file:
                image_name = json.load(file)['image_name']
            with open(root + json_name + '.json') as file:
                camera_matrix = json.load(file)['camera_matrix']
            with open(root + json_name + '.json') as file:
                distortion = json.load(file)['distortion']

            image = cv2.imread(image_name)
            center_point = [camera_matrix[2] / image.shape[1] * 160, camera_matrix[5] / image.shape[0] * 160]
            Dd_map1 = gpu_calculate_ddm(160, 9, center_point[0], center_point[1])
            # Dd_map1 = ddm_color_map(160, 160, center_point[0], center_point[1], distortion)
            plt.imshow(Dd_map1, cmap='hot', interpolation='nearest')
            # plt.show()
            plt.axis("off")
            plt.savefig("our.png", bbox_inches="tight", pad_inches=0)
            # Dd_map1 = Dd_map1.cpu().detach().numpy()
            # length_normalized = cv2.normalize(Dd_map1, None, 0, 255, cv2.NORM_MINMAX)
            # hsv_image = np.zeros((160, 160, 3), dtype=np.uint8)
            # hsv_image[..., 0] = 255-length_normalized  # 色调表示方向
            # hsv_image[..., 1] = 255  # 饱和度设置为最大
            # hsv_image[..., 2] = 255  # 亮度表示长度
            # color_map = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            # color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("ori", color_map)
            # cv2.waitKey(0)
            # _, binary = cv2.threshold(length_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #
            # M = cv2.moments(binary)
            # if M["m00"] != 0:
            #     cX = int(M["m10"] / M["m00"])
            #     cY = int(M["m01"] / M["m00"])
            # else:
            #     cX, cY = 0, 0
            #
            # print("Centroid coordinates:", (cX, cY))
            # img = cv2.circle(color_map, (cX, cY), 3, (255, 255, 255), -1)
            # img = cv2.circle(color_map, (int(center_point[0]), int(center_point[1])), 3,
            #                  (0,0,0), -1)
            # cv2.imshow("b", binary)
            # cv2.imshow("Image", color_map)
            # cv2.imshow("out", img)
            # # cv2.imshow("sobel", dst)
            # # cv2.imshow("abs_sobel", dst1)
            # #
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # polar_image(color_map, center_point[0], center_point[1])
            # max_pixel = np.max(Dd_map1)
            # min_pixel = np.min(Dd_map1)
            # delta = max_pixel - min_pixel
            # labels_int = ((Dd_map1 - min_pixel) / delta * 255)
            # # print(Dd_map1)
            # labels_int = labels_int * (-1)
            # labels_int = labels_int + 255
            # labels_int = labels_int.astype(np.uint8)
            # color_label = cv2.applyColorMap(labels_int, cv2.COLORMAP_JET)
            # cv2.imwrite("ddm_picture/ddm"+json_name+".jpg", color_label)


"""
箭头图
"""


def draw_flow(height, width, x_c, y_c, distortion, flow, step=40, norm=1):
    x_coords = np.arange(width, dtype=np.float32)
    y_coords = np.arange(height, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(y_coords, x_coords)

    # Calculate squared distances from the center
    r2 = np.sqrt((x_grid - x_c) ** 2 + (y_grid - y_c) ** 2)

    # Compute the distortion
    ddm = x_grid*(1 + distortion[0] * r2 + distortion[1] * r2 ** 2 + distortion[2] * r2 ** 3 + distortion[3] * r2 ** 4)
    # 在间隔分开的像素采样点处绘制光流
    im = np.ones((width, height, 3))
    h, w = im.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    if norm:
        fx, fy = flow[y, x].T / abs(flow[y, x]).max() * step // 2
    else:
        fx, fy = flow[y, x].T  # / flow[y, x].max() * step // 2
    # 创建线的终点
    ex = x + fx
    ey = y + fy
    lines = np.vstack([x, y, ex, ey]).T.reshape(-1, 2, 2)
    lines = lines.astype(np.uint32)
    # 创建图像并绘制
    vis = im.astype(np.uint8)  # cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (x1, y1), 2, (0, 0, 255), -1)
    return vis


if __name__ == '__main__':
    main()
    # img = ddm_color_map(320, 320, 160, 160)
    # cv2.imwrite("DDM.jpg", img)
    # cv2.imshow("sobel", dst)
    # cv2.imshow("abs_sobel", dst1)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
