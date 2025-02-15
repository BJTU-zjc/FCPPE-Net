import json
import math
import os
import os.path as osp
import random

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor

from PrincipalPoint_version_1.model.HeatmapNets.yolo_loss_base_v0 import YOLOLoss_hrnet
# from model.HeatmapNets.model_main_edge import ModelMain_PL
from PrincipalPoint_version_1.model.HeatmapNets.model_main import ModelMain_PL
from PrincipalPoint_version_1.train_heatmap_ddm_edge_mask_pconv_v1 import general_mask
from PrincipalPoint_version_1.util.DDM import gpu_calculate_ddm
from PrincipalPoint_version_1.model.vgg import InceptionV3


# from model.HeatmapNets.yolo_loss_base import YOLOLoss_hrnet
# from model.HeatmapNets.model_main_gt import ModelMain_PL
# from model.inception_v3 import InceptionV3

def grid2contour(grid):
    '''
    grid--image_grid used to show deform field
    type: numpy ndarray, shape： (h, w, 2), value range：(-1, 1)
    '''
    assert grid.ndim == 3
    x = np.arange(-1, 1, 2 / grid.shape[1])
    y = np.arange(-1, 1, 2 / grid.shape[0])
    X, Y = np.meshgrid(x, y)
    Z1 = grid[:, :, 0] + 2  # remove the dashed line
    Z1 = Z1[::-1]  # vertical flip
    Z2 = grid[:, :, 1] + 2

    plt.figure()
    plt.contour(X, Y, Z1, 15, colors='k')
    plt.contour(X, Y, Z2, 15, colors='k')
    plt.xticks(()), plt.yticks(())  # remove x, y ticks
    plt.title('deform field')
    plt.show()


def generate_heatmap(size, sigma, cx, cy):
    heatmap = np.zeros((size, size))
    heatmap[cx][cy] = 1
    heatmap = cv2.GaussianBlur(heatmap, (sigma, sigma), 0)
    am = np.amax(heatmap)
    heatmap /= am
    # plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    # plt.show()
    return heatmap


def draw_flow(offst, step=40, norm=1):
    # 在间隔分开的像素采样点处绘制光流
    h, w = 160, 160
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    # if norm:
    #     fx, fy = flow[y, x].T / abs(flow[y, x]).max() * step // 2
    # else:
    #     fx, fy = flow[y, x].T  # / flow[y, x].max() * step // 2
    # 创建线的终点
    e = (offst[y, x] + 1) * 80
    ex = e[:, 0]
    ey = e[:, 1]
    # ey = offst[y, x, 1] * 160
    lines = np.vstack([x, y, ex, ey]).T.reshape(-1, 2, 2)
    lines = lines.astype(np.uint32)
    # 创建图像并绘制
    vis = np.ones((h, w, 3), dtype=np.uint8)  # cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (x1, y1), 2, (0, 0, 255), -1)
    return vis


def test_detector():
    """Train detector."""
    # args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cpu')
    # torch.set_grad_enabled(True)
    # detector = FlowColumn().to(device)
    # detector = CameraPoseDetector(  # 3,16,6
    #     3, 16, 1).to(device)
    detector = InceptionV3().to(device)
    # detector = ModelMain_PL().to(device)
    detector.load_state_dict(torch.load(
        "D:/ZJC/project_instrinctCalib/Deep_instrinct/PrincipalPoint_version_1/weights/heatmap_ddm_edge_mask_pconv_v1/ddm_cameraParameter_detector_augmentation_4900_vgg_reg.pth"))
    # detector.load_state_dict(torch.load(
    #     "weights/regression_point/weights_inception_v3/ddm_cameraParameter_detector_augmentation_90_oriddm.pth"))

    detector.eval()
    root = "D:/ZJC/data/LaneLine_guide_PrincipalPointEstimate/test/"
    # root = "D:/ZJC/data/CenterPointDataset/test_dataset/"
    image_dir = "D:/ZJC/data/test_TRAINING/B/"
    center_loss_x = 0
    center_loss_y = 0
    center_loss = 0
    loss_hmap = 0
    center_loss_x_ppm = 0
    center_loss_y_ppm = 0
    count = 0
    loss = torch.nn.L1Loss()
    err_list = []

    method = 1
    # txt所在的路径
    # loss_save = "results/heatmap_ddm/heatmap_good/heatmap_err.txt"
    # 打开文件，mode='a'表示可以写入多行，mode='w'我是使用后只能写入1行
    # file_save = open(loss_save, mode='a')
    # for path in sorted(os.listdir(image_dir)):
    #     image_name = osp.join(image_dir, path)
    offset = []
    for i in range(-40, 50, 10):
        for j in range(-40, 50, 10):
            offset.append((i, j))
    result_jsons = []
    loss_coord = YOLOLoss_hrnet([[320, 320]], 35, (320, 320), device)
    circle_mask = general_mask(device)
    # for index in range(len(os.listdir(root+'img/'))):
    for index in range(370):
        index_num = index % 81
        # index_num = random.randint(0, 80)
        json_name = 'parameter/' + str(index).zfill(7)
        # with open(root + json_name + '.json') as file:
        #     image_name = json.load(file)['image_name']
        with open(root + json_name + '.json') as file:
            camera_matrix = json.load(file)['center_point']
        image_name = root + 'img/' + str(index).zfill(7) + ".jpg"
        mask_name = root + 'mask/' + str(index).zfill(7) + '.jpg'
        image = cv2.imread(image_name)
        origin_mask = cv2.imread(mask_name)
        input_size = 320
        image_transform = ToTensor()
        # image = image_transform(cv2.resize(origin_image, (input_size, input_size)))
        image_mask = image_transform(cv2.resize(origin_mask, (input_size, input_size)))
        # images = image.unsqueeze(0).to(device)
        lane_img_set = image_mask.unsqueeze(0).to(device)
        # 随机环切
        center_point = [camera_matrix[0], camera_matrix[1]]  # 原始主点
        # center_point_set = [6.3518368429424913e+02 / image.shape[1], 5.4604808802459536e+02 / image.shape[0]]
        # center_point = [6.3518368429424913e+02, 5.4604808802459536e+02]
        # image_size = [1080, 1280]

        dst = np.zeros(image.shape, dtype=np.uint8)
        # Create a single-channel mask image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # xc = int(center_point[0] + offset[index_num][0])
        # yc = int(center_point[1] + offset[index_num][1])
        xc = int(image.shape[1] / 2)
        yc = int(image.shape[0] / 2)
        # Calculate the center and radius for the minimum enclosing circle
        circleCenter = (xc, yc)
        # radius = min(mask.shape[1]-x0, mask.shape[0]) // 2
        radius = min(mask.shape[1] - xc, mask.shape[0] - yc, xc, yc)
        # Draw a filled white circle on the mask
        cv2.circle(mask, circleCenter, radius, 255, -1)
        # Apply the mask to the source image
        cropped_img = cv2.bitwise_and(image, image, dst, mask=mask)
        cropped_img = cropped_img[yc - radius:yc + radius, xc - radius:xc + radius]
        image_size = (cropped_img.shape[0], cropped_img.shape[1])
        # lane, lane_img = general_imageAddLane(cropped_img)
        # cv2.imshow("lane_mask", lane_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # canny = canny_(cropped_img)
        # cropped_edge = region_of_interest(canny)
        # lane_img = exact_lane(cropped_edge)
        # lane_img = cv2.resize(lane_img, (320, 320))
        # lane_img = cv2.cvtColor(lane_img, cv2.COLOR_GRAY2RGB)

        image_clone = cropped_img.copy()
        image_clone = cv2.resize(image_clone, (299, 299))
        # image_transform = ToTensor()
        images = image_transform(image_clone).to(device)  # 输入
        # lane_img_set = image_transform(lane_img).to(device)

        # 新的主点归一化真值
        camera_matrix[0] = float(cropped_img.shape[1] / 2 - (xc - center_point[0]))
        camera_matrix[1] = float(cropped_img.shape[0] / 2 - (yc - center_point[1]))
        center_point_set = [camera_matrix[0] / cropped_img.shape[1], camera_matrix[1] / cropped_img.shape[0]]
        center_point_set = torch.tensor(center_point_set).to(device)
        # 新的主点真值
        center_point = [camera_matrix[0], camera_matrix[1]]
        # camera_matrix_x = float(cropped_img.shape[0] / 2 - (xc - center_point[0]))
        # camera_matrix_y = float(cropped_img.shape[1] / 2 - (yc - center_point[1]))

        # 新的主点真值
        # center_point = [camera_matrix_x, camera_matrix_y]
        center_point = torch.tensor(center_point).to(device)
        # heatmap_l = generate_heatmap(int(input_size / 4), 9, int(camera_matrix[0]/cropped_img.shape[1] * 80), int(camera_matrix[1] / cropped_img.shape[0] * 80))
        heatmap_l = gpu_calculate_ddm(input_size / 4, input_size / 4, int(0.5 * 80), int(0.5 * 80))

        inl = torch.Tensor(heatmap_l).unsqueeze(0).unsqueeze(0).to(device)
        images = images.unsqueeze(0)
        # lane_img_set = lane_img_set.unsqueeze(0)
        label = np.zeros((1135, 2))
        # for i in range(14):
        #     for j in range(81):
        #         label[i * 81 + j + 1] = offset[j]
        # label[index_num+1] = 1

        with torch.no_grad():
            if method == 1:
                pre_center = detector(images, False)
                pre_center_x = pre_center[0, 0]
                pre_center_y = pre_center[0, 1]
                loss_mse = torch.nn.MSELoss(reduction='mean')

                # a = pre_center_x * oriimage.shape[1]-center_point[0]
                center_loss_x += loss(pre_center_x * cropped_img.shape[1], center_point[0])
                center_loss_y += loss(pre_center_y * cropped_img.shape[0], center_point[1])
                center_loss += loss_mse(pre_center, center_point_set)
                err = math.sqrt(pow(float(center_point[0] - (pre_center_x * cropped_img.shape[1])), 2) +
                                pow(float(center_point[1] - (pre_center_y * cropped_img.shape[0])), 2))
                # err = err / (math.sqrt(pow(image_size[1], 2) + pow(image_size[0], 2)))
                err_list.append(err)
                # result_json = {
                #     "pred_result": err}
                # result_jsons.append(result_json)
                print(err)
                # img1 = cv2.circle(cropped_img, (int(center_point[0]), int(center_point[1])), 9, (255, 0, 0), -1)
                # # cv2.imwrite("results/CenterPoint/" + json_name + "_gt.jpg", img1)
                # img2 = cv2.circle(img1,
                #                   (int(pre_center_x * cropped_img.shape[1]), int(pre_center_y * cropped_img.shape[0])),
                #                   9,
                #                   (0, 0, 255), -1)
                # cv2.imwrite("results/CenterPoint/" + str(index) + "_pred_regression.jpg", img2)
            elif method == 2:
                pre_label = detector(images, False)
                pre_class = torch.argmax(pre_label, dim=-1).item()
                pre_offset = label[pre_class]

                pre_center_x = float(cropped_img.shape[1] / 2 - pre_offset[0])
                pre_center_y = float(cropped_img.shape[0] / 2 - pre_offset[1])
                pre_center = [pre_center_x, pre_center_y]
                pre_center = torch.tensor(pre_center).to(device)
                loss_mse = torch.nn.MSELoss(reduction='mean')

                # a = pre_center_x * oriimage.shape[1]-center_point[0]
                center_loss_x += loss(pre_center[0], center_point[0])
                center_loss_y += loss(pre_center[1], center_point[1])
                center_loss += loss_mse(pre_center, center_point_set)
                err = math.sqrt(pow(float(center_point[0] - (pre_center_x)), 2) +
                                pow(float(center_point[1] - (pre_center_y)), 2))
                # err = err / (math.sqrt(pow(image_size[1], 2) + pow(image_size[0], 2)))
                err_list.append(err)
                result_json = {
                    "pred_result": err}
                result_jsons.append(result_json)
                # img1 = cv2.circle(cropped_img, (int(center_point[0]), int(center_point[1])), 9, (255, 0, 0), -1)
                # # cv2.imwrite("results/CenterPoint/" + json_name + "_gt.jpg", img1)
                # img2 = cv2.circle(img1, (int(pre_center_x), int(pre_center_y)), 9,
                #                   (0, 0, 255), -1)
                # cv2.imwrite("results/CenterPoint/" + str(index) + "_pred_cls.jpg", img2)
                print(str(count) + ":", err)
            elif method == 3:
                # pred = detector(images, True)
                # hm, outh, coord, offset = detector(images, inl)
                # hm, outh, coord, offset = detector(images, inl)
                hm, outh, coord = detector(images)
                # pred_np = pred.cpu().numpy()
                # predh_np = predh.cpu().numpy()

                # src = cv2.resize(cropped_img, (80, 80))
                # plt.imshow(pred_np[0][0], cmap='hot', interpolation='nearest')
                # # plt.show()
                # plt.axis("off")
                # plt.savefig("results/night/" + str(index) + "_80lane_conf.jpg", bbox_inches="tight", pad_inches=0)
                # src1 = cv2.imread("results/night/" + str(index) + "_80heatmap.jpg")
                # img1 = cv2.addWeighted(src, 1.0, cv2.resize(src1, (80, 80)), 0.5, 1, dtype=cv2.CV_32F)
                # cv2.imwrite("results/night/" + str(index) + "_80ddm.jpg", img1)

                # src = cv2.resize(cropped_img, (160, 160))
                # plt.imshow(predh_np[0][0], cmap='hot', interpolation='nearest')
                # # plt.show()
                # plt.axis("off")
                # plt.savefig("results/night/" + str(index) + "_160heatmap.jpg", bbox_inches="tight",
                #             pad_inches=0)
                # src1 = cv2.imread("results/heatmapAddImage/" + str(index) + "_160heatmap.jpg")
                # img1 = cv2.addWeighted(src, 1.0, cv2.resize(src1, (160, 160)), 0.5, 1, dtype=cv2.CV_32F)
                # cv2.imwrite("results/heatmapAddImage/" + str(index) + "_160ddm.jpg", img1)
                # heatmap1 = torch.mean(offset.cpu().detach(), dim=0).squeeze()
                # img_shape = [80, 80]
                # x = np.arange(-1, 1, 2 / img_shape[1])
                # y = np.arange(-1, 1, 2 / img_shape[0])
                # X, Y = np.meshgrid(x, y)
                # regular_grid = np.stack((X, Y), axis=2)
                # rand_field = np.random.rand(*img_shape, 2)
                # rand_field_norm = rand_field.copy()
                # rand_field_norm[:, :, 0] = rand_field_norm[:, :, 0] * 2 / img_shape[1]
                # rand_field_norm[:, :, 1] = rand_field_norm[:, :, 1] * 2 / img_shape[0]
                #
                # sampling_grid = regular_grid + rand_field_norm
                # sampling_grid = (heatmap1).numpy()
                # grid2contour(sampling_grid)
                det, conf = loss_coord(coord)

                for id, detections in enumerate(det):
                    # for id, detections in enumerate(detl):
                    _, indextensor = detections.max(0)
                    detections = detections[indextensor[2]].unsqueeze(0)

                    for x1, y1, conf in detections:
                        org_x = center_point[0]
                        org_y = center_point[1]
                        center_loss_x += loss(x1 / 320 * cropped_img.shape[1], center_point[0])
                        center_loss_y += loss(y1 / 320 * cropped_img.shape[0], center_point[1])
                        err = math.sqrt(pow(float(org_x - (x1 / 320 * cropped_img.shape[1])), 2) + pow(
                            float(org_y - (y1 / 320 * cropped_img.shape[0])), 2))
                        # err = err_dist / (math.sqrt(pow(image_size[1], 2) + pow(image_size[0], 2)))
                        err_list.append(err)
                        result_json = {
                            "pred_result": err}
                        result_jsons.append(result_json)
                        # img1 = cv2.circle(cropped_img, (int(center_point[0]), int(center_point[1])), 9, (255, 0, 0), -1)
                        # # cv2.imwrite("results/CenterPoint/" + json_name + "_gt.jpg", img1)
                        # img2 = cv2.circle(img1, (int(x1 / 320 * image_size[1]), int(y1 / 320 * image_size[0])), 9,
                        #                   (0, 0, 255), -1)
                        # cv2.imwrite("results/CenterPoint/" + str(index) + "_ddmori.jpg", img2)
                        print(err)
            elif method == 4:
                # pred = detector(images, True)
                pred, predh, coord, offset = detector(images, lane_img_set, circle_mask, inl)
                # pred_offset = torch.mean(offset.detach(), dim=0).squeeze()
                # pred_np = pred_offset.cpu().numpy()
                # vis = draw_flow(pred_np)
                # plt.imshow(vis, cmap='hsv')
                # plt.axis('off')
                # plt.show()
                # image0 = cv2.applyColorMap(pred_np, cv2.COLORMAP_JET)
                # cv2.imshow("mask", image0)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                #
                # # src = cv2.resize(cropped_img, (80, 80))
                # plt.imshow(pred_np[0], cmap='hot', interpolation='nearest')
                # # plt.show()
                # plt.axis("off")
                # plt.savefig("results/heatmap_ddm_edge_mask_pconv_v1/lane/" + str(index) + "_160offset.jpg", bbox_inches="tight", pad_inches=0)

                # pred_np = pred.cpu().numpy()
                #
                # # src = cv2.resize(cropped_img, (80, 80))
                # plt.imshow(pred_np[0][0], cmap='hsv', interpolation='nearest')
                # # plt.show()
                # plt.axis("off")
                # plt.savefig("results/lane_conf/" + str(index) + "_80lane_conf.jpg", bbox_inches="tight", pad_inches=0)
                det, conf = loss_coord(coord)

                for id, detections in enumerate(det):
                    # for id, detections in enumerate(detl):
                    _, indextensor = detections.max(0)
                    detections = detections[indextensor[2]].unsqueeze(0)

                    for x1, y1, conf in detections:
                        org_x = center_point[0]
                        org_y = center_point[1]
                        center_loss_x += loss(x1 / 320 * cropped_img.shape[1], center_point[0])
                        center_loss_y += loss(y1 / 320 * cropped_img.shape[0], center_point[1])
                        err = math.sqrt(pow(float(org_x - (x1 / 320 * cropped_img.shape[1])), 2) + pow(
                            float(org_y - (y1 / 320 * cropped_img.shape[0])), 2))
                        # if err > 100:
                        #     img1 = cv2.circle(cropped_img, (int(center_point[0]), int(center_point[1])), 9, (255, 0, 0), -1)
                        #     img2 = cv2.circle(img1, (int(x1 / 320 * image_size[1]), int(y1 / 320 * image_size[0])), 9,
                        #                                       (0, 0, 255), -1)
                        #     cv2.imwrite("results/CenterPoint/" + str(index) + "_err.jpg", img2)
                        # err = err_dist / (math.sqrt(pow(image_size[1], 2) + pow(image_size[0], 2)))
                        err_list.append(err)
                        print(err)
        count += 1
        torch.cuda.empty_cache()
    # for i in range(len(result_jsons)):
    #     result = result_jsons[i]
    #     with open("results/result_jsons/" + str(i).zfill(7) + "_ddmori.json", 'w',
    #               encoding='utf-8') as json_file:
    #         json.dump(result, json_file)
    print("center_loss_x : ", center_loss_x / count, center_loss_x_ppm / count)
    print("center_loss_y : ", center_loss_y / count, center_loss_y_ppm / count)
    print("loss_h: ", loss_hmap / count)
    print("mean_err:", sum(err_list) / len(err_list))
    print([sum(i <= 0.1 for i in err_list) / len(err_list),
           sum(i <= 0.2 for i in err_list) / len(err_list),
           sum(i <= 0.3 for i in err_list) / len(err_list),
           sum(i <= 0.4 for i in err_list) / len(err_list),
           sum(i <= 0.5 for i in err_list) / len(err_list),
           sum(i <= 0.6 for i in err_list) / len(err_list),
           sum(i <= 0.7 for i in err_list) / len(err_list),
           sum(i <= 0.8 for i in err_list) / len(err_list),
           sum(i <= 0.9 for i in err_list) / len(err_list),
           sum(i <= 1 for i in err_list) / len(err_list),
           sum(i > 1 for i in err_list) / len(err_list)])
    # 关闭文件


test_detector()
