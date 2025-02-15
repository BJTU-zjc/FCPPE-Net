import xml
import os.path as osp
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def calculate_ddm(height, width, x_c, y_c, k1, k2, k3, k4):
    ddm = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            r2 = (x - x_c) ** 2 + (y - y_c) ** 2
            ddm[y, x] = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3 + k4 * r2 ** 4
    return ddm


def get_camera_parameter(xml_name):
    dom = xml.dom.minidom.parse(osp.join(xml_name))  # 解析XML文件
    camera_matrix = dom.getElementsByTagName('data')[0].firstChild.data.split()
    camera_matrix = list(map(float, camera_matrix))
    distortion = dom.getElementsByTagName('data')[1].firstChild.data.split()
    distortion = list(map(float, distortion))
    return camera_matrix, distortion


def pinhole_direction(cx, cy, fx, fy, device, width=1920, height=1080):
    i, j = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device))

    # 计算每个像素相对于光心的方向向量
    directions = torch.stack([(j - cx),
                              (i - cy),
                              torch.ones_like(i)], dim=-1)

    # 归一化方向向量得到单位向量
    unit_directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    # 计算每个点的θ角和rd
    ru_x = (j - cx)
    ru_y = (i - cy)
    # theta = torch.atan2(ru_y, ru_x)
    ru = torch.sqrt(ru_x ** 2 + ru_y ** 2)
    # 计算每个点的θ角
    theta_x = torch.arctan2(ru_y, fx)  # 使用arctan2来处理x方向
    theta_y = torch.arctan2(ru_x, fy)  # 使用arctan2来处理y方向

    # 鱼眼相机等距模型中的rd
    rd_x = fx * theta_x
    rd_y = fy * theta_y
    rd = torch.sqrt((rd_x) ** 2 + (rd_y) ** 2)

    # 计算颜色编码的长度
    length = torch.log((ru - rd) + 0.00001)
    # length = ru - rd
    length_normalized = cv2.normalize(length.cpu().detach().numpy(), None, 0, 255, cv2.NORM_MINMAX)
    # mask = np.isnan(length_normalized) | np.isinf(length_normalized)
    # length_normalized = length_normalized[mask]
    # 创建颜色 映射
    # angle = torch.atan2(unit_directions[:, :, 1], unit_directions[:, :, 0])
    # angle_normalized = ((angle + np.pi) / (2 * np.pi) * 180).byte().cpu().detach().numpy()
    hsv_image = np.zeros((height, width, 3), dtype=np.uint8)
    # hsv_map = np.zeros((height, width, 2), dtype=np.uint8)
    # hsv_map[..., 1] = length_normalized  # 亮度表示长度
    # hsv_map[..., 0] = angle_normalized  # 色调表示方向
    hsv_image[..., 0] = length_normalized  # 色调表示方向
    hsv_image[..., 1] = 255  # 饱和度设置为最大
    hsv_image[..., 2] = 255 - length_normalized  # 亮度表示长度
    color_map = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2GRAY)
    # color_label = cv2.applyColorMap(np.uint8(length_normalized), cv2.COLORMAP_JET)
    # cv2.imwrite('heatmap2.jpg', color_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return color_map


def get_target(target, anchors, in_w, in_h, ignore_threshold, num_anchors):
    bs = target.size(0)
    mask = torch.zeros(bs, num_anchors, in_h, in_w, requires_grad=False)
    noobj_mask = torch.ones(bs, num_anchors, in_h, in_w, requires_grad=False)
    tx = torch.zeros(bs, num_anchors, in_h, in_w, requires_grad=False)
    ty = torch.zeros(bs, num_anchors, in_h, in_w, requires_grad=False)
    # tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
    # th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
    # tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
    # tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)
    for b in range(bs):
        # Convert to position relative to box
        gx = target[b, 0, 1] * in_w
        gy = target[b, 0, 2] * in_h
        gw = target[b, 0, 3] * in_w
        gh = target[b, 0, 4] * in_h
        # Get grid box indices
        gi = int(gx)
        gj = int(gy)
        # Get shape of gt box
        gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
        # Get shape of anchor box
        anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((num_anchors, 2)),
                                                          np.array(anchors)), 1))
        # Calculate iou between gt and anchor shapes
        anch_ious = bbox_iou(gt_box, anchor_shapes)
        noobj_mask[b, anch_ious > ignore_threshold, gj, gi] = 0
        best_n = np.argmax(anch_ious)
        # Masks
        mask[b, best_n, gj, gi] = 1
        # Coordinates
        tx[b, best_n, gj, gi] = gx - gi
        ty[b, best_n, gj, gi] = gy - gj

    return mask, noobj_mask, tx, ty


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def canny_(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    canny = cv2.Canny(blur, 200, 300)
    return canny


def region_of_interest(image):
    height, width = image.shape
    polygons = np.array([
        [(0, int(height * 0.1)), (0, int(height * 0.8)), (width, int(height * 0.8)), (width, int(height * 0.1))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def region_of_edge(image, polygons):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 20)
    return line_image


def exact_lane(cropped_image):
    contours, hierarchy = cv2.findContours(cropped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找出连通域
    # print(len(contours),hierarchy)
    box_list = []
    for i in range(len(contours)):
        # 筛掉面积过小的轮廓
        area = cv2.contourArea(contours[i])
        if area < 30:
            continue
        # 找到包含轮廓的最小矩形框
        rect = cv2.minAreaRect(contours[i])
        # 计算矩形框的四个顶点坐标
        box = cv2.boxPoints(rect)
        area = cv2.contourArea(box)
        # 筛掉面积过小的矩形
        if area < 150:
            continue
        box = np.intp(box)
        box_list.append(box)
    if len(box_list) == 0:
        draw_img = cropped_image.copy()
        dst = np.zeros(draw_img.shape, dtype=np.uint8)

        # Create a single-channel mask image
        mask = np.zeros(draw_img.shape[:2], dtype=np.uint8)

        xc = int(draw_img.shape[1] / 2)
        yc = int(draw_img.shape[0] / 2)
        # Calculate the center and radius for the minimum enclosing circle
        # circleCenter = (mask.shape[1] // 2, mask.shape[0] // 2)
        circleCenter = (xc, yc)
        # radius = min(mask.shape[1]-x0, mask.shape[0]) // 2
        radius = min(mask.shape[1] - xc, mask.shape[0] - yc, xc, yc)

        # Draw a filled white circle on the mask
        cv2.circle(mask, circleCenter, radius - 5, 255, -1)

        # Apply the mask to the source image
        draw_img = cv2.bitwise_and(draw_img, draw_img, dst, mask=mask)
        return draw_img
    else:
        edge_poly = np.stack(box_list)
        draw_img = region_of_edge(cropped_image, edge_poly)

        dst = np.zeros(draw_img.shape, dtype=np.uint8)

        # Create a single-channel mask image
        mask = np.zeros(draw_img.shape[:2], dtype=np.uint8)

        xc = int(draw_img.shape[1] / 2)
        yc = int(draw_img.shape[0] / 2)
        # Calculate the center and radius for the minimum enclosing circle
        # circleCenter = (mask.shape[1] // 2, mask.shape[0] // 2)
        circleCenter = (xc, yc)
        # radius = min(mask.shape[1]-x0, mask.shape[0]) // 2
        radius = min(mask.shape[1] - xc, mask.shape[0] - yc, xc, yc)

        # Draw a filled white circle on the mask
        cv2.circle(mask, circleCenter, radius - 5, 255, -1)

        # Apply the mask to the source image
        draw_img = cv2.bitwise_and(draw_img, draw_img, dst, mask=mask)
        lines = cv2.HoughLinesP(draw_img, 1, np.pi / 180, 70, np.array([]), minLineLength=5, maxLineGap=30)
        # averaged_lines = average_slope_intercept(lane_image, lines)
        line_image = display_lines(draw_img, lines)
        # plt.imshow(line_image, interpolation='nearest')
        # plt.show()
        return line_image
