"""Train camera pose detector."""
import logging
import multiprocessing
import time

import cv2
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from PrincipalPoint_version_1 import config
from PrincipalPoint_version_1.util import log
from PrincipalPoint_version_1.data import dataset_ddm_mask
from PrincipalPoint_version_1.model.HeatmapNets.model_main_edge_contourlet_v1_4 import ModelMain_PL
from PrincipalPoint_version_1.model.HeatmapNets.yolo_loss_base import YOLOLoss_hrnet, AutomaticWeightedLoss


def train_detector(args):
    """Train detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.gpu_id) if args.cuda else 'cpu')
    detector = ModelMain_PL().to(device)
    if args.detector_weights:
        print("Loading weights: %s" % args.detector_weights)
        detector.load_state_dict(torch.load(args.detector_weights))
    detector.train()
    lr = 1e-3
    awl = AutomaticWeightedLoss(4)
    optimizer = torch.optim.Adam([
        {'params': detector.parameters()},
        {'params': awl.parameters(), 'weight_decay': 0}
    ], lr=lr)
    if args.optimizer_weights:
        print("Loading weights: %s" % args.optimizer_weights)
        optimizer.load_state_dict(torch.load(args.optimizer_weights))

    logger = log.Logger(curve_names=['train_loss'], enable_visdom=False)
    torch.multiprocessing.set_sharing_strategy('file_system')
    # data_len = len(os.listdir(args.dataset_directory+'img/'))
    train_data_loader = DataLoader(
        dataset_ddm_mask.CameraPoseDataset(args.dataset_directory, data_len=3255),
        batch_size=8, shuffle=True,
        num_workers=2, drop_last=True
    )
    l1_loss = nn.L1Loss()
    loss_coor = YOLOLoss_hrnet([[320, 320]], 35, (320, 320), device)
    epoch_th = 0
    glob_iter = 0
    writer = SummaryWriter()
    # circle_mask = general_mask(device)
    for epoch_index in range(args.num_epochs):
        if epoch_index > epoch_th + 10 and epoch_index > 10:
            lr = lr / 5.0
            epoch_th = epoch_th + 10
            if lr < 1e-5:
                lr = 1e-5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        for iter_idx, (
                images, image_mask, center_points_set, heatmap_l_set, heatmap_h_set, mask, noobj_mask, tx,
                ty) in enumerate(train_data_loader):
            # combine batch_size images
            # torch.autograd.set_detect_anomaly(True)
            images = images.to(device)
            # image_mask = image_mask.to(device)
            center_points_set = center_points_set.to(device)
            mask = mask.to(device)
            noobj_mask = noobj_mask.to(device)
            tx = tx.to(device)
            ty = ty.to(device)

            heatmap_l_set = heatmap_l_set.to(device)
            heatmap_h_set = heatmap_h_set.to(device)

            start_time = time.time()
            optimizer.zero_grad()

            hm, outh, coord, offset = detector(images, heatmap_l_set)
            # ms_loss = torch.tensor(0., device=device, dtype=torch.float)
            # for _, f in enumerate(hm):
            #     ms_loss += l1_loss(f, F.interpolate(heatmap_l_set, size=f.size()[1:3], mode='bilinear',
            #                                                   align_corners=True))
            cur_loss = l1_loss(hm, heatmap_l_set.float())
            # cur_loss = ms_loss
            cur_lossh = l1_loss(outh, heatmap_h_set.float())

            loss_xy, loss_conf = loss_coor(coord, mask, noobj_mask, tx, ty, center_points_set)
            all_loss = awl(cur_loss, cur_lossh, loss_xy, loss_conf)

            all_loss.backward()

            optimizer.step()

            logger.log(epoch=epoch_index, iter=iter_idx, cur_loss=cur_loss, cur_lossh=cur_lossh,
                       all_loss=all_loss)

            torch.cuda.empty_cache()
            # 记录loss曲线
            glob_iter += 1
            loss_l, loss_h, loss_coord = cur_loss.item(), cur_lossh.item(), all_loss.item()
            duration = float(time.time() - start_time)
            example_per_second = 8 / duration
            lr = optimizer.param_groups[0]['lr']
            logging.info(
                "epoch [%.3d] iter = %d lossl = %.2f lossh = %.2f losscoord = %.2f example/sec = %.3f lr = %.5f" %
                (epoch_index, iter_idx, loss_l, loss_h, loss_coord, example_per_second, lr)
            )
            writer.add_scalar("lr",
                              lr,
                              glob_iter)
            writer.add_scalar("example/sec",
                              example_per_second,
                              glob_iter)
            # loss_epoch += _loss
            writer.add_scalar("lossl",
                              loss_l,
                              glob_iter)
            writer.add_scalar("lossh",
                              loss_h,
                              glob_iter)
            writer.add_scalar("losscoord",
                              loss_coord,
                              glob_iter)

            # if epoch_index % 10 == 0:
            #     torch.save(detector.state_dict(),
            #                'weights/heatmap_ddm_edge_mask_pconv_v1/cameraParameter_detector_augmentation_'
            #                '%d_contourlet_v1_4.pth' % epoch_index)
            # if epoch_index % 100 == 0:
            #     torch.save(optimizer.state_dict(),
            #                'weights/heatmap_ddm_edge_mask_pconv_v1/cameraParameter_optimize_augmentation_'
            #                '%d_contourlet_v1_4.pth' % epoch_index)
            if epoch_index % 10 == 0:
                torch.save(detector.state_dict(),
                           'weights/heatmap_ddm_edge_mask_pconv_v1/cameraParameter_detector_augmentation_'
                           '%d_heatmap_9_contourlet_v1_4.pth' % epoch_index)
            if epoch_index % 100 == 0:
                torch.save(optimizer.state_dict(),
                           'weights/heatmap_ddm_edge_mask_pconv_v1/cameraParameter_optimize_augmentation_'
                           '%d_contourlet_v1_4.pth' % epoch_index)
    # writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    train_detector(config.get_parser_for_training().parse_args())
