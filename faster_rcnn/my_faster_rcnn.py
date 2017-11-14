import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.timer import Timer
from utils.blob import im_list_to_blob
from fast_rcnn.nms_wrapper import nms
from rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
import pdb
import network
from network import Conv2d, FC
# from roi_pooling.modules.roi_pool_py import RoIPool
from roi_pooling.modules.roi_pool import RoIPool
from my_vgg16 import VGG16

class RGBDNetwork(nn.Module):
    def __init__(self):
        super(RGBDNetwork, self).__init__()
        self.features = VGG16(bn = True)
        self.roi_pool = RoIPool(7, 7, 1.0/16)

    def forward(self, im_data, rois):
        im_data = network.np_to_variable(im_data, is_cuda = True)
        vgg_feature = self.features(im_data)
        pool_feature = self.roi_pool(vgg_feature, rois)
        return pool_feature

class Network(nn.Module):
    n_classes = 20
    classes = ['__background__', 'bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter',
                'desk', 'door', 'dresser', 'garbagebin', 'lamp', 'monitor', 'night_stand',
                'pillow', 'sink', 'sofa', 'table', 'tv', 'toilet']
    def __init__(self, classes=None, debug=False):
        super(Network, self).__init__()
        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
        self.rgbnet = RGBDNetwork()
        self.dnet = RGBDNetwork()
        self.fc6 = FC(512*7*7*2, 4096)
        self.fc7 = FC(4096, 4096)
        self.score_fc = FC(4096, self.n_classes, relu = False)
        self.bbox_fc = FC(4096, self.n_classes * 7, relu = False)
        #loss
        self.cross_entropy = None
        self.loss_box = None
        self.debug = debug
    def forward(self, rgb_data, depth_data, rois, bbox_3d_targets=None, bbox_loss_3d_weights=None, labels=None):
        if self.training:
            roi_data = self.proposal_target_layer(rois, labels, bbox_3d_targets, bbox_loss_3d_weights)
            rois = roi_data[0]
        else:
            rois = network.np_to_variable(rois, is_cuda = True)
        rgb_feature = self.rgbnet(rgb_data, rois)
        depth_feature = self.dnet(depth_data, rois)
        rgb_feature = rgb_feature.view(rgb_feature.size()[0], -1)
        depth_feature = depth_feature.view(depth_feature.size()[0], -1)
        x = torch.cat((rgb_feature, depth_feature), 1) #concatenate rgb and depth feature
        x = self.fc6(x)
        x = F.dropout(x, training = self.training)
        x = self.fc7(x)
        x = F.dropout(x, training = self.training)
        cls_score = self.score_fc(x)
        cls_prob = F.softmax(cls_score)
        bbox_pred = self.bbox_fc(x)
        if self.training:
            self.cross_entropy, self.loss_box = self.build_loss(cls_score, cls_prob, bbox_pred, roi_data)
        return cls_prob, bbox_pred, rois

    def build_loss(self, cls_score, cls_prob, bbox_pred, roi_data):
        #classification loss
        label =  roi_data[1].squeeze()
        fg_cnt = torch.sum(label.data.ne(0))
        bg_cnt = label.data.numel() - fg_cnt
        if self.debug:
            maxv, predict = cls_score.data.max(1)
            self.tp = torch.sum(predict[:fg_cnt].eq(label.data[:fg_cnt])) if fg_cnt > 0 else 0
            self.tf = torch.sum(predict[fg_cnt:].eq(label.data[fg_cnt:]))
            self.fg_cnt = fg_cnt
            self.bg_cnt = bg_cnt
        ce_weights = torch.ones(cls_score.size()[1])
        ce_weights[0] = float(fg_cnt) / bg_cnt
        ce_weights = ce_weights.cuda()
        cross_entropy = F.cross_entropy(cls_score, label, weight=ce_weights)
        #cross_entropy = F.cross_entropy(cls_score, label)
        # bounding box regression L1 loss
        bbox_targets, bbox_inside_weights = roi_data[2:]#bbox_targets and bbox_inside_weights is 3d
        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)
        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-4)
        #loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False)
        return cross_entropy, loss_box

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box

    @staticmethod
    def proposal_target_layer(rois, labels, bbox_3d_targets, bbox_loss_3d_weights):
        rois = network.np_to_variable(rois, is_cuda=True)
        labels = network.np_to_variable(labels, is_cuda=True, dtype=torch.LongTensor)
        bbox_targets = network.np_to_variable(bbox_3d_targets, is_cuda=True)
        bbox_inside_weights = network.np_to_variable(bbox_loss_3d_weights, is_cuda=True)
        return rois, labels, bbox_targets, bbox_inside_weights#, bbox_outside_weights











