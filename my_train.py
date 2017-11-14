import os
import torch
import numpy as np
import pdb
from datetime import datetime
import argparse
from faster_rcnn import network
#from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.my_faster_rcnn import Network
from faster_rcnn.utils.timer import Timer
import cPickle
import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.roi_data_layer.roidb import normalize_bbox_3d_targets
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None

def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

#python my_train.py --inputfile inria_train --iternum 100000 --savepath HHA

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inputfile', type=str,
        help='input file')
    parser.add_argument('--iternum', type=int, 
        help='number or iter')
    parser.add_argument('--savepath', type=str, 
        help='save path')
    return parser
parser = parse_args()
args = parser.parse_args()

# hyper-parameters
# ------------
#imdb_name = 'voc_2007_trainval'
#imdb_name = 'inria_train'
imdb_name = args.inputfile
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
#pretrained_model = 'data/pretrain_model/VGG_imagenet.npy'
pretrained_model = 'models/rgbd_3det_iter_40000.h5'
#output_dir = 'models/saved_model3'
output_dir = 'models/%s'%args.savepath

start_step = 0
#end_step = 100000
end_step = args.iternum
lr_decay_steps = {30000, 60000, 90000}
lr_decay = 0.1
print 'iter : %d, step_size : %s, lr_decay : %f'%(end_step, lr_decay_steps, lr_decay)
#pdb.set_trace()
rand_seed = 1024
_DEBUG = True
use_tensorboard = True
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
#lr = cfg.TRAIN.LEARNING_RATE
lr = 0.005
#momentum = cfg.TRAIN.MOMENTUM
momentum = 0.9
#weight_decay = cfg.TRAIN.WEIGHT_DECAY
weight_decay = 0.0005
#disp_interval = cfg.TRAIN.DISPLAY
disp_interval = 1
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load data
imdb = get_imdb(imdb_name)
#rdl_roidb.prepare_roidb(imdb)
#roidb = imdb.roidb
setType = imdb_name.split('_')[-1]
cache_file = os.path.join('./', 'roidb_' + setType + '_19.pkl')
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as fid:
        roidb = cPickle.load(fid)
    print 'raw data is loaded from {}'.format(cache_file)
roidb, means, stds = normalize_bbox_3d_targets(roidb)
#roidb = normalize_bbox_3d_targets(roidb)
data_layer = RoIDataLayer(roidb, imdb.num_classes)
# load net
net = Network(classes=imdb.classes, debug = _DEBUG)
network.weights_normal_init(net, dev=0.01)
#network.load_pretrained_npy(net, pretrained_model)
# model_file = '/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5'
#model_file = './%s/faster_rcnn_20000.h5'%output_dir
#network.load_net(pretrained_model, net)
network.load_caffe_net(pretrained_model, net)
pdb.set_trace()
# exp_name = 'vgg16_02-19_13-24'
# start_step = 60001
# lr /= 10.
# network.weights_normal_init(1[net.bbox_fc, net.score_fc, net.fc6, net.fc7], dev=0.01)

net.cuda()
net.train()

#params = list(net.parameters())
params = list(net.parameters())
#optimizer = torch.optim.Adam(params[-8:], lr=lr)
optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
#pdb.set_trace()
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        exp_name = datetime.now().strftime('vgg16_%m-%d_%H-%M')
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
for step in range(start_step, end_step+1):
    print step
    # get one batch
    blobs = data_layer.forward()
    #pdb.set_trace()
    #blobs.keys() = ['bbox_3d_targets', 'bbox_loss_3d_weights', 'img', 'labels', 'rois', 'dmap']
    im_data = blobs['img']
    im_dmap = blobs['dmap']
    bbox_3d_targets = blobs['bbox_3d_targets']
    bbox_loss_3d_weights = blobs['bbox_loss_3d_weights']
    labels = blobs['labels']
    rois = blobs['rois']
    pdb.set_trace()
    #im_info = blobs['im_info']
    #gt_boxes = blobs['gt_boxes']
    #gt_ishard = blobs['gt_ishard']
    #dontcare_areas = blobs['dontcare_areas']
    print 'now at %s and %s'%(blobs['name'][0], blobs['name'][1])
    # forward
    net(im_data, im_dmap, rois, bbox_3d_targets, bbox_loss_3d_weights, labels)
    #loss = net.loss + net.rpn.loss
    loss = net.loss
    if _DEBUG:
        tp += float(net.tp)
        tf += float(net.tf)
        fg += net.fg_cnt
        bg += net.bg_cnt
    #print 'tp:%f, tf:%f, fg:%f, bg:%f'%(tp, tf, fg, bg)
    train_loss += loss.data[0]
    step_cnt += 1
    #pdb.set_trace()
    # backward
    optimizer.zero_grad()
    loss.backward()
    network.clip_gradient(net, 10.)
    optimizer.step()
    #pdb.set_trace()
    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration

        #log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
        #    step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps)
        log_text = 'step %d, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
            step, train_loss / step_cnt, fps, 1./fps)
        log_print(log_text, color='green', attrs=['bold'])

        if _DEBUG:
            log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt))
            #log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
            #    net.rpn.cross_entropy.data.cpu().numpy()[0], net.rpn.loss_box.data.cpu().numpy()[0],
            #    net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
            #)
            log_print('\trcnn_cls: %.4f, rcnn_box: %.4f' % (
                net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
            )
            #print 'run'
            #pdb.set_trace()
        re_cnt = True
        pdb.set_trace()
    if use_tensorboard and step % log_interval == 0:
        exp.add_scalar_value('train_loss', train_loss / step_cnt, step=step)
        exp.add_scalar_value('learning_rate', lr, step=step)
        if _DEBUG:
            exp.add_scalar_value('true_positive', tp/fg*100., step=step)
            exp.add_scalar_value('true_negative', tf/bg*100., step=step)
            losses = {'rpn_cls': float(net.rpn.cross_entropy.data.cpu().numpy()[0]),
                      'rpn_box': float(net.rpn.loss_box.data.cpu().numpy()[0]),
                      'rcnn_cls': float(net.cross_entropy.data.cpu().numpy()[0]),
                      'rcnn_box': float(net.loss_box.data.cpu().numpy()[0])}
            exp.add_scalar_dict(losses, step=step)

    if (step % 20000 == 0) and step > 0:
    #if (step % 1 == 0) and step > 0:
        
        origin_1 = net.state_dict()['bbox_fc.fc.weight'].cpu().numpy().copy()
        origin_2 = net.state_dict()['bbox_fc.fc.bias'].cpu().numpy().copy()
        new_weight = torch.FloatTensor(net.state_dict()['bbox_fc.fc.weight'].cpu().numpy() \
                                                            *stds[:, np.newaxis]).cuda()
        new_bias = torch.FloatTensor(net.state_dict()['bbox_fc.fc.bias'].cpu().numpy() \
                                                                    *stds + means).cuda()
        net.state_dict()['bbox_fc.fc.weight'].copy_(new_weight)
        net.state_dict()['bbox_fc.fc.bias'].copy_(new_bias)
        #par_lst = list(net.parameters())
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        network.save_net(save_name, net)
        #print('save model: {}'.format(save_name))
        net.state_dict()['bbox_fc.fc.weight'].copy_(torch.FloatTensor(origin_1).cuda())
        net.state_dict()['bbox_fc.fc.bias'].copy_(torch.FloatTensor(origin_2).cuda())
        
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        network.save_net(save_name, net)
    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False

