import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Scale(nn.Module):
    def __init__(self, is_cuda=True):
        super(Scale, self).__init__()
        self.w, self.b = torch.rand(1), torch.rand(1)
        self.is_cuda = is_cuda
    def forward(self, x):
        self.w = (self.w).repeat(x.size()[0], x.size()[1], x.size()[2], x.size()[3])
        self.b = (self.b).repeat(x.size()[0], x.size()[1], x.size()[2], x.size()[3])
        self.w, self.b = Variable(self.w), Variable(self.b)
        if self.is_cuda:
            self.w = (self.w).cuda()
            self.b = (self.b).cuda()
        pdb.set_trace()
        x = self.w * x + self.b
        return x

class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def load_caffe_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    params = h5f['data']
    vgg16_rgb = net.rgbnet.features.state_dict()
    vgg16_d = net.dnet.features.state_dict()
    pdb.set_trace()
    for name, val in vgg16_rgb.items():
        i, j = int(name[4]), int(name[6]) + 1
        if name.find('bn.w') >= 0 or name.find('bn.b') >=0:
            ptype = '0' if name[-1] == 't' else '1'#0 is weight, 1 is biases
            key = 'sc_conv{}_{}'.format(i, j)
            param = torch.from_numpy(np.asarray(params[key][ptype]))
            val.copy_(param)
        elif name.find('.running_') >=0 :
            ptype = '0' if name[-1] == 'n' else '1'#0 is weight, 1 is biases
            key = 'bn_conv{}_{}'.format(i, j)
            param = torch.from_numpy(np.asarray(params[key][ptype]) / np.asarray(params[key]['2']))
            val.copy_(param)
        else:
            ptype = '0' if name[-1] == 't' else '1'#0 is weight, 1 is biases
            key = 'conv{}_{}'.format(i, j)
            param = torch.from_numpy(np.asarray(params[key][ptype]))
            val.copy_(param)
    for name, val in vgg16_d.items():
        i, j = int(name[4]), int(name[6]) + 1
        if name.find('bn.w') >= 0 or name.find('bn.b') >=0:
            ptype = '0' if name[-1] == 't' else '1'#0 is weight, 1 is biases
            key = 'sc_conv{}_{}d'.format(i, j)
            param = torch.from_numpy(np.asarray(params[key][ptype]))
            val.copy_(param)
        elif name.find('.running_') >=0 :
            ptype = '0' if name[-1] == 'n' else '1'#0 is weight, 1 is biases
            key = 'bn_conv{}_{}d'.format(i, j)
            param = torch.from_numpy(np.asarray(params[key][ptype]) / np.asarray(params[key]['2']))
            val.copy_(param)
        else:
            ptype = '0' if name[-1] == 't' else '1'#0 is weight, 1 is biases
            key = 'conv{}_{}d'.format(i, j)
            param = torch.from_numpy(np.asarray(params[key][ptype]))
            val.copy_(param)
    # fc6 fc7
    frcnn_dict = net.state_dict()
    pairs = {'fc6.fc': 'my_fc6', 'fc7.fc': 'fc7', 'score_fc.fc': 'cls_score', 'bbox_fc.fc': 'bbox_pred_3d'}
    #pairs = {'fc7.fc': 'fc7', 'score_fc.fc': 'cls_score', 'bbox_fc.fc': 'bbox_pred_3d'}
    for k, v in pairs.items():
        key = '{}.weight'.format(k)
        param = torch.from_numpy(np.asarray(params[v]['0']))
        frcnn_dict[key].copy_(param)

        key = '{}.bias'.format(k)
        param = torch.from_numpy(np.asarray(params[v]['1']))
        frcnn_dict[key].copy_(param)
def load_pretrained_npy(faster_rcnn_model, fname):
    params = np.load(fname).item()
    # vgg16
    vgg16_rgb = faster_rcnn_model.rgbnet.features.state_dict()
    for name, val in vgg16_rgb.items():
        if name.find('bn.') >= 0:
            continue
        i, j = int(name[4]), int(name[6]) + 1
        ptype = 'weights' if name[-1] == 't' else 'biases'
        key = 'conv{}_{}'.format(i, j)
        param = torch.from_numpy(params[key][ptype])
        if ptype == 'weights':
            param = param.permute(3, 2, 0, 1)
        val.copy_(param)
    vgg16_d = faster_rcnn_model.dnet.features.state_dict()
    for name, val in vgg16_d.items():
        if name.find('bn.') >= 0:
            continue
        i, j = int(name[4]), int(name[6]) + 1
        ptype = 'weights' if name[-1] == 't' else 'biases'
        key = 'conv{}_{}'.format(i, j)
        param = torch.from_numpy(params[key][ptype])
        if ptype == 'weights':
            param = param.permute(3, 2, 0, 1)
        val.copy_(param)
        '''
        if name.find('bn.') >= 0:
            continue
        i = int(name[4])
        if i < 3:
            j = int(name[6]) + 1
        else:
            j = int(name[6])/2 + 1
        ptype = 'weights' if name[-1] == 't' else 'biases'
        key = 'conv{}_{}'.format(i, j)
        param = torch.from_numpy(params[key][ptype])

        if ptype == 'weights':
            param = param.permute(3, 2, 0, 1)

        val.copy_(param)
        '''
    '''
    # fc6 fc7
    frcnn_dict = faster_rcnn_model.state_dict()
    pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7'}
    for k, v in pairs.items():
        key = '{}.weight'.format(k)
        param = torch.from_numpy(params[v]['weights']).permute(1, 0)
        frcnn_dict[key].copy_(param)

        key = '{}.bias'.format(k)
        param = torch.from_numpy(params[v]['biases'])
        frcnn_dict[key].copy_(param)
    '''

def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)
