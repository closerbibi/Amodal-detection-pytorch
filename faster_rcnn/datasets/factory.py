# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

#import datasets.pascal_voc
#import datasets.imagenet3d
#import datasets.kitti
#import datasets.kitti_tracking
import numpy as np
import pdb
from .pascal_voc import pascal_voc
#from .imagenet3d import imagenet3d
#from .kitti import kitti
#from .kitti_tracking import kitti_tracking
#from .nthu import nthu
#from .coco import coco
from .inria import inria
#from .kittivoc import kittivoc


inria_devkit_path = './'
for split in ['train', 'test']:
    name = '{}_{}'.format('inria', split)
    __sets[name] = (lambda split=split: inria(split, inria_devkit_path))
#pdb.set_trace()
def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = datasets.pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))
"""
# Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
# but only returning the first k boxes
for top_k in np.arange(1000, 11000, 1000):
    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}_top_{:d}'.format(year, split, top_k)
            __sets[name] = (lambda split=split, year=year, top_k=top_k:
                    _selective_search_IJCV_top_k(split, year, top_k))
"""

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        print name
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))
"""
# KITTI dataset
for split in ['train', 'val', 'trainval', 'test']:
    name = 'kitti_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.kitti(split))
"""
# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))
"""
# NTHU dataset
for split in ['71', '370']:
    name = 'nthu_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.nthu(split))
"""

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
