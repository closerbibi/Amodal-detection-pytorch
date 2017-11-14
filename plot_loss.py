import numpy as np
import matplotlib.pyplot as plt
import cv2, pdb, os, sys, argparse
sys.path.append('/home/kevin/3D_project/code')
import basic as ba

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfile', type=str,
        help='input log file')
    return parser

parser = parse_args()
args = parser.parse_args()

def plot_loss(data):
    save_path = '/home/kevin/3D_project/code/faster_rcnn_pytorch_new/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_loss = []
    for line in data[40000:]:
        if line.split('\n')[0].split(',')[0].split(':')[0] == 'iter ':
            title = line.split('\n')[0]
        #pdb.set_trace()
        if line.split(',')[0].split(' ')[0] == 'step':
            loss = float(line.split(',')[2].split(' ')[-1])
            all_loss.append(loss)
    x_axis = range(len(all_loss))
    #decay  = title.split(':')[2].split(')')[0].split('(')[-1].split('[')[-1].split(']')[0]
    #iter_num = title.split(',')[0].split(':')[-1].split(' ')[-1]
    #pdb.set_trace()
    plt.xlabel('iter')
    plt.ylabel('Loss')
    #plt.title(title)
    plt.plot(x_axis, all_loss, 'r')
    #plt.savefig(save_path + '/%s_%s_%s.jpg' \
    #    %(iter_num, decay.split(',')[0], decay.split(',')[-1].split(' ')[-1]))
    pdb.set_trace()
    


if __name__ == '__main__':
    
    data = ba.readtxt(args.inputfile)
    plot_loss(data)
