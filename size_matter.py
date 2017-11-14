import numpy as np
import os, sys, pdb
sys.path.append('../')
import basic as ba
import matplotlib.pyplot as plt


#path = './data/DIRE/Annotations'
path = '/home/closerbibi/workspace/data/label_19'
save_path_json = './'
obj_struct = {}
save_path = './result_fig/size_result/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
def all_obj(ID):
    file_name = path + '/' + ID
    data = ba.readtxt(file_name)
    for obj in data:
        obj_tmp = obj.split(' - ')
        X_min = int(obj_tmp[0].split(',')[0].split('(')[1])
        Y_min = int(obj_tmp[0].split(',')[1].split(' ')[1].split(')')[0])
        X_max = int(obj_tmp[1].split(',')[0].split('(')[1])
        Y_max = int(obj_tmp[1].split(',')[1].split(' ')[1].split(')')[0])
        name = obj_tmp[2].split('(')[1].split(')')[0]
        if not name in obj_struct.keys():
            obj_struct[name] = []
        #obj_struct[name].append([X_min,Y_min,X_max,Y_max])
        obj_struct[name].append([X_max-X_min, Y_max-Y_min])

def size_matter(obj_struct):
    val = {}
    for index, key in enumerate(obj_struct):
        #pdb.set_trace()
        if not key in val.keys():
            val[key] = []
        #val = ba.median([e[0]/float(e[1]) for e in obj_struct[key]])
        print '%s'%key
        size_lst = [e[0]*e[1] for e in obj_struct[key]]
        ratio = np.mean([e[0]/float(e[1]) for e in obj_struct[key]])
        size_mean = np.mean(size_lst)
        size_median = ba.median(size_lst)
        sd = np.sqrt(np.var(size_lst))
        val[key].append([ratio, 1/ratio, size_mean, sd])
        width = [e[0] for e in obj_struct[key]]
        height = [e[1] for e in obj_struct[key]]
        #pdb.set_trace()
        width_mean = np.mean(width)
        height_mean = np.mean(height)
        width_median = ba.median(width)
        height_median = ba.median(height)
        print 'height_mean : %f'%(height_mean)
        print 'height_median : %f'%(height_median)
        print 'width_mean : %f'%(width_mean)
        print 'width_median : %f'%(width_median)
        print 'size_mean : %f'%size_mean
        print 'size_median : %f'%size_median
        print 'size_SD : %f'%(sd)
        # plot width
        plt.title('width_median : %f, width_mean : %f, SD : %f'%(width_median, width_mean, np.sqrt(np.var(width))))
        plt.xlabel('number of %s : %d'%(key, len(size_lst)))
        plt.ylabel('width')
        plt.scatter(range(len(width)), width)
        #pdb.set_trace()
        plt.savefig('%s/%s_width.jpg'%(save_path, key))
        plt.close()
        # plot height
        plt.title('height_median : %f, height_mean : %f, SD : %f'%(height_median, height_mean, np.sqrt(np.var(height))))
        plt.xlabel('number of %s : %d'%(key, len(size_lst)))
        plt.ylabel('height')
        plt.scatter(range(len(height)), height)
        plt.savefig('%s/%s_height.jpg'%(save_path, key))
        plt.close()
        # plot size
        plt.title('size_median : %f, size_sean : %f, SD : %f'%(size_median ,size_mean, np.sqrt(np.var(size_lst))))
        plt.xlabel('number of %s : %d'%(key, len(size_lst)))
        plt.ylabel('size')
        plt.scatter(range(len(size_lst)), size_lst)
        plt.savefig('%s/%s_size.jpg'%(save_path, key))
        plt.close()
    ba.WriteJson(save_path_json, 'size_matter', val)
    #pdb.set_trace()


if __name__ == '__main__':
    file_lst = os.listdir(path)
    for ID in file_lst:
        all_obj(ID)
    size_matter(obj_struct)
    print 'Finish'
    #pdb.set_trace()
