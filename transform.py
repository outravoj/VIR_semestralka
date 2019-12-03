import numpy as np
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt


def transform_data(data):
    """
    normalizace jednotlivých bloku dat
    """
    n = data.shape[0]
    f = data.shape[2]
    t_data = np.zeros(data.shape)
    for i in range(n):
        if (data[i,0,0] == 0):
            continue
        off_x = np.mean(data[i,60,:])
        off_y = np.mean(data[i,61,:])
        ratio = 0.0
        for j in range(68):
            for s in range(f):
                t_data[i, 2*j, s] = -data[i, 2*j,s] + off_x
                t_data[i, 2*j+1, s] = -data[i, 2*j+1,s] + off_y
                if j in [7,9,21,22]:
                    ratio += np.sqrt((t_data[i,2*j,s] - t_data[i,60,s])**2 + (t_data[i,2*j+1,s] - t_data[i,61,s])**2)
        ratio = ratio / (4 * f)
        t_data[i,:,:] = t_data[i,:,:] / ratio
    return  t_data


def create_dataset(filename, output, number_of_frames=80, gen_step=55):
    data = np.load(filename).astype('f4')
    frame_num = data.shape[0]
    dataset = np.zeros((frame_num,2*68,number_of_frames))
    zero_mem = list()
    for i in range(frame_num):
        if np.max(data[i]) == np.min(data[i]):
            zero_mem.append(i)
    zero_mem.append(frame_num)
    last = -1
    i = 0
    while len(zero_mem) > 0:
        new = zero_mem.pop(0)
        if new-last-1 < number_of_frames:
            last = new
            continue
        elif new-last-1 == number_of_frames:
            dataset[i] = data[last+1:new,:].transpose()
            i+=1
            last = new
        else:
            curr = last+1
            while curr + number_of_frames <= new:
                if curr + number_of_frames >= 30000:
                    pass
                    print(curr)
                dataset[i] = data[curr:curr+number_of_frames,:].transpose()
                i+=1
                curr += gen_step

            if i > 29040:
                pass
            dataset[i] = data[new-number_of_frames:new,:].transpose()
            i+=1
            last = new
        #print(i)
    print("Nalezeno "+str(i))
    dataset = dataset[0:i,:,:]
    dataset = transform_data(dataset)
    print(dataset.shape)
    np.save(output, dataset.astype('f4'))
    print("hotovo")
