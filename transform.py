import numpy as np
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt

REDUCED_SET_INDICES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 20, 22, 23, 25, 27, 28, 30, 31, 32, 34, 36, 37, 38, 39, 40,
                       41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,52, 53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68]
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


def create_reduced_dataset(dataset):
    """
    vytvoří dataset s redukovaným počtem feature
    """
    reduced_dataset = np.zeros((dataset.shape[0], 2*len(REDUCED_SET_INDICES), dataset.shape[2]))
    offset_indices = np.array(REDUCED_SET_INDICES) - 1
    for i in range(dataset.shape[0]):
        for j in range(len(REDUCED_SET_INDICES)):
            #print(offset_indices[j])
            reduced_dataset[i,2*j,:] = dataset[i,2*offset_indices[j],:]
            reduced_dataset[i, 2*j+1,:] = dataset[i, 2*offset_indices[j]+1,:]
    return reduced_dataset


def create_dataset(filename, output, number_of_frames=80, gen_step=55):
    data = np.load("datasets/verca1.npy").astype('f4')
    data2 = np.load("datasets/verca2.npy").astype('f4')
    data3 = np.load("datasets/verca3.npy").astype('f4')
    print("Shapes of my data: ",data.shape,data2.shape,data3.shape)
    data = np.concatenate((data,data2,data3),0)
    print("Shapes of my data: ", data.shape)
    frame_num = data.shape[0]
    dataset = np.zeros((frame_num//80,2*68,number_of_frames))
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
                if curr + number_of_frames >= 300000:
                    pass
                    print(curr)
                dataset[i] = data[curr:curr+number_of_frames,:].transpose()
                i+=1
                curr += gen_step
            dataset[i] = data[new-number_of_frames:new,:].transpose()
            i+=1
            last = new
        #print(i)
    print("Nalezeno "+str(i))
    dataset = dataset[0:i,:,:]
    dataset = transform_data(dataset)
    print(dataset.shape)
    rdataset = create_reduced_dataset(dataset)
    np.save(output, dataset.astype('f4'))
    print(rdataset.shape)
    np.save(output + "_reduced",rdataset.astype('f4'))
    print("hotovo")

if __name__ == '__main__':
    """
    je třeba rozlišovat mezi input se 136 a 106 kanály - input resp reduced_input, vznikají oba v transform.py
    """
    create_dataset("verca1.npy","ultimate_dataset")
