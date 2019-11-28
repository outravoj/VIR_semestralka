import numpy as np
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt


def transform_data(filename, output):
    """
    znormalizuje data, prida sum
    """
    data = np.load(filename).astype('f4')
    t_data = np.zeros(data.shape).astype('f4')
    #video = cv2.VideoWriter(video_name,-1,frame_rate,(600,400))
    for i in range(data.shape[0]):
        if (data[i,0] == 0):
            continue

        for j in range(68):
            t_data[i, 2*j] = -data[i, 2*j] + data[i, 60]  # posunuti do 31 bodiku
            t_data[i, 2*j+1] = -data[i, 2*j+1] +data[i, 61]
        a_v = np.array([t_data[i, 16]-t_data[i, 60],t_data[i, 17]-t_data[i, 61]])
        b_v = np.array([t_data[i, 20]-t_data[i, 60],t_data[i, 21]-t_data[i, 61]])
        ration = 0.5 * (np.linalg.norm(a_v) + np.linalg.norm(b_v)) # normalizace
        #print(ration)
        t_data[i,:] = t_data[i,:] / ration + 0.001 * np.random.random(t_data[i].shape)
        #print(t_data[i])
    np.save(output, t_data)
    print("Hotovo")

def create_dataset(filename, output, number_of_frames=80, gen_step=55):
    """
    zpracuje data z vyse, vyhazi nuly, kdy se necetla zadna tvar, trochu prekryv, vysledek lze hodit do GAN
    """
    data = np.load(filename).astype('f4')
    frame_num = data.shape[0]
    dataset = np.zeros((frame_num,number_of_frames,2*68))
    zero_mem = list()
    print(data.shape)
    for i in range(frame_num):
        if np.max(data[i]) == np.min(data[i]):
            #print("nula {}".format(i))
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
            dataset[i] = data[last+1:new,:]
            i+=1
            last = new
        else:
            curr = last+1
            while curr + number_of_frames <= new:
                if curr + number_of_frames >= 30000:
                    pass
                    print(curr)
                dataset[i] = data[curr:curr+number_of_frames,:]
                i+=1
                curr += gen_step

            if i > 29040:
                pass
            dataset[i] = data[new-number_of_frames:new,:]
            i+=1
            last = new
        #print(i)
    print("Nalezeno "+str(i))
    for j in range(i):
        print(np.max(dataset[j]))
    np.save(output, dataset[0:i,:,:])
    print("hotovo")

