#import cv2
import numpy as np
#import dlib
import transform
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import time

'''
def detection(filename, output, output_video=False, output_name = "output.mov"):
    """
    zpracovava video filename, pokud filename = "", pouzije webku, radsi mensi rozliseni, jinak dlouho trva
    """
    if filename=="":
        cap = cv2.VideoCapture(0)
        length = 100000
    else:
        cap = cv2.VideoCapture(filename)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    [height, width] = cap.read()[1].shape[:2]
    video_frame_rate = 20 # nutno upravit pripad od pripadu
    desired_frame_rate = 20.0 #chcema 4 sekundy po 20 fps
    if output_video:
        #chceme-li vysledek ukladat i jako video
        video = cv2.VideoWriter(output_name,-1,desired_frame_rate,(width,height))

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    i = 0
    step = desired_frame_rate / video_frame_rate + 0.1
    counter = 0
    lim = 30000 #omezeni navic, klidne odstranit
    num_mem = np.zeros((min(length,lim), 68*2))
    while True:
        _, frame = cap.read()
        counter += step
        if counter < i:
            continue
        if i == lim or frame is None:
            break
        #print(frame.shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            #if width//2 > x2 or width//2 < x1: # omezeni treba kdyz je ve videu vic obliceju, aby bral jenom prostredni oblast
            #    continue
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            landmarks = predictor(gray, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                #reprezentujeme body jako vektor 2*68 ve formatu x0,y0,x1,y1,...
                num_mem[i,2*n] = x
                num_mem[i,2*n+1] = y
                cv2.circle(gray, (x, y), 4, (255, 0, 0), -1)
        i += 1
        if output_video:
            video.write(gray) #uklada video
        if (i%100) == 0:
            print("{0} out of {1}".format(i, length))
        cv2.imshow("Frame", gray) #output na obrazovku
        key = cv2.waitKey(1)
        if key == 97: #preruseni ackem, ale funguje asi jenom v consoli
            break
    num_mem = num_mem[0:i,:]
    np.save(output, num_mem)
    cv2.destroyAllWindows()
    cap.release()
    if output_video:
        video.release()
'''

def visualise(sample,number):
    """Funkcni, uklada 4s videa"""
    fig = plt.figure()
    ax = plt.axes(xlim=(-3, 3), ylim=(-3, 3))
    line, = ax.plot([], [], lw=3)
    line.set_data([], [])
    scat = ax.scatter([],[])

    def init():
        line.set_data([], [])
        return line,

    def animate(k):
        #x = np.random.random((68,1))
        #y = np.random.random((68,1))
        fin = np.zeros((sample.shape[0]//2,2))
        for i in range(sample.shape[0]//2):
            fin[i,0] = sample[2 * i,k]
            fin[i,1] = sample[2*i+1,k]
            #fin[i][0] = x[i]
            #fin[i][1] = y[i]

        scat.set_offsets(fin)
        return line,
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=80, interval=50, blit=True)

    anim.save('sequence_' +str(number)+'.gif', writer='imagemagick')
    print("amimace hotova")

def plot_curves(data):
    ep = np.arange(data.shape[1])
    r = data[0]
    f = data[1]
    g = data[2]

    plt.plot(ep, r, marker='', markerfacecolor='blue', linewidth=2, color='skyblue', label="real disk")
    plt.plot(ep, f, marker='', color='yellow', linewidth=2, label="fake disk")
    plt.plot(ep, g, marker='', color='green', linewidth=2, label="generator")
    plt.legend()
    plt.show()


if __name__=="__main__":
    #detection("", "data3", output_video=True, output_name="output3.mov")
    transform.create_dataset("data.npy", "correct_data_1")
    #sample2 = np.load("correct_data_1.npy")
    #sample = np.load("correct_data_1_reduced.npy")
    #visualise(sample2[355],6)
    #visualise(sample[355], 5)
    #print("hotovo")
