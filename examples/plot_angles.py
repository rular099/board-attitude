import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fname = './results/result_angle.txt'
    angles = np.loadtxt(fname)
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(311)
    ax.plot(angles[:,0],label='roll',marker='o',color='r')
    ax = fig.add_subplot(312)
    ax.plot(angles[:,1],label='pitch',marker='o',color='g')
    ax.hlines(90,xmin=0,xmax=len(angles))
    ax.hlines(-90,xmin=0,xmax=len(angles))
    ax = fig.add_subplot(313)
    ax.plot(angles[:,2],label='yaw',marker='o',color='b')
    fig.savefig('./results/euler_angles.png')
    plt.close('all')
