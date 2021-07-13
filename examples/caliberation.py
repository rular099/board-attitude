import numpy as np
import os
import matplotlib.pyplot as plt
from boardAttitude.censor import Censor
if __name__ == '__main__':
    acc_file = './sample_data/RECORD_acc.BIN'
    gyro_file = './sample_data/RECORD_gyro.BIN'
    mag_file = './sample_data/RECORD_mag.BIN'
    calib_file = './sample_data/calib.json'
    cs = Censor()
    cs.load_data(gyro_file)
    for key in cs.data.keys():
        cs.data[key] = cs.data[key][3*cs.sample_rate:-3:cs.sample_rate]
    cs.caliberateGyro()
    cs.load_data(mag_file)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.scatter(cs.data['lo_magnet'][:,0],cs.data['lo_magnet'][:,1],color='r',label='x-y')
    ax.scatter(cs.data['lo_magnet'][:,1],cs.data['lo_magnet'][:,2],color='g',label='y-z')
    ax.scatter(cs.data['lo_magnet'][:,0],cs.data['lo_magnet'][:,2],color='b',label='x-z')
    fig.savefig('./results/mag_before.png')
    plt.close('all')
    cs.caliberateMag()
    cs.apply_calib_info()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.scatter(cs.data['lo_magnet'][:,0],cs.data['lo_magnet'][:,1],color='r',label='x-y')
    ax.scatter(cs.data['lo_magnet'][:,1],cs.data['lo_magnet'][:,2],color='g',label='y-z')
    ax.scatter(cs.data['lo_magnet'][:,0],cs.data['lo_magnet'][:,2],color='b',label='x-z')
    fig.savefig('./results/mag_after.png')
    plt.close('all')
#    cs.load_data(acc_file)
#    cs.caliberateAccelerometer()
    cs.saveCalibData(calib_file)
