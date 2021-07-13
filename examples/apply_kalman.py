import numpy as np
import os
from boardAttitude.filters import Kalman
from boardAttitude.censor import Censor

if __name__ == '__main__':
    filename = './sample_data/RECORD_moderate.BIN'
    calibdata = './sample_data/calib.json'
    cs = Censor(filename,calibdata)
    data = cs.data
    dt = 1./cs.sample_rate

    res = np.zeros((len(data['index']),3))
    kalman_filter = Kalman()
#    kalman_filter.roll, kalman_filter.pitch = kalman_filter.computeRollAndPitch(data['lo_acc'][0])
#    kalman_filter.yaw = kalman_filter.computeYaw(kalman_filter.roll,
#                                                 kalman_filter.pitch,
#                                                 data['lo_magnet'][0])

    kalman_filter.roll,kalman_filter.pitch,kalman_filter.yaw = kalman_filter.computeRollPitchYaw(data['lo_acc'][0],data['lo_magnet'][0])
    res[0] = np.array([kalman_filter.yaw,kalman_filter.pitch,kalman_filter.roll])

    for i in range(len(data['index'])-1):
        kalman_filter.computeAndUpdateRollPitchYaw(data['lo_acc'][i+1],
                                                   data['lo_gyro'][i+1],
                                                   data['lo_magnet'][i+1],
                                                   dt)

#        kalman_filter.roll, kalman_filter.pitch = kalman_filter.computeRollAndPitch(data['lo_acc'][i+1])
#        kalman_filter.yaw = kalman_filter.computeYaw(kalman_filter.roll,
#                                                     kalman_filter.pitch,
#                                                     data['lo_magnet'][i+1])

#        kalman_filter.roll,kalman_filter.pitch,kalman_filter.yaw = kalman_filter.computeRollPitchYaw(data['lo_acc'][i+1],data['lo_magnet'][i+1])
        res[i+1] = np.array([kalman_filter.yaw,kalman_filter.pitch,kalman_filter.roll])
    np.savetxt('./results/result_angle.txt',res)
