import numpy as np
import os
import quaternion
from scipy.spatial.transform import Rotation as R
from boardAttitude.filters import Madgwick
from boardAttitude.censor import Censor

if __name__ == '__main__':
    filename = './sample_data/RECORD_moderate.BIN'
    calibdata = './sample_data/calib.json'
    cs = Censor(filename,calibdata)
    data = cs.data

    res = np.zeros((len(data['index']),4))
    res_angle = np.zeros((len(data['index']),3))

    mad_filter = Madgwick(sample_period=1/50.,beta=0.1)
    mad_filter.q = np.quaternion(0,1,0,0) # a + bi + cj + dk
    for i in range(len(data['index'])):
        mad_filter.update(data['lo_gyro'][i],data['lo_acc'][i],data['lo_magnet'][i])
        res[i] = mad_filter.q.components
        r = R.from_quat(np.roll(mad_filter.q.components,-1)) # ai + bj + ck + d
        res_angle[i] = r.as_euler('xyz',degrees=True)[::-1]
    np.savetxt('./results/result_q.txt',res)
    np.savetxt('./results/result_angle_q.txt',res_angle)
