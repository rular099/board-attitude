import numpy as np
import os
from boardAttitude.censor import Censor
if __name__ == '__main__':
    filename = './sample_data/RECORD_nw.BIN'
    calibdata = './sample_data/calib.json'
    cs = Censor(filename,calibdata)
    np.savetxt('./results/record_angle.txt',cs.data['angles'][:,[2,1,0]])
    np.savetxt('./results/record_q.txt',cs.data['q'])
    np.savetxt('./results/record_gyro.txt',cs.data['lo_gyro'])
    np.savetxt('./results/record_acc.txt',cs.data['lo_acc'])
    np.savetxt('./results/record_mag.txt',cs.data['lo_magnet'])

