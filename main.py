from madgwickahrs import MadgwickAHRS
import numpy as np
import os

def load_data(filename='./RECORD.BIN'):
    names = ['head','index',
             'hi_temp','hi_gyro','hi_acc',
             'lo_gyro','lo_acc','lo_magnet',
             'angles','q']
    formats = ['<H','<I',
               '<h','3<i','3<i',
               '3<h','3<h','3<h',
               '3<h','4<h']
    dtype = np.dtype({'names':names,'formats':formats})
    with open(filename,'rb') as fid:
        filelength = os.path.getsize(filename)
        # skip the first 13 data record
        fid.seek(13*64)
        RecordSize = int(filelength/64-13)
        print(RecordSize)
        data_orig = np.fromfile(fid,dtype=dtype)
    data = dict()
    data['index'] = data_orig['index']
    data['hi_temp'] = data_orig['hi_temp']/80.+25
    data['hi_gyro'] = data_orig['hi_gyro']* 0.00625/65536 # degree/sec
    data['hi_acc'] = data_orig['hi_acc']* 1.25/65536     # mg
    data['lo_gyro'] = data_orig['lo_gyro']* 2000./32768 #degree/sec
    data['lo_gyro'][:,1:3] = -data['lo_gyro'][:,1:3]
    data['lo_acc'] = data_orig['lo_acc']* 16000./32768 #mg
    data['lo_acc'][:,1:3] = -data['lo_acc'][:,1:3]
    data['lo_magnet'] = data_orig['lo_magnet'].astype(float)
    data['angles'] = data_orig['angles']* 180./32768 #degree
    data['q'] = data_orig['q']/32768. #degree
    return data

if __name__ == '__main__':
    filename = './RECORD.BIN'
    data = load_data(filename)

    res = np.zeros((len(data['index']),4))
    mad_filter = MadgwickAHRS()
    for i in range(len(data['index'])):
        mad_filter.update(data['lo_gyro'][i],data['lo_acc'][i],data['lo_magnet'][i])
        res[i] = mad_filter.q.components
    np.savetxt('result.txt',res)
