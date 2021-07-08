import numpy as np
import os
import json
from json import JSONEncoder

class Censor:
    def __init__(self,datafile=None,calib_data='',sample_rate=50):
        self.sample_rate= sample_rate
        self.loadCalibData(calib_data)
        if datafile is not None:
            self.load_data(datafile)

    def load_data(self,filename='./sample_data/RECORD.BIN'):
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
            print(f"{RecordSize} records in file")
            data_orig = np.fromfile(fid,dtype=dtype)
        self.data = dict()
        self.data['index'] = data_orig['index']
        self.data['hi_temp'] = data_orig['hi_temp']/80.+25
        self.data['hi_gyro'] = data_orig['hi_gyro']* 0.00625/65536 # degree/sec
        self.data['hi_acc'] = data_orig['hi_acc']* 1.25/65536     # mg
        self.data['lo_gyro'] = data_orig['lo_gyro']* 2000./32768 #degree/sec
        self.data['lo_gyro'][:,1:3] = -self.data['lo_gyro'][:,1:3]
        self.data['lo_acc'] = data_orig['lo_acc']* 16000./32768 #mg
        self.data['lo_acc'][:,1:3] = -self.data['lo_acc'][:,1:3]
        self.data['lo_magnet'] = data_orig['lo_magnet'].astype(float)
        self.data['angles'] = data_orig['angles']* 180./32768 #degree
        self.data['q'] = data_orig['q']/32768. #degree
        self.apply_calib_info()

    def apply_calib_info(self):
        self.data['lo_acc'] = self.data['lo_acc']*self.Accels - self.AccelBias
        self.data['lo_gyro'] = self.data['lo_gyro'] - self.GyroBias
        self.data['lo_magnet'] = np.dot(self.data['lo_magnet'] - self.MagBias, self.Magtransform)

    def caliberateGyro(self,sequence=None):
        """Calibrates gyroscope by finding the bias sets the gyro bias

        """
        if sequence is None:
            sequence = self.data['lo_gyro']
        self.gyroBias = np.mean(sequence,axis=0)

    def caliberateAccelerometer(self,sequence=None,G=9.8):
        """Caliberate Accelerometer by positioning it in 6 different positions

        This function expects the user to keep the imu in 6 different positions while caliberation.
        It gives cues on when to change the position. It is expected that in all the 6 positions,
        at least one axis of IMU is parallel to gravity of earth and no position is same. Hence we
        get 6 positions namely -> +x, -x, +y, -y, +z, -z.
        """
        if sequence is None:
            sequence = self.data['lo_acc']
        xbias = []
        ybias = []
        zbias = []
        xscale = []
        yscale = []
        zscale = []

        xscale_p = np.mean(sequence[sequence[:,0]>9][:,0])
        yscale_p = np.mean(sequence[sequence[:,1]>9][:,1])
        zscale_p = np.mean(sequence[sequence[:,2]>9][:,2])
        xscale_m = np.mean(sequence[sequence[:,0]<-9][:,0])
        yscale_m = np.mean(sequence[sequence[:,1]<-9][:,1])
        zscale_m = np.mean(sequence[sequence[:,2]<-9][:,2])

        self.AccelBias = -G*np.array([(xscale_p + xscale_m)/(abs(xscale_p) + abs(xscale_m)),
                                        (yscale_p + yscale_m)/(abs(yscale_p) + abs(yscale_m)),
                                        (zscale_p + zscale_m)/(abs(zscale_p) + abs(zscale_m))])

        self.Accels = 2.0*G/np.array([abs(xscale_p) + abs(xscale_m),
                                      abs(yscale_p) + abs(yscale_m),
                                      abs(zscale_p) + abs(zscale_m)])

    def caliberateMag(self,sequence=None):
        """Caliberate Magnetometer Use this method for more precise calculation

        This function uses ellipsoid fitting to get an estimate of the bias and
        transformation matrix required for mag data

        Note: Make sure you rotate the sensor in 8 shape and cover all the
        pitch and roll angles.

        """
        if sequence is None:
            sequence = self.data['lo_magnet']
        centre, evecs, radii, v = self.__ellipsoid_fit(sequence)

        a, b, c = radii
        r = (a * b * c) ** (1. / 3.)
        D = np.array([[r/a, 0., 0.], [0., r/b, 0.], [0., 0., r/c]])
        transformation = evecs.dot(D).dot(evecs.T)

        self.MagBias = centre
        self.Magtransform = transformation

    def __ellipsoid_fit(self, X):
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        D = np.array([x * x + y * y - 2 * z * z,
                    x * x + z * z - 2 * y * y,
                    2 * x * y,
                    2 * x * z,
                    2 * y * z,
                    2 * x,
                    2 * y,
                    2 * z,
                    1 - 0 * x])
        d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
        u = np.linalg.solve(D.dot(D.T), D.dot(d2))
        a = np.array([u[0] + 1 * u[1] - 1])
        b = np.array([u[0] - 2 * u[1] - 1])
        c = np.array([u[1] - 2 * u[0] - 1])
        v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
        A = np.array([[v[0], v[3], v[4], v[6]],
                    [v[3], v[1], v[5], v[7]],
                    [v[4], v[5], v[2], v[8]],
                    [v[6], v[7], v[8], v[9]]])

        center = np.linalg.solve(- A[:3, :3], v[6:9])

        translation_matrix = np.eye(4)
        translation_matrix[3, :3] = center.T

        R = translation_matrix.dot(A).dot(translation_matrix.T)

        evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
        evecs = evecs.T

        radii = np.sqrt(1. / np.abs(evals))
        radii *= np.sign(evals)

        return center, evecs, radii, v

    def saveCalibData(self, filePath):
        """ Save the caliberation vaslues

        Parameters
        ----------
        filePath : str
            Make sure the folder exists before giving the input.  The path
            has to be absolute.
            Otherwise it doesn't save the values.

        """

        calibVals = {}
        calibVals['Accels'] = self.Accels
        calibVals['AccelBias'] = self.AccelBias
        calibVals['GyroBias'] = self.GyroBias
        calibVals['MagBias'] = self.MagBias
        if self.Magtransform is not None:
            calibVals['Magtransform'] = self.Magtransform

        with open(filePath, 'w') as outFile:
            json.dump(calibVals, outFile, cls =NumpyArrayEncoder)

    def loadCalibData(self, filename=''):
        """ Save the caliberation vaslues

        Parameters
        ----------
        filePath : str
            Make sure the file exists before giving the input. The path
            has to be absolute.
            Otherwise it doesn't save the values.

        """
        try:
            with open(filename, 'r') as jsonFile:
                calibVals = json.load(jsonFile)
                self.Accels = np.asarray(calibVals['Accels'])
                self.AccelBias = np.asarray(calibVals['AccelBias'])
                self.GyroBias = np.asarray(calibVals['GyroBias'])
                self.MagBias = np.asarray(calibVals['MagBias'])
                if 'Magtransform' in calibVals.keys():
                    self.Magtransform = np.asarray(calibVals['Magtransform'])
        except FileNotFoundError:
            print('No caliberation data provided. Using default caliberation value')
            self.Accels = np.array([1.,1.,1.])
            self.AccelBias = np.array([0.,0.,0.])
            self.GyroBias = np.array([0.,0.,0.])
            self.MagBias = np.array([0.,0.,0.])
            self.Magtransform = np.identity(3)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
