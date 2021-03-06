# -*- coding: utf-8 -*-
"""
    Copyright (c) 2021 Zhang Bei, rular099@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import warnings
import numpy as np
from numpy.linalg import norm
import quaternion
from scipy.spatial.transform import Rotation as R

class Madgwick:
    sample_period = 1/256
    q = np.quaternion(0, 1, 0, 0)
    gyro_error = np.pi*5/180
    gyro_drift = np.pi*0.2/180
    beta = np.sqrt(3./4)*gyro_error
    zeta = np.sqrt(3./4)*gyro_drift
    b = np.array([0,1,0,0])

    def __init__(self, sample_period=None, q=None, beta=None, zeta=None):
        """
        Initialize the class with the given parameters.
        :param sample_period: The sample period
        :param q: Initial quaternion
        :param beta: Algorithm gain beta
        :return:
        """
        if sample_period is not None:
            self.sample_period = sample_period
        if q is not None:
            self.q = q
        if beta is not None:
            self.beta = beta
        if zeta is not None:
            self.zeta = zeta
        self.w_b = np.zeros(3)

    def update(self, gyroscope, accelerometer, magnetometer):
        """
        Perform one update step with data from a AHRS sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        :param magnetometer: A three-element array containing the magnetometer data. Can be any unit since a normalized value is used.
        :return:
        """
        q = self.q

        gyroscope = np.radians(np.array(gyroscope, dtype=float).flatten())
        accelerometer = np.array(accelerometer, dtype=float).flatten()
        magnetometer = np.array(magnetometer, dtype=float).flatten()

        # Normalise accelerometer measurement
        if np.isclose(norm(accelerometer),0):
            warnings.warn("accelerometer is zero")
            return
        accelerometer /= norm(accelerometer)

        # Normalise magnetometer measurement
        if np.isclose(norm(magnetometer) , 0):
            warnings.warn("magnetometer is zero")
            return
        magnetometer /= norm(magnetometer)

        h = q * (np.quaternion(0, magnetometer[0], magnetometer[1], magnetometer[2]) * q.conj())
        b = np.array([0, norm(h.components[1:3]), 0, h.components[3]])

        q_comp = q.components
        # objective function and Jacobian
        f = np.array([
            2*(q_comp[1]*q_comp[3] - q_comp[0]*q_comp[2]) - accelerometer[0],
            2*(q_comp[0]*q_comp[1] + q_comp[2]*q_comp[3]) - accelerometer[1],
            2*(0.5 - q_comp[1]**2 - q_comp[2]**2) - accelerometer[2],
            2*b[1]*(0.5 - q_comp[2]**2 - q_comp[3]**2) + 2*b[3]*(q_comp[1]*q_comp[3] - q_comp[0]*q_comp[2]) - magnetometer[0],
            2*b[1]*(q_comp[1]*q_comp[2] - q_comp[0]*q_comp[3]) + 2*b[3]*(q_comp[0]*q_comp[1] + q_comp[2]*q_comp[3]) - magnetometer[1],
            2*b[1]*(q_comp[0]*q_comp[2] + q_comp[1]*q_comp[3]) + 2*b[3]*(0.5 - q_comp[1]**2 - q_comp[2]**2) - magnetometer[2]
        ])
        j = np.array([
            [-2*q_comp[2],                  2*q_comp[3],                  -2*q_comp[0],                  2*q_comp[1]],
            [2*q_comp[1],                   2*q_comp[0],                  2*q_comp[3],                   2*q_comp[2]],
            [0,                        -4*q_comp[1],                 -4*q_comp[2],                  0],
            [-2*b[3]*q_comp[2],             2*b[3]*q_comp[3],             -4*b[1]*q_comp[2]-2*b[3]*q_comp[0], -4*b[1]*q_comp[3]+2*b[3]*q_comp[1]],
            [-2*b[1]*q_comp[3]+2*b[3]*q_comp[1], 2*b[1]*q_comp[2]+2*b[3]*q_comp[0], 2*b[1]*q_comp[1]+2*b[3]*q_comp[3],  -2*b[1]*q_comp[0]+2*b[3]*q_comp[2]],
            [2*b[1]*q_comp[2],              2*b[1]*q_comp[3]-4*b[3]*q_comp[1], 2*b[1]*q_comp[0]-4*b[3]*q_comp[2],  2*b[1]*q_comp[1]]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # estimate direction of the gyroscope error
        w_err = np.array([2*(q_comp[0]*step[1] - q_comp[1]*step[0] - q_comp[2]*step[3] + q_comp[3]*step[2]),
                          2*(q_comp[0]*step[2] + q_comp[1]*step[3] - q_comp[2]*step[0] - q_comp[3]*step[1]),
                          2*(q_comp[0]*step[3] - q_comp[1]*step[2] + q_comp[2]*step[1] - q_comp[3]*step[0])])

        # remove the gyroscope bais
        self.w_b += w_err*self.sample_period*self.zeta
        gyroscope -= self.w_b

        # Compute rate of change of quaternion
        qdot = (q * np.quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])) * 0.5 - self.beta * np.quaternion(*step.T)
#        qdot = - self.beta * np.quaternion(*step.T)

        # Integrate to yield quaternion
        q += qdot * self.sample_period
        self.q = q.normalized()  # normalise quaternion

    def update_imu(self, gyroscope, accelerometer):
        """
        Perform one update step with data from a IMU sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        """
        q = self.q

        gyroscope = np.radians(np.array(gyroscope, dtype=float).flatten())
        accelerometer = np.array(accelerometer, dtype=float).flatten()

        # Normalise accelerometer measurement
        if np.isclose(norm(accelerometer) , 0):
            warnings.warn("accelerometer is zero")
            return
        accelerometer /= norm(accelerometer)

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2]
        ])
        j = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Compute rate of change of quaternion
        qdot = (q * np.quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])) * 0.5 - self.beta * np.quaternion(*step.T)

        # Integrate to yield quaternion
        q += qdot * self.sample_period
        self.q = q.normalized()  # normalise quaternion

class Kalman:
    """
    Kalman filter for sensor fusion of IMU

    The class fuses the roll, pitch and yaw from accelrometer
    and magneotmeter with gyroscope.

    """
    def __init__(self):
        """
        Initialises all the variables.

        The option of setting your own values is given in the form of
        set functions

        """

        self.currentRollState = np.vstack((0.0, 0.0)) # updates
        self.roll = 0 # updates
        self.rollCovariance = np.zeros((2,2)) # updates
        self.rollError = 0.001
        self.rollDriftError = 0.001
        self.rollMeasurementError = 0.03

        self.currentPitchState = np.vstack((0.0, 0.0)) # updates
        self.pitch = 0 # updates
        self.pitchCovariance = np.zeros((2,2)) # updates
        self.pitchError= 0.001
        self.pitchDriftError = 0.001
        self.pitchMeasurementError = 0.03

        self.currentYawState = np.vstack((0.0, 0.0)) # updates
        self.yaw = 0 # updates
        self.yawCovariance = np.zeros((2,2)) #updates
        self.yawError = 0.001
        self.yawDriftError = 0.001
        self.yawMeasurementError = 0.03

    def computeAndUpdateRollPitchYaw(self, a, g, m, dt):
        """
        Computes roll, pitch and yaw

        Parameters
        ----------
        a: float
            acceleration in x,y,z axis
        g: float
            angular velocity about x,y,z axis
        mx: float
            magnetic moment about x,y,z axis
        dt: float
            time interval for kalman filter to be applied

        Note: It saves the roll, pitch and yaw in the class
            properties itself. You can directly call them by
            classname.roll

        """
        ax,ay,az = a
        #gx,gy,gz = g
        mx,my,mz = m
        rot_matrix = self.computeRotmatrix(a,m)
        r = R.from_matrix(rot_matrix)
        measuredRoll, measuredPitch, measuredYaw = r.as_euler('xyz',degrees=True)
#        measuredRoll, measuredPitch = self.computeRollAndPitch(a)
#        measuredYaw = self.computeYaw(measuredRoll, measuredPitch, m)
#        measuredRoll,measuredPitch, measuredYaw = self.__degimblock(measuredRoll,measuredPitch,measuredYaw)

#        reset, gy = self.__restrictRollAndPitch(measuredRoll, measuredPitch, gy)
        reset = 0
        gx,gy,gz = rot_matrix @ g
        if not reset:
            self.roll, self.currentRollState, self.rollCovariance = self.update(self.currentRollState, \
                                                                measuredRoll, self.rollCovariance, \
                                                                self.rollError, self.rollDriftError, \
                                                                self.rollMeasurementError, gx, dt)

        self.pitch, self.currentPitchState, self.pitchCovariance = self.update(self.currentPitchState, \
                                                                    measuredPitch, self.pitchCovariance, \
                                                                    self.pitchError, self.pitchDriftError, \
                                                                    self.pitchMeasurementError, gy, dt)

        self.yaw, self.currentYawState, self.yawCovariance = self.update(self.currentYawState, \
                                                            measuredYaw, self.yawCovariance, \
                                                            self.yawError, self.yawDriftError, \
                                                            self.yawMeasurementError, gz, dt)

    def __degimblock(self,measuredRoll,measuredPitch,measuredYaw):
        if np.isclose(measuredPitch,90.):
            tmp = measuredRoll - measuredYaw
            measuredYaw = 0
            measuredRoll = tmp
        return measuredRoll,measuredPitch,measuredYaw

    def __restrictRollAndPitch(self, measuredRoll, measuredPitch, gy):

        reset = 0
        if (measuredRoll < -90 and self.roll > 90) or (measuredRoll > 90 and self.roll < -90):
            self.roll = measuredRoll
            reset = 1
#        if abs(self.roll) > 90:
#            gy = -1*gy
        return reset, gy


    def computeAndUpdateRollPitch(self, a, g, dt):
        """
        Computes roll and pitch

        Parameters
        ----------
        a: float
            acceleration in x,y,z axis
        g: float
            angular velocity about x,y,z axis
        dt: float
            time interval for kalman filter to be applied

        Note: It saves the roll and pitch in the class
            properties itself. You can directly call them by
            classname.roll

        """
        ax,ay,az = a
        gx,gy,gz = g
        measuredRoll, measuredPitch = self.computeRollAndPitch(a)

        reset, gy = self.__restrictRollAndPitch(measuredRoll, measuredPitch, gy)

        if not reset:
            self.roll, self.currentRollState, self.rollCovariance = self.update(self.currentRollState, \
                                                                measuredRoll, self.rollCovariance, \
                                                                self.rollError, self.rollDriftError, \
                                                                self.rollMeasurementError, gx, dt)

        self.pitch, self.currentPitchState, self.pitchCovariance = self.update(self.currentPitchState, \
                                                                    measuredPitch, self.pitchCovariance, \
                                                                    self.pitchError, self.pitchDriftError, \
                                                                    self.pitchMeasurementError, gy, dt)

    def updateRollPitchYaw(self, roll, pitch, yaw, gx, gy, gz, dt):
        """
        Computes sensor fused roll, pitch and yaw

        Parameters
        ----------
        roll: float
            estimate obtained from accelerometer
        pitch: float
            estimate obtained from accelerometer
        yaw: float
            estimate obtained from magnetometer
        gx: float
            angular velocity about x axis
        gy: float
            angular velocity about y axis
        gz: float
            angular velocity about z axis
        dt: float
            time interval for kalman filter to be applied

        Note: It saves the roll, pitch and yaw in the class
            properties itself. You can directly call them by
            classname.roll

        """

        self.updateRollPitch(roll, pitch, gx, gy, dt)

        self.yaw, self.currentYawState, self.yawCovariance = self.update(self.currentYawState, \
                                                            yaw, self.yawCovariance, \
                                                            self.yawError, self.yawDriftError, \
                                                            self.yawMeasurementError, gz, dt)

    def updateRollPitch(self, roll, pitch, gx, gy, dt):
        """
        Computes sensor fused roll and pitch

        Parameters
        ----------
        roll: float
            estimate obtained from accelerometer
        pitch: float
            estimate obtained from accelerometer
        gx: float
            angular velocity about x axis
        gy: float
            angular velocity about y axis
        dt: float
            time interval for kalman filter to be applied

        Note: It saves the roll and pitch  in the class
            properties itself. You can directly call them by
            classname.roll

        """

        self.roll, self.currentRollState, self.rollCovariance = self.update(self.currentRollState, \
                                                                roll, self.rollCovariance, \
                                                                self.rollError, self.rollDriftError, \
                                                                self.rollMeasurementError, gx, dt)

        self.pitch, self.currentPitchState, self.pitchCovariance = self.update(self.currentPitchState, \
                                                                    pitch, self.pitchCovariance, \
                                                                    self.pitchError, self.pitchDriftError, \
                                                                    self.pitchMeasurementError, gy, dt)

    def computeRollAndPitch(self, a):
        """
        Computes measured roll and pitch from accelerometer

        Parameters
        ----------
        a: float
            acceleration in x,y,z axis

        Returns
        -------
        measuresRoll: float
                    It is estimated roll from sensor values
        measuresPitch: float
                    It is estimated pitch from sensor values

        """

        ax,ay,az = a
# original
#        measuredRoll = np.degrees(np.arctan2(ay,az))
#        measuredPitch = np.degrees(np.arctan2(-1*ax, np.sqrt(np.square(ay) + np.square(az)) ) )
# my method
#        measuredRoll = np.degrees(np.arctan2(-ay,az))
#        measuredPitch = np.degrees(np.arctan2(ax, np.sqrt(np.square(ay) + np.square(az)) ) )
# their method
        measuredRoll = -np.degrees(np.arctan2(ay,ax))
        measuredPitch = np.degrees(np.arctan2(np.sqrt(np.square(ax)+np.square(ay)),az))

        return measuredRoll, measuredPitch

    def computeYaw(self, roll, pitch, m):
        """
        Computes measured yaw

        Parameters
        ----------
        roll: float
            estimate obtained from accelerometer
        pitch: float
            estimate obtained from accelerometer
        m: float
            magnetic moment about x,y,z axis
        Returns
        -------
        measuresYaw: float
                    It is estimated yaw from sensor values

        """

        mx,my,mz = m
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        magLength = np.linalg.norm(m)
        mx = mx/magLength
        my = my/magLength
        mz = mz/magLength
#original
#        measuredYaw = np.degrees(np.arctan2(np.sin(roll)*mz - np.cos(roll)*mx,\
#                    np.cos(pitch)*mx + np.sin(roll)*np.sin(pitch)*my \
#                    + np.cos(roll)*np.sin(pitch)*mz) )
# my method
#        measuredYaw = np.degrees(np.arctan2(np.sin(roll)*mz + np.cos(roll)*my,
#                    np.cos(pitch)*mx + np.sin(roll)*np.sin(pitch)*my
#                    - np.cos(roll)*np.sin(pitch)*mz) )
# their method
        measuredYaw = np.degrees(np.arctan2(np.sin(roll)*mx + np.cos(roll)*my,
                    np.cos(pitch)*mz + np.sin(roll)*np.cos(pitch)*my
                    - np.cos(roll)*np.sin(pitch)*mx) )

        return measuredYaw

    def computeRotmatrix(self, a, m):
        rot_matrix = np.zeros((3,3))
        z_axis = a/np.linalg.norm(a)
        m = m/np.linalg.norm(m)
        y_axis = np.cross(z_axis,m)
        x_axis = np.cross(y_axis,z_axis)
        rot_matrix[0,:] = x_axis/np.linalg.norm(x_axis)
        rot_matrix[1,:] = y_axis/np.linalg.norm(y_axis)
        rot_matrix[2,:] = z_axis
        return rot_matrix

    def computeRollPitchYaw(self, a, m):
        rot_matrix = self.computeRotmatrix(a,m)
        r = R.from_matrix(rot_matrix)
        return r.as_euler('xyz',degrees=True)

    def update(self, currentState, measurement, currentCovariance, error, driftError, measurementError, angularVelocity ,dt):
        """
        Core function of Kalman relating to its implmentation

        Parameters
        ----------
        currentState: float array
                    It is current state of the sensor which implies current
                    orientation in a specific axis and its corresponding
                    bias. ex - [roll, roll_bias]
        measurement: float
            estimate of the orinetation by the sensor. ex - measuredRoll
        currentCovariance: 2*2 array
                        This represents matrix relating orientation and bias
                        ex - rollCovariance
        error: float
            This represents error in estimating the orientation
        driftError: float
                This represents error in estimating the  bias in orientation
        measurementError: float
                        This represents error in sensor values
        angularVelocity: float
                        The angular velocity about the direction
                        of orientation
        dt: float
            time interval for kalman filter to be applied

        Returns
        -------
        orientation: float
                    It is the corrected angle from previous
                    estimate and current measurment
        correctedState:
                    It is the corrected state from previous
                    estimate and current measurment
        updatedCovariance:
                    New updated covariance after taking
                    new measurement into consideration

        """

        motionModel = np.array([[1,-1*dt],[0,1]])

        prediction = np.matmul(motionModel,currentState) + dt*(np.vstack((angularVelocity,0.0)))
        if np.abs(prediction[0]-measurement) > 180:
            prediction[0] += np.sign(measurement - prediction[0])*360

        errorMatrix = np.array([error, driftError])*np.identity(2)
        predictedCovariance = np.matmul(np.matmul(motionModel, currentCovariance), (motionModel.T)) + errorMatrix

        difference = measurement - np.matmul(np.array([1.0, 1.0]), prediction)

        measurementCovariance = np.matmul(np.matmul(np.array([1.0, 0.0]), predictedCovariance),np.vstack((1.0,0.0))) + measurementError
        kalmanGain = np.matmul(predictedCovariance, np.vstack((1.0, 0.0)))/measurementCovariance

        correctedState = prediction + kalmanGain*(measurement - np.matmul(np.array([1.0, 0.0]), prediction))
#        correctedState = prediction

        updatedCovariance = np.matmul( np.identity(2) - np.matmul(kalmanGain, np.array([1.0, 0.0]).reshape((1,2))), predictedCovariance)

        return correctedState[0,0], correctedState, updatedCovariance

    @property
    def roll(self):
        return self._roll

    @roll.setter
    def roll(self, roll):
        self._roll = roll
        self.currentRollState[0,0] = roll

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, pitch):
        self._pitch = pitch
        self.currentPitchState[0,0] = pitch

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, yaw):
        self._yaw = yaw
        self.currentYawState[0,0] = yaw

    @property
    def rollError(self):
        return self._rollError

    @rollError.setter
    def rollError(self, error):
        self._rollError = error

    @property
    def rollDriftError(self):
        return self._rollDriftError

    @rollDriftError.setter
    def rollDriftError(self, error):
        self._rollDriftError = error


    @property
    def rollMeasurementError(self):
        return self._rollMeasurementError

    @rollMeasurementError.setter
    def rollMeasurementError(self, error):
        self._rollMeasurementError = error

    @property
    def pitchError(self):
        return self._pitchError

    @pitchError.setter
    def pitchError(self, error):
        self._pitchError = error

    @property
    def pitchDriftError(self):
        return self._pitchDriftError

    @pitchDriftError.setter
    def pitchDriftError(self, error):
        self._pitchDriftError = error

    @property
    def pitchMeasurementError(self):
        return self._pitchMeasurementError

    @pitchMeasurementError.setter
    def pitchMeasurementError(self, error):
        self._pitchMeasurementError = error

    @property
    def yawError(self):
        return self._yawError

    @yawError.setter
    def yawError(self, error):
        self._yawError = error

    @property
    def yawDriftError(self):
        return self._yawDriftError

    @yawDriftError.setter
    def yawDriftError(self, error):
        self._yawDriftError = error

    @property
    def yawMeasurementError(self):
        return self._yawMeasurementError

    @yawMeasurementError.setter
    def yawMeasurementError(self, error):
        self._yawMeasurementError = error
