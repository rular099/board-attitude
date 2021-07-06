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

class MadgwickN:
	"""
	Madgwick filter for sensor fusion of IMU

	The class fuses the roll, pitch and yaw from accelrometer
	and magneotmeter with gyroscope. 
	reference article : https://www.x-io.co.uk/res/doc/madgwick_internal_report.pdf
	refer to examples of the git repo

	"""
	def __init__(self, b = 0.1):
		"""
		Initialises all the variables. 

		The option of setting your own values is given in the form of 
		set functions

		"""
		
		GyroMeasError = np.pi * (40.0 / 180.0)
		self.beta = np.sqrt(3.0 / 4.0) * GyroMeasError
		# self.beta = b
		self.q = np.array([1.0, 0.0, 0.0, 0.0])
		self.roll = 0
		self.pitch = 0
		self.yaw = 0

	def computeOrientation(self, q):
		"""
		Computes euler angles from quaternion

		Parameter
		---------
		q: array containing quaternion vals

		"""

		self.yaw = np.degrees(np.arctan2(2*q[1]*q[2] + 2*q[0]*q[3],\
							 q[0]*q[0] + q[1]*q[1] - q[2]*q[2] -q[3]*q[3]))
		self.pitch = np.degrees(-1*np.arcsin(2*(q[1]*q[3] - q[0]*q[2])))
		self.roll = np.degrees(np.arctan2(2*q[0]*q[1] + 2*q[2]*q[3],\
								q[0]*q[0] + q[3]*q[3] - q[1]*q[1] - q[2]*q[2]))


	def quaternionMul(self, q1, q2):
		"""
		Provides quaternion multiplication

		Parameters
		----------
		q1: array containing quaternion vals
		q2: array containing quaternion vals

		Return
		------
		finalq: new quaternion obtained from q1*q2
		
		"""
		mat1 = np.array([[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]])
		mat2 = np.array([[0,0,1,0],[0,0,0,-1],[-1,0,0,0],[0,1,0,0]])
		mat3 = np.array([[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]])

		k1 = np.matmul(q1,mat1)[np.newaxis,:].T
		k2 = np.matmul(q1,mat2)[np.newaxis,:].T
		k3 = np.matmul(q1,mat3)[np.newaxis,:].T
		k0 = q1[np.newaxis,:].T

		mat = np.concatenate((k0,k1,k2,k3), axis = 1)

		finalq = np.matmul(mat,q2)

		return finalq

	def getAccelJacobian(self, q):

		jacob = np.array([[-2.0*q[2], 2.0*q[3], -2.0*q[0], 2.0*q[1]],\
						[2.0*q[1], 2.0*q[0], 2.0*q[3], 2.0*q[2]],\
						[0.0, -4.0*q[1], -4.0*q[2], 0.0]])
		return jacob

	def getAccelFunction(self, q, a):

		func = np.array([[2.0*(q[1]*q[3] - q[0]*q[2]) - a[1]],\
						[2.0*(q[0]*q[1] + q[2]*q[3]) - a[2]],\
						[2.0*(0.5 - q[1]*q[1] - q[2]*q[2]) - a[3]]])
		return func

	def normalizeq(self, q):
		"""
		Normalizing quaternion 

		Parameters
		----------
		q: array containing quaternion vals

		Return
		------
		q: Normalized quaternion
		
		"""

		qLength = np.sqrt(np.sum(np.square(q)))
		q = q/qLength
		return q

	def updateRollAndPitch(self, ax, ay, az, gx, gy, gz, dt):
		"""
		Computes roll and pitch

		Parameters
		----------
		ax: float 
			acceleration in x axis
		ay: float 
			acceleration in y axis
		az: float 
			acceleration in z axis
		gx: float 
			angular velocity about x axis
		gy: float 
			angular velocity about y axis
		dt: float
			time interval for kalman filter to be applied

		Note: It saves the roll and pitch in the class 
			properties itself. You can directly call them by
			classname.roll 

		"""

		g = np.array([0.0, gx, gy, gz])
		g = np.radians(g)
		qDot = 0.5*(self.quaternionMul(self.q,g))

		a = np.array([0.0, ax, ay, az])
		a = self.normalizeq(a)

		accelJacob = self.getAccelJacobian(self.q)
		accelF = self.getAccelFunction(self.q, a)

		deltaF = self.normalizeq(np.squeeze(np.matmul(accelJacob.T, accelF)))

		self.q = self.q + (qDot - self.beta*deltaF)*dt
		self.q = self.normalizeq(self.q)
		self.computeOrientation(self.q)

	def getMagJacob(self, q, b):

		magJacob = np.array([[-2*b[3]*q[2], 2*b[3]*q[3], -4*b[1]*q[2] -2*b[3]*q[0], -4*b[1]*q[3] +2*b[3]*q[1] ],\
							[-2*b[1]*q[3] +2*b[3]*q[1], 2*b[1]*q[2] +2*b[3]*q[0], 2*b[1]*q[1] +2*b[3]*q[3], -2*b[1]*q[0] +2*b[3]*q[2]],\
							[2*b[1]*q[2], 2*b[1]*q[3] -4*b[3]*q[1], 2*b[1]*q[0] -4*b[3]*q[2], 2*b[1]*q[1]]])
		return magJacob

	def getMagFunc(self, q, b, m):

		magFunc = np.array([[2*b[1]*(0.5 - q[2]*q[2] - q[3]*q[3]) +2*b[3]*(q[1]*q[3] - q[0]*q[2]) - m[1]],\
							[2*b[1]*(q[1]*q[2] - q[0]*q[3]) +2*b[3]*(q[0]*q[1] + q[2]*q[3]) - m[2]],\
							[2*b[1]*(q[0]*q[2] + q[1]*q[3]) +2*b[3]*(0.5 - q[1]*q[1] -q[2]*q[2]) -m[3]]])
		return magFunc

	def getRotationMat(self, q):

		rotMat = np.array([[2*q[0]*q[0] -1 + 2*q[1]*q[1], 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],\
						[2*(q[1]*q[2] + q[0]*q[3]), 2*q[0]*q[0] -1 + 2*q[2]*q[2], 2*(q[2]*q[3] - q[0]*q[1])],\
						[2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), 2*q[0]*q[0] -1 +2*q[3]*q[3]]])
		return rotMat

	def updateRollPitchYaw(self, ax, ay, az, gx, gy, gz, mx, my, mz, dt):
		"""
		Computes roll, pitch and yaw

		Parameters
		----------
		ax: float 
			acceleration in x axis
		ay: float 
			acceleration in y axis
		az: float 
			acceleration in z axis
		gx: float 
			angular velocity about x axis
		gy: float 
			angular velocity about y axis
		gz: float 
			angular velocity about z axis
		mx: float 
			magnetic moment about x axis
		my: float 
			magnetic moment about y axis
		mz: float 
			magnetic moment about z axis
		dt: float
			time interval for kalman filter to be applied

		Note: It saves the roll, pitch and yaw in the class 
			properties itself. You can directly call them by
			classname.roll 

		"""

		g = np.array([0.0, gx, gy, gz])
		g = np.radians(g)
		qDot = 0.5*(self.quaternionMul(self.q,g))

		a = np.array([0.0, ax, ay, az])
		a = self.normalizeq(a)

		accelJacob = self.getAccelJacobian(self.q)
		accelF = self.getAccelFunction(self.q, a)

		m = np.array([0.0, mx, my, mz])
		m = self.normalizeq(m)
		q_rot_mat = self.getRotationMat(self.q)
		h = np.matmul(q_rot_mat,m[1:])
		b = np.array([0.0, 1, 0.0, h[2]])
		b[1] = np.sqrt(np.sum(h[0]*h[0] + h[1]*h[1]))


		magJacob = self.getMagJacob(self.q, b)
		magFunc = self.getMagFunc(self.q, b, m)

		finalJacob = np.concatenate((accelJacob,magJacob), axis=0)
		finalFunc = np.concatenate((accelF, magFunc), axis=0)
		deltaF = self.normalizeq(np.squeeze(np.matmul(finalJacob.T, finalFunc)))

		self.q = self.q + (qDot - self.beta*deltaF)*dt
		self.q = self.normalizeq(self.q)
		self.computeOrientation(self.q)

	@property
	def roll(self):
		return self._roll

	@roll.setter
	def roll(self, roll):
		self._roll = roll

	@property
	def pitch(self):
		return self._pitch

	@pitch.setter
	def pitch(self, pitch):
		self._pitch = pitch

	@property
	def yaw(self):
		return self._yaw

	@yaw.setter
	def yaw(self, yaw):
		self._yaw = yaw

	@property
	def beta(self):
		return self._beta

	@beta.setter
	def beta(self, beta):
		if beta >= 0 and beta <= 1:
			self._beta = beta
		else:
			raise Exception("Please put beta value between 0 and 1")

	@property
	def q(self):
		return self._q

	@q.setter
	def q(self, q):
		if q is not None and q.shape[0] == 4:
			self._q = q
		else:
			raise Exception("q has to be a numpy array of 4 elements")

class MadgwickAHRS:
    sample_period = 1/256
    q = np.quaternion(1, 0, 0, 0)
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

        gyroscope = np.array(gyroscope, dtype=float).flatten()
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

        gyroscope = np.array(gyroscope, dtype=float).flatten()
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
