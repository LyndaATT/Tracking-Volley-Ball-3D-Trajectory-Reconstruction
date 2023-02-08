import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, dt, a_x,a_y, x_measurement_std, y_measurement_std, acceleration_std):
        """
        :param dt: time step
        :param a_x: acceleration in x-direction
        :param a_y: acceleration in y-direction
        :param x_measurement_std: standard deviation of the measurement (in x-direction)
        :param y_measurement_std: standard deviation of the measurement (in y-direction)
        :param acceleration_std: stadard deviation of the acceleration
        """

        # Time step
        self.dt = dt

        # Acceleration state (defined using acceleration wrt x-direction and y-direction)
        self.acc= np.matrix([[a_x],[a_y]])

        # Intial State
        self.x = np.matrix([[0], [0], [0], [0]])

        # Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],[0, 1, 0, self.dt],[0, 0, 1, 0],[0, 0, 0, 1]])


        # Control Input Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])


        # Mapping State to Measurement
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Process Covariance
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * acceleration_std**2

        # Measurement Covariance
        self.R = np.matrix([[x_measurement_std**2,0],
                           [0, y_measurement_std**2]])

        # Covariance Matrix, initialized to identity
        self.P = np.eye(self.A.shape[1])

    def predict(self):

        # State update
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.acc)

        # Error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]

    def update(self, z):

        # Updates
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Kalman Gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))   

        I = np.eye(self.H.shape[1])

        # Covariance matrix
        self.P = (I - (K * self.H)) * self.P   

        
        return self.x[0:2]