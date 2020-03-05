import numpy as np

# equation
# m\ddot{x} = u + f_ext

class SingleMassModel:
    def __init__(self):
        self.m = 1.0
        self.x = 0.0
        self.x_dot = 0.0
        self.x_ddot = 0.0

        self.model_dt = 1e-5

        self.f_ext = 0.0

        self.v = 0.001
        self.w = 0.01

    def updateOnce(self, u):
        self.x += self.x_dot * self.model_dt
        self.x_dot += self.x_ddot * self.model_dt
        self.x_ddot = (u + (self.f_ext + np.random.normal(0, self.v))) / self.m

    def updateModel(self, u, time):

        for i in range(int(time / self.model_dt)):
            self.updateOnce(u)

    def getObservation(self):
        return np.array([self.x + np.random.normal(0, self.w),
                         self.x_dot + np.random.normal(0, self.w)])

    def getGroundTruth(self):
        return np.array([self.x, self.x_dot])
