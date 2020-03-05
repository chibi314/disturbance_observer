import numpy as np
import model
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #model
    model = model.SingleMassModel()

    #param

    dt = 1e-2
    m = 1.0

    A = np.array([[1, dt, 0],
                  [0, 1, 1/m * dt],
                  [0, 0, 1]])

    B = np.array([0, 1/m * dt, 0])

    C = np.array([[1, 0, 0],
                  [0, 1, 0]])

    Q = np.diag((0, 0, 100))

    R = np.diag((0.01, 0.01))

    # variables

    x = np.zeros(3) #x, x_dot, f_ext

    P = np.zeros((3, 3))

    #input
    u = 0.2


    #log
    x_est_list = []
    x_dot_est_list = []
    x_gt_list = []
    x_dot_gt_list = []
    f_ext_list = []

    for i in range(int(1e3)):

        if i == 500:
            model.f_ext = 1.0

        #update model (simulation)
        model.updateModel(u, dt)
        z = model.getObservation()

        #kalman filter
        #prediction

        x_ = A.dot(x) + B * u

        P_ = A.dot(P).dot(A.transpose()) + Q

        #update

        G = P_.dot(C.transpose()).dot(C.dot(P_).dot(C.transpose()) + R)

        x = x_ + G.dot(z - C.dot(x_))

        P = (np.eye(3) - G.dot(C)).dot(P_)

        gt = model.getGroundTruth()
        x_est_list.append(x[0])
        x_dot_est_list.append(x[1])
        x_gt_list.append(gt[0])
        x_dot_gt_list.append(gt[1])
        f_ext_list.append(x[2])

    x = np.linspace(0, 10, 1000)

    plt.subplot(311)
    plt.plot(x, x_est_list, label="x_est")
    plt.plot(x, x_gt_list, label="x_gt")
    plt.legend()


    plt.subplot(312)
    plt.plot(x, x_dot_est_list, label="x_dot_est")
    plt.plot(x, x_dot_gt_list, label="x_dot_gt")
    plt.legend()

    plt.subplot(313)
    plt.plot(x, f_ext_list)

    plt.show()



