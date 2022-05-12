'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)



def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    # print(cov)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')
    # print('X',X)
    # print('P',P)
    plt.draw()
    plt.waitforbuttonpress(0)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad -= 2 * np.pi*np.floor((angle_rad+np.pi)/(2*np.pi))
    return angle_rad


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    TODO: initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    '''

    k = init_measure.shape[0] // 2

    landmark = np.zeros((2 * k, 1))
    landmark_cov = np.zeros((2 * k, 2 * k))
    beta = init_measure[::2]
    x_t = init_pose[0]
    y_t = init_pose[1]
    theta_t = init_pose[2]
    r = init_measure[1::2]
    l_x = []
    l_y = []
    for i in range(k):
        l_x = x_t+r[i]*(np.cos(beta[i]+theta_t))
        l_y = y_t+r[i]*(np.sin(beta[i]+theta_t))
        # landmark = np.vstack((l_x[i],l_y[i]))
        landmark[2*i] = l_x
        landmark[2*i+1] = l_y
        # print(landmark)
        #Pose Jacobian
        H_p = np.asarray([[1,0,-r[i]*np.sin(theta_t+beta[i])],
        [0,1,r[i]*np.cos(theta_t+beta[i])]],dtype=object)
        #Measured Jacobian
        H_l = np.asarray([[-r[i]*np.sin(beta[i]+theta_t), np.cos(beta[i]+theta_t)],
        [r[i]*np.cos(beta[i]+theta_t),np.sin(beta[i]+theta_t)]],dtype=object)
        H_l = H_l.reshape((2,2))
        measured_cov = H_l@init_measure_cov@H_l.T
        pose_cov = H_p@init_pose_cov@H_p.T
        landmark_cov[2*i:2*(i+1),2*i:2*(i+1)] = measured_cov + pose_cov 

    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''
    # X_pre = np.copy(X)
    # P_pre = np.copy(P)
    # Defining the variables
    x_t = X[0]
    y_t = X[1]
    theta_t = X[2]
    d_t = control[0]
    alpha_t = control[1]
    A = np.eye(3+2*k,3+2*k)
    rot = np.zeros((3+2*k,3+2*k))
    R_t1 = np.eye(3+2*k,3+2*k)
    A[0:3,0:3] = np.asarray([[1,0,-d_t*np.sin(theta_t)],[0,1,d_t*np.cos(theta_t)],[0,0,1]],dtype=object)
    R_t1[0:3,0:3] = block_diag(control_cov)
    rot[0:3,0:3] = np.asarray([[np.cos(theta_t), -np.sin(theta_t), 0],[np.sin(theta_t),np.cos(theta_t),0],[0,0,1]],dtype=object)
    P = (A @ P @ A.T) + (rot @ R_t1 @ rot.T)
    # print(P)
    pose = np.asarray([[x_t+d_t*np.cos(theta_t),y_t+d_t*np.sin(theta_t),theta_t+alpha_t]],dtype=object).T
    X[0:3] = pose.reshape(3,1)
    return X, P


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    TODO: update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    '''
    xt = X_pre[0]
    yt = X_pre[1]
    theta_t = X_pre[2]
    alpha_t = measure[1]
    Qt = block_diag(measure_cov,measure_cov,measure_cov,measure_cov,measure_cov,measure_cov)
    lx = []
    ly = []
    H_l = np.zeros((2*k,2*k))
    H_p = np.zeros((2*k,3))
    for idx in range(k):
        l_x = X_pre[3+2*idx]
        l_y = X_pre[3+2*idx+1]
        lx.append(l_x)
        ly.append(l_y)
    Ht = np.zeros((2*k,3+2*k))
    hu = np.zeros((2*k,1))
    for i in range(k):
        H_l[2*i:2*(i+1),2*i:2*(i+1)] = np.asarray([[(-ly[i]+yt)/((lx[i]-xt)**2 + (ly[i]-yt)**2), (-lx[i]+xt)/((lx[i]-xt)**2 + (ly[i]-yt)**2)],
        [(lx[i]-xt)/np.sqrt(((lx[i]-xt)**2 + (ly[i]-yt)**2)), (ly[i]-yt)/np.sqrt(((lx[i]-xt)**2 + (ly[i]-yt)**2))]],dtype=object).reshape(2,2)
        
        H_p[2*i:2*(i+1),0:3] = np.asarray([[(-yt+ly[i])/((ly[i]-yt)**2 + (lx[i]-xt)**2), (-xt+lx[i])/((ly[i]-yt)**2 + (lx[i]-xt)**2),-1],
        [(-lx[i]+xt)/np.sqrt((lx[i]-xt)**2 + (ly[i]-yt)**2), (-ly[i]+yt)/np.sqrt((lx[i]-xt)**2 + (ly[i]-yt)**2),0]],dtype=object)
        
        hu[2*i+1,0] = np.sqrt((lx[i]-xt)**2 +(ly[i]-yt)**2)
        hu[2*i,0] =  warp2pi(np.arctan2(ly[i]-yt,lx[i]-xt)-theta_t)
    Ht = np.hstack((H_p,H_l))
    Kt = P_pre @ Ht.T @ np.linalg.pinv((Ht @ P_pre @ Ht.T) + Qt)
    X_pre += Kt @ (measure-hu)
    P_pre = (np.eye(2*k+3) - Kt @ Ht) @ P_pre
    return X_pre, P_pre

def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    l_true = np.array([3,6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    l_true = l_true.reshape((12,1))
    X_arr = X[3:,:]
    diff = (-X_arr+l_true)
    euc_dist = np.sqrt(diff[::2]**2 + diff[1::2]**2)
    mahalanobis_dist = np.zeros((k,1))
    mahalanobis = 0

    for i in range(k):
        diff_new = diff.reshape((6,2))[i,:]
        P_maha = P[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
        mahalanobis_dist[i] = np.sqrt(diff_new @ P_maha @ diff_new.T)

    print('Euclidean Distances: ',euc_dist)
    print('Mahalanobis Distances: ',mahalanobis_dist)
 
    plt.scatter(l_true[0::2], l_true[1::2],cmap = 'magenta',linewidths=4)
    plt.draw()
    plt.waitforbuttonpress(0)


def main():
    # TEST: Setup uncertainty parameters
    #Each To be multiplied by 10 at a time for part 3.2
    sig_x = 0.25
    sig_y = 0.1
    sig_alpha = 0.1
    sig_beta = 0.01
    sig_r = 0.08


    # Generate variance from standard deviation
    sig_x2 = (sig_x**2)
    sig_y2 = (sig_y**2)
    sig_alpha2 = (sig_alpha**2)
    sig_beta2 = (sig_beta**2)
    sig_r2 = (sig_r**2)

    # Open data file and read the initial measurements
    data_file = open("../data/data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov)

    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])

    # Plot initial state and covariance
    last_X = np.copy(X)
    draw_traj_and_map(X, last_X, P, 0)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = np.copy(X)
            t += 1

    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)


if __name__ == "__main__":
    main()
