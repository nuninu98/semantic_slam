import numpy as np
import matplotlib.pyplot as plt
proposed = np.loadtxt("/home/nuninu98/proposed.txt")
proposed_loop = np.loadtxt("/home/nuninu98/proposed_loop.txt")
orbslam = np.loadtxt("/home/nuninu98/orbslam.txt")

gt = np.loadtxt("/home/nuninu98/gt.txt")
time_min = gt[0, 0]
time_max = gt[ np.shape(gt)[0] - 1,0]
print(time_min, time_max)

#proposed = np.asmatrix(proposed)


def linear_interpolation(val1, val2, time1, time2, time):
    return val1 + (val2 - val1)/(time2 - time1) * (time - time1)

def rmse(data, ground_truth):
    err_sum = 0.0
    cnt = 0
    max_err = 0.0
    for i in range(np.shape(data)[0]):
        stamp = data[i, 0]
        if stamp < time_min or stamp > time_max:
            continue
        id = 0
        while id < np.shape(ground_truth)[0] - 1:
            if stamp > ground_truth[id, 0] and stamp < ground_truth[id + 1, 0]:
                break
            id = id + 1
            # linear interpolation
        time1 = ground_truth[id, 0]
        time2 = ground_truth[id+ 1, 0]
        x1 = ground_truth[id, 1]
        x2 = ground_truth[id+1, 1]
        x_gt = linear_interpolation(x1, x2, time1, time2, stamp)

        y1 = ground_truth[id, 2]
        y2 = ground_truth[id+1, 2]
        y_gt = linear_interpolation(y1, y2, time1, time2, stamp)

        z1 = ground_truth[id, 3]
        z2 = ground_truth[id+1, 3]
        z_gt = linear_interpolation(z1, z2, time1, time2, stamp)

        err = (x_gt- data[i, 1])**2 +  (y_gt- data[i, 2])**2 + (z_gt- data[i, 3])**2
        if(err > max_err):
            max_err = err
        err_sum += err
        #rmse = np.sqrt(rmse)
        cnt = cnt + 1
    return np.sqrt(err_sum/cnt), np.sqrt(max_err)


print('RMSE Proposed: ', rmse(proposed, gt))
print('RMSE ORB-SLAM: ', rmse(orbslam, gt))

plt.figure(0)
plt.plot(proposed[:,1], proposed[:, 2], '-r', label="Proposed Method")
plt.plot(gt[:,1], gt[:, 2], '--k', label="Ground Truth")
plt.plot(orbslam[:,1], orbslam[:, 2], '-b', label="ORB-SLAM3")
plt.legend()
plt.title('Trajectory Comparison')
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.show()