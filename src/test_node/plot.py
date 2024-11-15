import numpy as np
import matplotlib.pyplot as plt
proposed = np.loadtxt("/home/nuninu98/proposed.txt")
proposed_loop = np.loadtxt("/home/nuninu98/proposed_loop.txt")
quadricslam = np.loadtxt('/home/nuninu98/quadricslam.txt')

orbslam = np.loadtxt('/home/nuninu98/orbslam.txt')
orbslam_loop = np.loadtxt('/home/nuninu98/orbslam_loop.txt')

smslam = np.loadtxt('/home/nuninu98/smslam_lcd.txt')
#proposed = np.asmatrix(proposed)
plt.figure(0)
plt.plot(proposed[:,0], proposed[:, 1], '-r', label="Trajectory")
for i in range(len(proposed_loop)):
    qid = int(proposed_loop[i, 0]) -1
    tid = int(proposed_loop[i, 1]) -1
    xo = proposed_loop[i, 2]
    yo = proposed_loop[i, 3]
    xq = proposed[qid,0]
    yq = proposed[qid,1]
    xt = proposed[tid,0]
    yt = proposed[tid,1]
    if i == 0:
        plt.plot(xo, yo, "mo", label="Unique Objects")
        plt.plot([xq, xo], [yq, yo], '--b', label='Loop Closure')
    else:
        plt.plot(xo, yo, "mo")
        plt.plot([xq, xo], [yq, yo], '--b')
    
    plt.plot([xt, xo], [yt, yo], '--b')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Proposed Method')
plt.legend()

#plt.plot(quadricslam[:,0], quadricslam[:, 1], '-g', label='QuadricSLAM')
plt.figure(1)
plt.plot(orbslam[:,0], orbslam[:, 1], '-r', label='Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.title('ORB SLAM3 Trajectory')
for i in range(len(orbslam_loop)):
    qid = int(orbslam_loop[i, 0]) -1
    tid = int(orbslam_loop[i, 1]) -1
    xq = orbslam[qid,0]
    yq = orbslam[qid,1]
    xt = orbslam[tid,0]
    yt = orbslam[tid,1]
    if i == 0:
        plt.plot([xq, xt], [yq, yt], '--b', label='Loop Closure')
    else:
         plt.plot([xq, xt], [yq, yt], '--b')   
plt.legend()

plt.figure(2)
plt.plot(smslam[:,0], smslam[:, 1], '-m')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SmSLAM-LCD Trajectory')
plt.legend()

plt.show()
