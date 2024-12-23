import numpy as np
import matplotlib.pyplot as plt
proposed = np.loadtxt("/home/nuninu98/proposed.txt")
proposed_loop = np.loadtxt("/home/nuninu98/proposed_loop.txt")
quadricslam = np.loadtxt('/home/nuninu98/quadricslam.txt')

orbslam = np.loadtxt('/home/nuninu98/data_saves/orbslam.txt')
orbslam_loop = np.loadtxt('/home/nuninu98/data_saves/orbslam_loop.txt')

smslam = np.loadtxt('/home/nuninu98/data_saves/smslam_lcd.txt')
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
        plt.plot(xo, yo, "m^", label="Unique Objects")
        plt.plot([xq, xo], [yq, yo], '--b', label='Loop Closure')
    else:
        plt.plot(xo, yo, "m^")
        plt.plot([xq, xo], [yq, yo], '--b')
    
    plt.plot([xt, xo], [yt, yo], '--b')

plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Proposed Method')
plt.xlim(-5, 80)
plt.ylim(-5, 40)
plt.legend()

#plt.plot(quadricslam[:,0], quadricslam[:, 1], '-g', label='QuadricSLAM')
plt.figure(1)
plt.plot(orbslam[:,0], orbslam[:, 1], '-r', label='Trajectory')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
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
plt.xlim(-5, 80)
plt.ylim(-5, 40)
plt.legend()

plt.figure(2)
plt.plot(smslam[:,0], smslam[:, 1], '-r', label='Trajectory')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('SmSLAM-LCD Trajectory')
plt.xlim(-5, 80)
plt.ylim(-5, 40)
plt.legend()

plt.figure(3)
loop_scores = np.loadtxt("/home/nuninu98/match_test/2940/scores.txt")
ids = loop_scores[:,0]
ids = np.asarray(ids, dtype=int)
plt.plot(proposed[:,0], proposed[:, 1], '-r', label="Trajectory")
plt.plot(proposed[ids, 0], proposed[ids, 1], 'yo', label='Loop Candidates')
qid = 2940
ki = 0
for id in ids:
    plt.text(proposed[id, 0], proposed[id, 1], ki)
    for i in range(len(proposed_loop)):
        if proposed_loop[i, 1] == id:
            xo = proposed_loop[i, 2]
            yo = proposed_loop[i, 3]
            xq = proposed[id-1,0]
            yq = proposed[id-1,1]
            plt.plot(xo, yo, "m^", label='Unique Object')
            plt.plot([xq, xo], [yq, yo], '--g', label='Visibility')
            
    ki = ki + 1
plt.plot(proposed[qid-1, 0], proposed[qid-1, 1], 'co', label='Query Keyframe')
for i in range(len(proposed_loop)):
    if proposed_loop[i, 0] == qid:
        xo = proposed_loop[i, 2]
        yo = proposed_loop[i, 3]
        xq = proposed[qid-1,0]
        yq = proposed[qid-1,1]
        plt.plot(xo, yo, "m^")
        plt.plot([xq, xo], [yq, yo], '--g')
        break
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()

plt.figure(4)
plt.plot(loop_scores[:, 1], label='Proposed Method')
plt.plot(1.0 - loop_scores[:, 2], label= 'VBoW Method')
plt.title('Loop Validation Score')
plt.legend()
plt.xlabel('Keyframe')
plt.ylabel('L2 Distance')
plt.show()
