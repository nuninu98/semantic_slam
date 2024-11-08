import numpy as np
import matplotlib.pyplot as plt
proposed = np.loadtxt("/home/nuninu98/proposed.txt")
quadricslam = np.loadtxt('/home/nuninu98/quadricslam.txt')
orbslam = np.loadtxt('/home/nuninu98/orbslam.txt')
smslam = np.loadtxt('/home/nuninu98/smslam_lcd.txt')
#proposed = np.asmatrix(proposed)

plt.figure(0)
plt.plot(proposed[:,0], proposed[:, 1], '-r', )
plt.xlabel('x')
plt.ylabel('y')
plt.title('Proposed Method Trajectory')
plt.legend()

#plt.plot(quadricslam[:,0], quadricslam[:, 1], '-g', label='QuadricSLAM')
plt.figure(1)
plt.plot(orbslam[:,0], orbslam[:, 1], '-b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('ORB SLAM3 Trajectory')
plt.legend()

plt.figure(2)
plt.plot(smslam[:,0], smslam[:, 1], '-m')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SmSLAM-LCD Trajectory')
plt.legend()

plt.show()
