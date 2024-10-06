import numpy as np
import matplotlib.pyplot as plt
sem_data = np.loadtxt('/home/nuninu98/loopscore/sem3596.txt')
bow_data = np.loadtxt('/home/nuninu98/loopscore/bow3596.txt')

plt.figure(0)
plt.plot(sem_data[:, 0], sem_data[:, 1], '.r')
plt.xlabel('Keyframe ID')
plt.ylabel('Loop Score')
plt.title('Uniqueness Semantic Matching Score')

plt.figure(1)
plt.plot(bow_data[:, 0], bow_data[:, 1], '.r')
plt.xlabel('Keyframe ID')
plt.ylabel('Loop Score')
plt.title('Bag of Words Matching Score')
plt.show()
