import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
proposed = np.loadtxt("/home/nuninu98/proposed.txt")
proposed_loop = np.loadtxt("/home/nuninu98/proposed_loop.txt")
proposed_map = o3d.io.read_point_cloud("/home/nuninu98/proposed_map.pcd")
proposed_map_points = np.asarray(proposed_map.points)
quadricslam = np.loadtxt('/home/nuninu98/quadricslam.txt')

orbslam = np.loadtxt('/home/nuninu98/orbslam.txt')
orbslam_loop = np.loadtxt('/home/nuninu98/orbslam_loop.txt')
orbslam_map = o3d.io.read_point_cloud("/home/nuninu98/orbslam_map.pcd")
orbslam_map_points = np.asarray(orbslam_map.points)

smslam = np.loadtxt('/home/nuninu98/data_saves/smslam_lcd.txt')
#proposed = np.asmatrix(proposed)
plt.figure(0)
plt.plot(proposed[:,1], proposed[:, 2], '-y', label="Trajectory", zorder=10)
#plt.plot(proposed_map_points[:,0], proposed_map_points[:, 1], '.k', label="test", markersize=0.1, alpha=0.8, zorder=0)


def plot_break_bar(y, top_y_range, bottom_y_range) :

    # 그래프 두 개를 한 figure 내에 그리기
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True) 
    # 두 그래프 사이의 상하 간격 설정
    fig.subplots_adjust(hspace = 0.1) 
 
    # 각각의 그래프 그리기 ax1이 위, ax2가 아래 임
    ax1.bar(np.arange(len(y)),y)
    ax2.bar(np.arange(len(y)),y)
 
    ax1.set_ylim(top_y_range[0], top_y_range[1]) # 윗쪽 그래프 y축 범위 설정
    ax2.set_ylim(bottom_y_range[0], bottom_y_range[1]) # 아랫쪽 그래프 y축 범위 설정
 
    # 지수형태로 label을 쓰지 않기 - 작은 숫자 안보이니까 
    ax2.ticklabel_format(axis='y', style='plain')
 
    # 두 그래프 사이의 경계선 제거
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()
 
    # 두 그래프 사이의 경계선 제거
    # ax1.yaxis.grid()
    # ax2.yaxis.grid()
 
    # 두 그래프 사이의 y축에 물결선 효과 마커 표시
    kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
 
for i in range(len(proposed_loop)):
    qid = int(proposed_loop[i, 0]) -1
    tid = int(proposed_loop[i, 1]) -1
    xo = proposed_loop[i, 2]
    yo = proposed_loop[i, 3]
    xq = proposed[qid,1]
    yq = proposed[qid,2]
    xt = proposed[tid,1]
    yt = proposed[tid,2]
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
# plt.xlim(-10, 80)
# plt.ylim(-10, 50)
plt.legend()

#plt.plot(quadricslam[:,0], quadricslam[:, 1], '-g', label='QuadricSLAM')
plt.figure(1)
plt.plot(orbslam[:,1], orbslam[:, 2], '-r', label='Trajectory', zorder=10)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('ORB SLAM3 Trajectory')
for i in range(len(orbslam_loop)):
    qid = int(orbslam_loop[i, 0]) -1
    tid = int(orbslam_loop[i, 1]) -1
    xq = orbslam[qid,1]
    yq = orbslam[qid,2]
    xt = orbslam[tid,1]
    yt = orbslam[tid,2]
    if i == 0:
        plt.plot([xq, xt], [yq, yt], '--b', label='Loop Closure')
    else:
         plt.plot([xq, xt], [yq, yt], '--b')   

plt.plot(orbslam_map_points[:,0], orbslam_map_points[:, 1], '.k', label="test", markersize=0.1, alpha=1.0, zorder=0)

plt.xlim(-10, 80)
plt.ylim(-10, 50)
plt.legend()

plt.figure(2)
plt.plot(smslam[:,0], smslam[:, 1], '-r', label='Trajectory')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('SmSLAM-LCD Trajectory')
plt.xlim(-5, 80)
plt.ylim(-5, 40)
plt.legend()

# plt.figure(3)
# loop_scores = np.loadtxt("/home/nuninu98/match_test/3020/scores.txt")
# ids = loop_scores[:,0]
# ids = np.asarray(ids, dtype=int)
# plt.plot(proposed[:,1], proposed[:, 2], '-r', label="Trajectory")
# plt.plot(proposed[ids, 1], proposed[ids, 2], 'yo', label='Loop Candidates')
# qid = 3020
# ki = 0
# for id in ids:
#     plt.text(proposed[id, 1], proposed[id, 2], ki)
#     for i in range(len(proposed_loop)):
#         if proposed_loop[i, 1] == id:
#             xo = proposed_loop[i, 2]
#             yo = proposed_loop[i, 3]
#             xq = proposed[id-1,1]
#             yq = proposed[id-1,2]
#             plt.plot(xo, yo, "m^", label='Unique Object')
#             plt.plot([xq, xo], [yq, yo], '--g', label='Visibility')
            
#     ki = ki + 1
# plt.plot(proposed[qid-1, 1], proposed[qid-1, 2], 'co', label='Query Keyframe')
# for i in range(len(proposed_loop)):
#     if proposed_loop[i, 0] == qid:
#         xo = proposed_loop[i, 2]
#         yo = proposed_loop[i, 3]
#         xq = proposed[qid-1,1]
#         yq = proposed[qid-1,2]
#         plt.plot(xo, yo, "m^")
#         plt.plot([xq, xo], [yq, yo], '--g')
#         break
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.legend()

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,  gridspec_kw={'height_ratios': [2, 1]})
# fig.subplots_adjust(hspace=0.05)  # adjust space between Axes

# # plot the same data on both Axes

# # zoom-in / limit the view to different portions of the data
# ax1.set_ylim(0.05, 1.0)  # outliers only
# ax2.set_ylim(0.009, 0.03)  # most of the data

# # hide the spines between ax and ax2
# ax1.spines.bottom.set_visible(False)
# ax2.spines.top.set_visible(False)
# #ax2.spines.bottom.set_visible(False)

# ax1.xaxis.tick_top()
# ax1.tick_params(labeltop=False, top=False) 

# # ax2.tick_params(
# #     axis='x',          # changes apply to the x-axis
# #     which='',      # both major and minor ticks are affected
# #     bottom=True,      # ticks along the bottom edge are off
# #     top=False,         # ticks along the top edge are off
# #     labelbottom=False)

# kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=12,
#                 linestyle="none", color='k', mec='k', mew=1, clip_on=False)
# ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
# ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

# ax1.plot(loop_scores[:, 1],'-ro', markersize=4, label='Ours')
# ax2.plot(loop_scores[:, 2], '-bo', markersize=4, label='VBoW')
# ax1.plot(loop_scores[:, 3], '-go', markersize=4, label='SmSLAM+LCD')
# ax1.plot(loop_scores[:, 4], '-co', markersize=4, label='GoogleNet')
# ax2.plot(loop_scores[:, 5], '-mo', markersize=4, label='SALAD')

# lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.suptitle('Loop Validation Score')
# fig.legend(lines, labels, prop={'size':10}, loc='upper left', bbox_to_anchor=(0.6, 0.9))
# fig.supxlabel('Keyframe')
# fig.supylabel('L2 Distance')
# plt.figure(5)
# plt.plot(loop_scores[:, 1],'-ro', label='Ours', markersize=4)
# plt.plot(loop_scores[:, 2], '-bo', label='VBoW', markersize=4)
# plt.plot(loop_scores[:, 3], '-go', label='SmSLAM+LCD', markersize=4)
# plt.plot(loop_scores[:, 4], '-co', label='GoogleNet', markersize=4)
# plt.plot(loop_scores[:, 5], '-mo', label='SALAD', markersize=4)
# plt.plot(loop_scores[:, 6], '-ko', label='MixVPR', markersize=4)
# plt.title('Loop Validation Score')
# plt.xlabel('Keyframe')
# plt.ylabel('L2 Distance')
# plt.legend(prop={'size':8}, loc='upper left', bbox_to_anchor=(0.7, 0.85))

plt.show()
