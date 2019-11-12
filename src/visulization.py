"""
    Created on Jan 20, 2019
    @author: Yixin Chen
    Visualization function.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import world2camera, pose3d_all2partial
import numpy as np
import cv2
import os


def draw_3dline_bdb(ax, ind, single_bdb, color='C1'):
    ax.plot([single_bdb[ind[0], 0], single_bdb[ind[1], 0]],
            [single_bdb[ind[0], 1], single_bdb[ind[1], 1]],
            [single_bdb[ind[0], 2], single_bdb[ind[1], 2]],
            color=color, linewidth=3.0)


def draw_3dbdb(ax, single_bdb, color='C1'):
    draw_3dline_bdb(ax, [0, 1], single_bdb, color,)
    draw_3dline_bdb(ax, [1, 2], single_bdb, color)
    draw_3dline_bdb(ax, [2, 3], single_bdb, color)
    draw_3dline_bdb(ax, [3, 0], single_bdb, color)

    draw_3dline_bdb(ax, [4, 5], single_bdb, color)
    draw_3dline_bdb(ax, [5, 6], single_bdb, color)
    draw_3dline_bdb(ax, [6, 7], single_bdb, color)
    draw_3dline_bdb(ax, [7, 4], single_bdb, color)

    draw_3dline_bdb(ax, [0, 4], single_bdb, color)
    draw_3dline_bdb(ax, [1, 5], single_bdb, color)
    draw_3dline_bdb(ax, [2, 6], single_bdb, color)
    draw_3dline_bdb(ax, [3, 7], single_bdb, color)
    draw_vertex = False
    if draw_vertex:
        for _ind, _vertex in enumerate(single_bdb):
            ax.text(_vertex[0], _vertex[1], _vertex[2], str(_ind))


def draw_3dpose(ax, pose_3d):
    _CONNECTION = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
        [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
        [15, 16]]
    for conn in _CONNECTION:
        draw_3dline_bdb(ax, conn, pose_3d, color='blue')


def draw_3dpose_sunrgbd(ax, pose_3d):
    _CONNECTION = [
        [3, 2], [2, 4], [2, 8], [2, 1], [1, 0], [0, 12], [0, 16], [4, 5],
        [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
        [12, 13], [13, 14], [14, 15], [16, 17], [17, 18], [18, 19]]
    for conn in _CONNECTION:
        draw_3dline_bdb(ax, conn, pose_3d)


def vis_scene(layout, bdb, pose):
    fig = plt.figure()
    ax = Axes3D(fig)
    # draw room layout
    # note here bottom points is actually the roof
    maxhight = 1.2
    totalpoints = layout.shape[0]
    bottompoints = layout[0: int(totalpoints / 2), :]

    toppoints = layout[int(totalpoints / 2): totalpoints, :]
    toppoints[toppoints[:, 2] > maxhight, 2] = maxhight
    bottompoints[bottompoints[:, 2] > maxhight, 2] = maxhight
    ind = np.argmin(toppoints[:, 1])

    for _iter in range(toppoints.shape[0]-1):
        if _iter != ind and _iter+1 != ind:
            ax.plot([toppoints[_iter, 0], toppoints[_iter + 1, 0]],
                    [toppoints[_iter, 1], toppoints[_iter + 1, 1]],
                    [toppoints[_iter, 2], toppoints[_iter + 1, 2]])
            ax.plot([bottompoints[_iter, 0], bottompoints[_iter + 1, 0]],
                    [bottompoints[_iter, 1], bottompoints[_iter + 1, 1]],
                    [bottompoints[_iter, 2], bottompoints[_iter + 1, 2]])
            ax.plot([toppoints[_iter, 0], bottompoints[_iter, 0]],
                    [toppoints[_iter, 1], bottompoints[_iter, 1]],
                    [toppoints[_iter, 2], bottompoints[_iter, 2]])

    if ind != 0 and ind != toppoints.shape[0]-1:
        ax.plot([toppoints[-1, 0], toppoints[0, 0]],
                [toppoints[-1, 1], toppoints[0, 1]],
                [toppoints[-1, 2], toppoints[0, 2]])
        ax.plot([bottompoints[-1, 0], bottompoints[0, 0]],
                [bottompoints[-1, 1], bottompoints[0, 1]],
                [bottompoints[-1, 2], bottompoints[0, 2]])
        ax.plot([toppoints[-1, 0], bottompoints[-1, 0]],
                [toppoints[-1, 1], bottompoints[-1, 1]],
                [toppoints[-1, 2], bottompoints[-1, 2]])

    # draw 3d bdb
    for single_bdb in bdb:
        draw_3dbdb(ax, single_bdb)

    # draw 3d pose
    draw_3dpose(ax, pose)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')


def vis_scene_w_ax(layout, bdb, pose, ax):
    pass
    # draw room layout
    # note here bottom points is actually the roof
    maxhight = 1.2
    totalpoints = layout.shape[0]
    bottompoints = layout[0: int(totalpoints / 2), :]

    toppoints = layout[int(totalpoints / 2): totalpoints, :]
    toppoints[toppoints[:, 2] > maxhight, 2] = maxhight
    bottompoints[bottompoints[:, 2] > maxhight, 2] = maxhight
    ind = np.argmin(toppoints[:, 1])

    for _iter in range(toppoints.shape[0] - 1):
        if _iter != ind and _iter + 1 != ind:
            ax.plot([toppoints[_iter, 0], toppoints[_iter + 1, 0]],
                    [toppoints[_iter, 1], toppoints[_iter + 1, 1]],
                    [toppoints[_iter, 2], toppoints[_iter + 1, 2]])
            ax.plot([bottompoints[_iter, 0], bottompoints[_iter + 1, 0]],
                    [bottompoints[_iter, 1], bottompoints[_iter + 1, 1]],
                    [bottompoints[_iter, 2], bottompoints[_iter + 1, 2]])
            ax.plot([toppoints[_iter, 0], bottompoints[_iter, 0]],
                    [toppoints[_iter, 1], bottompoints[_iter, 1]],
                    [toppoints[_iter, 2], bottompoints[_iter, 2]])

    if ind != 0 and ind != toppoints.shape[0] - 1:
        ax.plot([toppoints[-1, 0], toppoints[0, 0]],
                [toppoints[-1, 1], toppoints[0, 1]],
                [toppoints[-1, 2], toppoints[0, 2]])
        ax.plot([bottompoints[-1, 0], bottompoints[0, 0]],
                [bottompoints[-1, 1], bottompoints[0, 1]],
                [bottompoints[-1, 2], bottompoints[0, 2]])
        ax.plot([toppoints[-1, 0], bottompoints[-1, 0]],
                [toppoints[-1, 1], bottompoints[-1, 1]],
                [toppoints[-1, 2], bottompoints[-1, 2]])

    # draw 3d bdb
    for single_bdb in bdb:
        draw_3dbdb(ax, single_bdb)

    # draw 3d pose
    draw_3dpose(ax, pose)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')


def vis_scene_w_ax_multi(layout, bdb, pose, ax):
    pass
    # draw room layout
    # note here bottom points is actually the roof
    maxhight = 1.2
    totalpoints = layout.shape[0]
    bottompoints = layout[0: int(totalpoints / 2), :]

    toppoints = layout[int(totalpoints / 2): totalpoints, :]
    toppoints[toppoints[:, 2] > maxhight, 2] = maxhight
    bottompoints[bottompoints[:, 2] > maxhight, 2] = maxhight
    ind = np.argmin(toppoints[:, 1])

    for _iter in range(toppoints.shape[0] - 1):
        if _iter != ind and _iter + 1 != ind:
            ax.plot([toppoints[_iter, 0], toppoints[_iter + 1, 0]],
                    [toppoints[_iter, 1], toppoints[_iter + 1, 1]],
                    [toppoints[_iter, 2], toppoints[_iter + 1, 2]])
            ax.plot([bottompoints[_iter, 0], bottompoints[_iter + 1, 0]],
                    [bottompoints[_iter, 1], bottompoints[_iter + 1, 1]],
                    [bottompoints[_iter, 2], bottompoints[_iter + 1, 2]])
            ax.plot([toppoints[_iter, 0], bottompoints[_iter, 0]],
                    [toppoints[_iter, 1], bottompoints[_iter, 1]],
                    [toppoints[_iter, 2], bottompoints[_iter, 2]])

    if ind != 0 and ind != toppoints.shape[0] - 1:
        ax.plot([toppoints[-1, 0], toppoints[0, 0]],
                [toppoints[-1, 1], toppoints[0, 1]],
                [toppoints[-1, 2], toppoints[0, 2]])
        ax.plot([bottompoints[-1, 0], bottompoints[0, 0]],
                [bottompoints[-1, 1], bottompoints[0, 1]],
                [bottompoints[-1, 2], bottompoints[0, 2]])
        ax.plot([toppoints[-1, 0], bottompoints[-1, 0]],
                [toppoints[-1, 1], bottompoints[-1, 1]],
                [toppoints[-1, 2], bottompoints[-1, 2]])

    # draw 3d bdb
    for single_bdb in bdb:
        draw_3dbdb(ax, single_bdb)

    # draw 3d pose
    if pose is not None:
        for single_pose in pose:
            draw_3dpose(ax, single_pose)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')


def draw_3d_bbx_in2d(image, single_bdb, color=(42, 42, 165), width=13):
    cv2.line(image, tuple(single_bdb[0]), tuple(single_bdb[1]), color, width)
    cv2.line(image, tuple(single_bdb[1]), tuple(single_bdb[2]), color, width)
    cv2.line(image, tuple(single_bdb[2]), tuple(single_bdb[3]), color, width)
    cv2.line(image, tuple(single_bdb[3]), tuple(single_bdb[0]), color, width)

    cv2.line(image, tuple(single_bdb[4]), tuple(single_bdb[5]), color, width)
    cv2.line(image, tuple(single_bdb[5]), tuple(single_bdb[6]), color, width)
    cv2.line(image, tuple(single_bdb[6]), tuple(single_bdb[7]), color, width)
    cv2.line(image, tuple(single_bdb[7]), tuple(single_bdb[4]), color, width)

    cv2.line(image, tuple(single_bdb[0]), tuple(single_bdb[4]), color, width)
    cv2.line(image, tuple(single_bdb[1]), tuple(single_bdb[5]), color, width)
    cv2.line(image, tuple(single_bdb[2]), tuple(single_bdb[6]), color, width)
    cv2.line(image, tuple(single_bdb[3]), tuple(single_bdb[7]), color, width)


def draw_2d_bbx(image, single_bdb, color=(0, 255, 0)):
    # cv2: [width height]
    cv2.line(image, tuple([single_bdb[0], single_bdb[1]]), tuple([single_bdb[0], single_bdb[3]]), color, 3)
    cv2.line(image, tuple([single_bdb[0], single_bdb[1]]), tuple([single_bdb[2], single_bdb[1]]), color, 3)
    cv2.line(image, tuple([single_bdb[0], single_bdb[3]]), tuple([single_bdb[2], single_bdb[3]]), color, 3)
    cv2.line(image, tuple([single_bdb[2], single_bdb[1]]), tuple([single_bdb[2], single_bdb[3]]), color, 3)


def draw_2dpose(image, pose_2d_pred, color=(255, 0, 0), width=13):

    cv2.line(image, tuple(pose_2d_pred[0]), tuple(pose_2d_pred[1]), color, width)
    cv2.line(image, tuple(pose_2d_pred[1]), tuple(pose_2d_pred[2]), color, width)
    cv2.line(image, tuple(pose_2d_pred[2]), tuple(pose_2d_pred[3]), color, width)
    cv2.line(image, tuple(pose_2d_pred[3]), tuple(pose_2d_pred[4]), color, width)
    cv2.line(image, tuple(pose_2d_pred[1]), tuple(pose_2d_pred[5]), color, width)
    cv2.line(image, tuple(pose_2d_pred[5]), tuple(pose_2d_pred[6]), color, width)
    cv2.line(image, tuple(pose_2d_pred[6]), tuple(pose_2d_pred[7]), color, width)
    cv2.line(image, tuple(pose_2d_pred[1]), tuple(pose_2d_pred[8]), color, width)
    cv2.line(image, tuple(pose_2d_pred[8]), tuple(pose_2d_pred[9]), color, width)
    cv2.line(image, tuple(pose_2d_pred[9]), tuple(pose_2d_pred[10]), color, width)
    cv2.line(image, tuple(pose_2d_pred[1]), tuple(pose_2d_pred[11]), color, width)
    cv2.line(image, tuple(pose_2d_pred[11]), tuple(pose_2d_pred[12]), color, width)
    cv2.line(image, tuple(pose_2d_pred[12]), tuple(pose_2d_pred[13]), color, width)
    # for joint in pose_2d_pred:
    #     cv2.circle(image, tuple(joint), 10, (0, 0, 255), -1)


def vis_2d(sampler, image_dir, seperate_show=True):
    image = cv2.imread(image_dir)

    pose_3d_tmp = sampler.pg_best.pose_3d.copy()
    pose_3d_tmp = world2camera(pose_3d_tmp)
    pose_3d_cam = pose3d_all2partial(pose_3d_tmp)
    pose_2d_pred = (np.matmul(sampler.pg_current.camera_mat, sampler.pg_current.r_mat).dot(
        pose_3d_cam.T)).T
    for _iter in range(pose_2d_pred.shape[0]):
        pose_2d_pred[_iter, :] = pose_2d_pred[_iter, :] / pose_2d_pred[_iter, 2]
    pose_2d_pred = pose_2d_pred[:, 0:2].reshape([14, 2])

    zero_ind = np.where(np.sum(sampler.pg_current.pose_2d, axis=1) == 0)[0]
    error2d3d = 0
    pose_2d_gt = sampler.pg_current.pose_2d.copy()
    pose_2d_gt = pose_2d_gt[:, [1, 0]]
    for _iter in range(pose_2d_pred.shape[0]):
        if _iter not in zero_ind:
            error2d3d += np.sum(np.abs(pose_2d_pred[_iter, :] - pose_2d_gt[_iter, :]))

    pose_2d_pred = pose_2d_pred.astype(int)
    draw_2dpose(image, pose_2d_pred.astype(int))

    draw_2dpose(image, pose_2d_gt.astype(int), color=(0, 255, 0))

    for single_3d_bdb in sampler.pg_best.bbx_3d:
        single_3d_bdb = world2camera(single_3d_bdb)
        bbx_2d_pred = (np.matmul(sampler.pg_current.camera_mat, sampler.pg_current.r_mat).dot(
            single_3d_bdb.T)).T
        for _iter in range(bbx_2d_pred.shape[0]):
            bbx_2d_pred[_iter, :] = bbx_2d_pred[_iter, :] / bbx_2d_pred[_iter, 2]
        bbx_2d_pred = bbx_2d_pred[:, 0:2].reshape([8, 2])

        draw_3d_bbx_in2d(image, bbx_2d_pred.astype(int))
        # for corner in bbx_2d_pred:
        #     cv2.circle(image, tuple(corner), 10, (0, 0, 255), -1)
    for single_2d_bdb in sampler.pg_current.bbx_2d:
        draw_2d_bbx(image, single_2d_bdb)

    if seperate_show:
        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


def vis_2d_multi(sampler, image_dir, seperate_show=True):
    image = cv2.imread(image_dir)

    if sampler.pg_best.pose_3d is not None:
        for _jter in range(len(sampler.pg_best.pose_3d)):
            pose_3d_tmp = sampler.pg_best.pose_3d[_jter].copy()
            pose_3d_tmp = world2camera(pose_3d_tmp)
            pose_3d_cam = pose3d_all2partial(pose_3d_tmp)
            pose_2d_pred = (np.matmul(sampler.pg_current.camera_mat, sampler.pg_current.r_mat).dot(
                pose_3d_cam.T)).T
            for _iter in range(pose_2d_pred.shape[0]):
                pose_2d_pred[_iter, :] = pose_2d_pred[_iter, :] / pose_2d_pred[_iter, 2]
            pose_2d_pred = pose_2d_pred[:, 0:2].reshape([14, 2])

            zero_ind = np.where(np.sum(sampler.pg_current.pose_2d, axis=1) == 0)[0]
            error2d3d = 0
            pose_2d_gt = sampler.pg_current.pose_2d[_jter].copy()
            # pose_2d_gt = pose_2d_gt[:, [1, 0]] # natural image or sunrgb needs comment
            for _iter in range(pose_2d_pred.shape[0]):
                if _iter not in zero_ind:
                    error2d3d += np.sum(np.abs(pose_2d_pred[_iter, :] - pose_2d_gt[_iter, :]))
            pose_2d_pred = pose_2d_pred.astype(int)
            draw_2dpose(image, pose_2d_pred.astype(int))

            draw_2dpose(image, pose_2d_gt.astype(int), color=(0, 255, 0))

    for single_3d_bdb in sampler.pg_best.bbx_3d:
        single_3d_bdb = world2camera(single_3d_bdb)
        bbx_2d_pred = (np.matmul(sampler.pg_current.camera_mat, sampler.pg_current.r_mat).dot(
            single_3d_bdb.T)).T
        for _iter in range(bbx_2d_pred.shape[0]):
            bbx_2d_pred[_iter, :] = bbx_2d_pred[_iter, :] / bbx_2d_pred[_iter, 2]
        bbx_2d_pred = bbx_2d_pred[:, 0:2].reshape([8, 2])

        draw_3d_bbx_in2d(image, bbx_2d_pred.astype(int))
        # for corner in bbx_2d_pred:
        #     cv2.circle(image, tuple(corner), 10, (0, 0, 255), -1)
    for single_2d_bdb in sampler.pg_current.bbx_2d:
        draw_2d_bbx(image, single_2d_bdb.astype(int))

    if seperate_show:
        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


def vis_scene_multi(layout, bdb, pose):
    fig = plt.figure()
    ax = Axes3D(fig)
    # draw room layout
    # note here bottom points is actually the roof
    maxhight = 1.2
    totalpoints = layout.shape[0]
    bottompoints = layout[0: int(totalpoints / 2), :]

    toppoints = layout[int(totalpoints / 2): totalpoints, :]
    toppoints[toppoints[:, 2] > maxhight, 2] = maxhight
    bottompoints[bottompoints[:, 2] > maxhight, 2] = maxhight
    ind = np.argmin(toppoints[:, 1])

    for _iter in range(toppoints.shape[0]-1):
        if _iter != ind and _iter+1 != ind:
            ax.plot([toppoints[_iter, 0], toppoints[_iter + 1, 0]],
                    [toppoints[_iter, 1], toppoints[_iter + 1, 1]],
                    [toppoints[_iter, 2], toppoints[_iter + 1, 2]])
            ax.plot([bottompoints[_iter, 0], bottompoints[_iter + 1, 0]],
                    [bottompoints[_iter, 1], bottompoints[_iter + 1, 1]],
                    [bottompoints[_iter, 2], bottompoints[_iter + 1, 2]])
            ax.plot([toppoints[_iter, 0], bottompoints[_iter, 0]],
                    [toppoints[_iter, 1], bottompoints[_iter, 1]],
                    [toppoints[_iter, 2], bottompoints[_iter, 2]])

    if ind != 0 and ind != toppoints.shape[0]-1:
        ax.plot([toppoints[-1, 0], toppoints[0, 0]],
                [toppoints[-1, 1], toppoints[0, 1]],
                [toppoints[-1, 2], toppoints[0, 2]])
        ax.plot([bottompoints[-1, 0], bottompoints[0, 0]],
                [bottompoints[-1, 1], bottompoints[0, 1]],
                [bottompoints[-1, 2], bottompoints[0, 2]])
        ax.plot([toppoints[-1, 0], bottompoints[-1, 0]],
                [toppoints[-1, 1], bottompoints[-1, 1]],
                [toppoints[-1, 2], bottompoints[-1, 2]])

    '''draw 3d bdb'''
    for single_bdb in bdb:
        draw_3dbdb(ax, single_bdb)

    '''draw 3d pose'''
    for single_pose in pose:
        draw_3dpose(ax, single_pose)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')


def vis_all(sampler, image_dir, result_dir, save_image=True):
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    image_ori = cv2.imread(image_dir)
    image_2d_detection = cv2.imread(os.path.join(result_dir, image_dir.split('/')[-1].split('.')[0]+'_result.jpg'))
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(cv2.cvtColor(image_2d_detection, cv2.COLOR_BGR2RGB))
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(sampler.energy_landscape)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(cv2.cvtColor(vis_2d(sampler, image_dir, seperate_show=False), cv2.COLOR_BGR2RGB))
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    vis_scene_w_ax(sampler.pg_ini.layout, sampler.pg_ini.bbx_3d, sampler.pg_ini.pose_3d, ax5)
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    vis_scene_w_ax(sampler.pg_best.layout, sampler.pg_best.bbx_3d, sampler.pg_best.pose_3d, ax6)
    ax6.grid(False)
    ax6.set_axis_off()
    if save_image:
        fig.savefig(os.path.join(result_dir, image_dir.split('/')[-1].split('.')[0]+'all_result.png'))
    plt.axis('equal')
    plt.close(fig)
    # plt.show()


def vis_all_multi(sampler, image_dir, result_dir, save_image=True):
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    image_ori = cv2.imread(image_dir)
    image_2d_detection = cv2.imread(os.path.join(result_dir, image_dir.split('/')[-1].split('.')[0]+'_result.jpg'))
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(cv2.cvtColor(image_2d_detection, cv2.COLOR_BGR2RGB))
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(sampler.energy_landscape)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(cv2.cvtColor(vis_2d_multi(sampler, image_dir, seperate_show=False), cv2.COLOR_BGR2RGB))
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    vis_scene_w_ax_multi(sampler.pg_ini.layout, sampler.pg_ini.bbx_3d, sampler.pg_ini.pose_3d, ax5)
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    vis_scene_w_ax_multi(sampler.pg_best.layout, sampler.pg_best.bbx_3d, sampler.pg_best.pose_3d, ax6)
    plt.axis('equal')
    if save_image:
        fig.savefig(os.path.join(result_dir, image_dir.split('/')[-1].split('.')[0]+'all_result.png'))
        plt.close(fig)


if __name__ == "__main__":
    pass
