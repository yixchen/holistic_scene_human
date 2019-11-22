"""
    Created on Jan 20, 2019
    @author: Yixin Chen
    Sampler class.
"""
import os
import argparse
import datetime
import time
import random
import json
import copy
from pg import HOI
from mcmc import metropolis_hasting
from utils import *


OBJ_CATEGORY_CLEAN = ['recycle_bin', 'cpu', 'paper', 'toilet', 'stool', 'whiteboard', 'coffee_table', 'picture',
                      'keyboard', 'dresser', 'painting', 'bookshelf', 'night_stand', 'endtable', 'drawer', 'sink',
                      'monitor', 'computer', 'cabinet', 'shelf', 'lamp', 'garbage_bin', 'box', 'bed', 'sofa',
                      'sofa_chair', 'pillow', 'desk', 'table', 'chair']

hoi_detector = None  # PosePred('pred_all/weights-action-0.2-0.99.hdf5', 'pred_all/weights-hoi-0.2-0.91.hdf5')


def hoi_obj_label(hoi_type):
    if hoi_type == 'sit':
        return ['sofa_chair', 'chair', 'sofa']
    elif hoi_type == 'call':
        return ['phone']
    elif hoi_type == 'drink_eat':
        return ['cup']
    elif hoi_type == 'look_at':
        return ['paper', 'keyboard']
    elif hoi_type == 'stand' or hoi_type == 'haggle':
        return []


class Sampler:
    def __init__(self, arglist, pg):
        self.pg_current = pg
        self.energy_current = None
        # path
        self.ROOT_PATH = os.path.join(arglist.data_dir, 'image', 
                arglist.image_name)
        # configuration
        self.max_step = arglist.max_step_len
        self.sample_pose = arglist.sample_pose
        self.sample_obj = arglist.sample_obj
        self.sample_layout = arglist.sample_layout
        self.save_history = arglist.save_history
        self.save_dir = arglist.save_dir
        self.save_count = 0
        self.save_rate = arglist.save_rate
        self.scale = 1
        self.early_end_step = 100
        self.stagnate_step_pose = 0
        self.stagnate_step_obj = 0
        # recordings
        self.pg_ini = None
        self.energy_ini = None
        self.pg_best = None
        self.energy_best = None
        self.energy_landscape = list()
        self.record = list()
        self.inference_step = 0
        if self.save_history:
            self.pg_history = list()
        else:
            self.pg_history = None

    @staticmethod
    def finalize_hoi(pg):
        for _jter in range(len(pg.pose_3d)):
            for _iter in range(len(pg.hoi[_jter].type)):
                if pg.hoi[_jter].bbx_ind[_iter] is None:
                    if pg.hoi[_jter].type[_iter] == 'sit':
                        hip = pg.pose_3d[_jter][0]
                        hip_height = pg.pose_3d[_jter][0, 2] - np.min(pg.layout[:, 2])
                        bdb_3d = np.array([[hip[0] - 0.32, hip[1] - 0.26, np.min(pg.layout[:, 2]) +
                                            min(1.2, 1.8*hip_height)],
                                           [hip[0] - 0.32, hip[1] + 0.26, np.min(pg.layout[:, 2]) +
                                            min(1.2, 1.8*hip_height)],
                                           [hip[0] + 0.32, hip[1] + 0.26, np.min(pg.layout[:, 2]) +
                                            min(1.2, 1.8*hip_height)],
                                           [hip[0] + 0.32, hip[1] - 0.26, np.min(pg.layout[:, 2]) +
                                            min(1.2, 1.8*hip_height)],
                                           [hip[0] - 0.32, hip[1] - 0.26, np.min(pg.layout[:, 2])],
                                           [hip[0] - 0.32, hip[1] + 0.26, np.min(pg.layout[:, 2])],
                                           [hip[0] + 0.32, hip[1] + 0.26, np.min(pg.layout[:, 2])],
                                           [hip[0] + 0.3, hip[1] - 0.26, np.min(pg.layout[:, 2])]])
                        pg.bbx_3d.append(bdb_3d)
                    if pg.hoi[_jter].type[_iter] == 'call':
                        if pg.pose_3d[_jter][16, 2] > pg.pose_3d[_jter][13, 2]:
                            right_hand = pg.pose_3d[_jter][16]
                        else:
                            right_hand = pg.pose_3d[_jter][13]
                        bdb_3d = np.array([[right_hand[0] - 0.01, right_hand[1] - 0.03, right_hand[2] + 0.1],
                                           [right_hand[0] - 0.01, right_hand[1] + 0.03, right_hand[2] + 0.1],
                                           [right_hand[0] + 0.01, right_hand[1] + 0.03, right_hand[2] + 0.1],
                                           [right_hand[0] + 0.01, right_hand[1] - 0.03, right_hand[2] + 0.1],
                                           [right_hand[0] - 0.01, right_hand[1] - 0.03, right_hand[2] - 0.1],
                                           [right_hand[0] - 0.01, right_hand[1] + 0.03, right_hand[2] - 0.1],
                                           [right_hand[0] + 0.01, right_hand[1] + 0.03, right_hand[2] - 0.1],
                                           [right_hand[0] + 0.01, right_hand[1] - 0.03, right_hand[2] - 0.1]])
                        pg.bbx_3d.append(bdb_3d)
                    if pg.hoi[_jter].type[_iter] == 'drink_eat':
                        if pg.pose_3d[_jter][16, 2] > pg.pose_3d[_jter][13, 2]:
                            right_hand = pg.pose_3d[_jter][16]
                        else:
                            right_hand = pg.pose_3d[_jter][13]
                        x_mean = 0.0212392
                        y_mean = -0.03728930
                        z_mean = -0.06195784
                        bdb_3d = np.array([[right_hand[0] - 0.025, right_hand[1] - 0.025, right_hand[2] + 0.07],
                                           [right_hand[0] - 0.025, right_hand[1] + 0.025, right_hand[2] + 0.07],
                                           [right_hand[0] + 0.025, right_hand[1] + 0.025, right_hand[2] + 0.07],
                                           [right_hand[0] + 0.025, right_hand[1] - 0.025, right_hand[2] + 0.07],
                                           [right_hand[0] - 0.025, right_hand[1] - 0.025, right_hand[2] - 0.07],
                                           [right_hand[0] - 0.025, right_hand[1] + 0.025, right_hand[2] - 0.07],
                                           [right_hand[0] + 0.025, right_hand[1] + 0.025, right_hand[2] - 0.07],
                                           [right_hand[0] + 0.025, right_hand[1] - 0.025, right_hand[2] - 0.07]])
                        bdb_3d[0, :] += x_mean
                        bdb_3d[1, :] += y_mean
                        bdb_3d[2, :] += z_mean
                        pg.bbx_3d.append(bdb_3d)
                    if pg.hoi[_jter].type[_iter] == 'look at':
                        right_hand = pg.pose_3d[_jter][16]
                        x_mean = 0.11192783
                        y_mean = -0.07329582
                        z_mean = -0.05192857
                        bdb_3d = np.array([[right_hand[0] - 0.15, right_hand[1] - 0.1, right_hand[2] + 0.01],
                                           [right_hand[0] - 0.15, right_hand[1] + 0.1, right_hand[2] + 0.01],
                                           [right_hand[0] + 0.15, right_hand[1] + 0.1, right_hand[2] + 0.01],
                                           [right_hand[0] + 0.15, right_hand[1] - 0.1, right_hand[2] + 0.01],
                                           [right_hand[0] - 0.15, right_hand[1] - 0.1, right_hand[2] - 0.01],
                                           [right_hand[0] - 0.15, right_hand[1] + 0.1, right_hand[2] - 0.01],
                                           [right_hand[0] + 0.15, right_hand[1] + 0.1, right_hand[2] - 0.01],
                                           [right_hand[0] + 0.15, right_hand[1] - 0.1, right_hand[2] - 0.01]])
                        bdb_3d[0, :] += x_mean
                        bdb_3d[1, :] += y_mean
                        bdb_3d[2, :] += z_mean
                        pg.bbx_3d.append(bdb_3d)

    # propose the moving method
    @staticmethod
    def q_moving_proposal():
        r = random.random()
        if r < 0.95:
            return 0, 0.95  # propose gradient descent algorithm
        if 0.95 <= r <= 1:
            return 1, 0.05  # propose gradient ascent algorithm

    @staticmethod
    def compute_2d3d_error_pose(pg):
        pose_3d_all = pg.pose_3d.copy()
        error2d3d_total = 0
        for _jter, pose_3d_tmp in enumerate(pose_3d_all):
            pose_3d_tmp = world2camera(pose_3d_tmp)
            pose_3d_cam = pose3d_all2partial(pose_3d_tmp)
            pose_2d_pred = (np.matmul(pg.camera_mat, pg.r_mat).dot(
                pose_3d_cam.T)).T

            for _iter in range(pose_2d_pred.shape[0]):
                pose_2d_pred[_iter, :] = pose_2d_pred[_iter, :]/pose_2d_pred[_iter, 2]

            pose_2d_pred = pose_2d_pred[:, 0:2].reshape([14, 2])
            pose_2d_gt = pg.pose_2d[_jter].copy()
            zero_ind = np.where(np.sum(pose_2d_gt, axis=1) == 0)[0]
            error2d3d = 0
            for _iter in range(pose_2d_pred.shape[0]):
                if _iter not in zero_ind:
                    error2d3d += np.sum(np.abs(pose_2d_pred[_iter, :] - pose_2d_gt[_iter, :]))
            error2d3d_total += error2d3d / float(pose_2d_pred.shape[0] - zero_ind.shape[0])
        return error2d3d_total

    @staticmethod
    def compute_2d3d_error_obj(pg):
        error2d3d = 0
        for _iter in range(len(pg.bbx_2d)):
            obj_2d_gt = pg.bbx_2d[_iter].copy()
            obj_3d_tmp = pg.bbx_3d[_iter].copy()
            obj_3d_tmp = world2camera(obj_3d_tmp)
            obj_2d_pred = (np.matmul(pg.camera_mat, pg.r_mat).dot(
                obj_3d_tmp.T)).T

            for _jter in range(obj_2d_pred.shape[0]):
                obj_2d_pred[_jter, :] = obj_2d_pred[_jter, :] / obj_2d_pred[_jter, 2]

            obj_2d_pred = obj_2d_pred[:, 0:2].reshape((8, 2))

            error_metric = 'iou'  # 'min_max_bdb'
            if error_metric == 'min_max_bdb':
                x_min_pred, x_max_pred = np.min(obj_2d_pred[:, 0]), np.max(obj_2d_pred[:, 0])
                y_min_pred, y_max_pred = np.min(obj_2d_pred[:, 1]), np.max(obj_2d_pred[:, 1])
                x_min_gt, x_max_gt = obj_2d_gt[0], obj_2d_gt[2]
                y_min_gt, y_max_gt = obj_2d_gt[1], obj_2d_gt[3]
                error2d3d += np.sqrt((np.square(x_min_gt - x_min_pred) + np.square(x_max_gt - x_max_pred) +
                                      np.square(y_min_gt - y_min_pred) + np.square(y_max_gt - y_max_pred))) / 4.0
            elif error_metric == 'iou':
                error2d3d += np.exp(8*(1-iou_2dobj(obj_2d_pred, obj_2d_gt)))

        return error2d3d

    @staticmethod
    def compute_4dhoi_error(pg):
        energy = 0
        if len(pg.hoi) == 0:
            return 0
        for _jter in range(len(pg.pose_3d)):
            for _iter in range(len(pg.hoi[_jter].type)):
                if pg.hoi[_jter].type[_iter] == 'sit':
                    energy_model = hoi_model([0.00257409, 0.01829389, -0.37240382],
                                             [0.15987131, 0.17563206, 0.27549762])
                    li = 1
                    hip = pg.pose_3d[_jter][0]
                    if pg.hoi[_jter].bbx_ind[_iter] is not None:
                        obj_center = np.mean(pg.bbx_3d[pg.hoi[_jter].bbx_ind[_iter]], axis=0)
                        li *= energy_model.get_energy(obj_center[0] - hip[0], 0)
                        li *= energy_model.get_energy(obj_center[1] - hip[1], 1)
                        li *= energy_model.get_energy(obj_center[2] - hip[2], 2)

                        energy += np.exp(10 * (1 - li))
                if pg.hoi[_jter].type[_iter] == 'look_at':
                    energy_model = hoi_model([0.11192783, -0.07329582, -0.05192857],
                                             [0.08321479, 0.15574328, 0.04512738])
                    li = 1
                    joint = pg.pose_3d[_jter][16]
                    if pg.hoi[_jter].bbx_ind[_iter] is not None:
                        obj_center = np.mean(pg.bbx_3d[pg.hoi[_jter].bbx_ind[_iter]], axis=0)
                        li *= energy_model.get_energy(obj_center[0] - joint[0], 0)
                        li *= energy_model.get_energy(obj_center[1] - joint[1], 1)
                        li *= energy_model.get_energy(obj_center[2] - joint[2], 2)

                        energy += np.exp(10 * (1 - li))
                if pg.hoi[_jter].type[_iter] == 'drink_eat':
                    energy_model = sit_hoi_model([0.0212392, -0.03728930, -0.06195784],
                                                 [0.0219587, 0.02563206, 0.15517897])
                    li = 1
                    hip = pg.pose_3d[_jter][0]
                    if pg.hoi[_jter].bbx_ind[_iter] is not None:
                        obj_center = np.mean(pg.bbx_3d[pg.hoi[_jter].bbx_ind[_iter]], axis=0)
                        li *= energy_model.get_energy(obj_center[0] - hip[0], 0)
                        li *= energy_model.get_energy(obj_center[1] - hip[1], 1)
                        li *= energy_model.get_energy(obj_center[2] - hip[2], 2)

                        energy += np.exp(10 * (1 - li))

        return energy

    @staticmethod
    def compute_layout_likelihood(pg):
        layout_error = 0
        # pose and layout
        layout_x_min, layout_x_max = np.min(pg.layout[:, 0]), np.max(pg.layout[:, 0])
        layout_y_min, layout_y_max = np.min(pg.layout[:, 1]), np.max(pg.layout[:, 1])

        def layout_error_for_joint(_joint, x_max, x_min, y_max, y_min):
            return (np.max([0, _joint[0] - x_max]) +
                    np.max([0, _joint[1] - y_max]) +
                    np.max([0, x_min - _joint[0]]) +
                    np.max([0, y_min - _joint[1]]))
        for _iter in range(len(pg.pose_3d)):
            for _joint in pg.pose_3d[_iter]:
                layout_error += layout_error_for_joint(_joint, layout_x_max, layout_x_min, layout_y_max, layout_y_min)
        # obj and layout
        for bdb3d in pg.bbx_3d:
            for _joint in bdb3d:
                layout_error += layout_error_for_joint(_joint, layout_x_max, layout_x_min, layout_y_max, layout_y_min)
        return 10 * layout_error

    @staticmethod
    def compute_physical_error(pg, count_pose=False, count_obj=False):
        floor_height = np.min(pg.layout[:, 2])
        # human-ground relation
        if count_pose:
            pg.e_pose_ground = 0
            for _iter in range(len(pg.pose_3d)):
                foot_height = np.abs(0.5 * (pg.pose_3d[_iter][3, 2] + pg.pose_3d[_iter][6, 2]) - floor_height)
                pg.e_pose_ground += np.exp(30 * foot_height)

        # support relation
        if count_obj:
            pg.e_support = 0
            for _iter in range(len(pg.support_rel)):
                supporting = pg.support_rel[_iter]
                supporting_obj = supporting[1]
                if supporting_obj == -2:
                    continue
                elif supporting_obj == -1:
                    distance_floor = np.abs(np.min(pg.bbx_3d[_iter][:, 2]) - floor_height)
                    pg.e_support += np.exp(10*distance_floor)
                else:
                    supporting_upper_height = np.max(pg.bbx_3d[supporting_obj][:, 2])
                    supported_lower_height = np.min(pg.bbx_3d[_iter][:, 2])
                    height_diff = np.abs(supported_lower_height - supporting_upper_height)
                    intersect_ratio = 1 - intersection_2d_ratio(pg.bbx_3d[supporting_obj], pg.bbx_3d[_iter])
                    pg.e_support += np.exp(height_diff*10 + intersect_ratio*6)

        return pg.e_pose_ground + pg.e_support

    @staticmethod
    def detect_hoi_load(pg):
        adjust_hoi = False
        # load HOI
        for _iter in range(len(pg.pose_3d)):
            hoi_tmp = HOI()
            hoi_tmp.type = pg.hoi_type[_iter]
            hoi_tmp.obj_label = pg.hoi_obj_label[_iter]
            hoi_tmp.bbx_ind = [None, None]
            pg.hoi.append(copy.copy(hoi_tmp))

        for _jter in range(len(pg.pose_3d)):
            for _iter in range(len(pg.hoi[_jter].type)):
                if pg.hoi[_jter].type[_iter] is not None:
                    min_dis = 100
                    for ind, label in enumerate(pg.bbx_label):
                        if OBJ_CATEGORY_CLEAN[label] in hoi_obj_label(pg.hoi[_jter].type[_iter]):
                            if np.linalg.norm(np.mean(pg.pose_3d[_jter], axis=0) -
                                              np.mean(pg.bbx_3d[ind], axis=0)) < min(1, min_dis):
                                min_dis = np.linalg.norm(np.mean(pg.pose_3d[_jter], axis=0) -
                                                         np.mean(pg.bbx_3d[ind], axis=0))
                                pg.hoi[_jter].bbx_ind[_iter] = ind
                                adjust_hoi = True
        return adjust_hoi

    # compute total likelihood, return energy
    def compute_total_likelihood(self, pg, show_energy=False, count_pose=False,
                                 count_obj=False, count_layout=False, count_hoi=False):
        self.inference_step += 1

        if count_pose:
            pg.e_2d3d_pose = self.compute_2d3d_error_pose(copy.deepcopy(pg))
        if count_obj:
            pg.e_2d3d_obj = self.compute_2d3d_error_obj(copy.deepcopy(pg))
        if count_hoi:
            pg.e_4dhoi = self.compute_4dhoi_error(copy.deepcopy(pg))
        if count_layout:
            pg.e_layout = self.compute_layout_likelihood(copy.deepcopy(pg))

        pg.e_physical = self.compute_physical_error(pg, count_pose=count_pose, count_obj=count_obj)

        pg.e_total = 2.0 * pg.e_2d3d_pose + pg.e_2d3d_obj + pg.e_physical + pg.e_4dhoi + pg.e_layout
        if show_energy:
            print('e_total is :{}. e_2d3d_pose is :{}. e_2d3d_obj is :{}.'
                  ' e_4dhoi is :{}. e_physical is :{}. e_layout is :{}'.
                  format(pg.e_total, pg.e_2d3d_pose, pg.e_2d3d_obj, pg.e_4dhoi, pg.e_physical, pg.e_layout))
        return pg.e_total

    def init_all_3d(self):
        # put initial 3d Pose based on init_method
        pg_new = copy.deepcopy(self.pg_current)

        init_method = 'obj'
        pose_3d_des = None
        for _iter in range(len(pg_new.pose_3d)):
            if init_method == 'proj_head':
                # init pose by project head in 2d back into 3d
                gound_height = np.min(pg_new.layout[:, 2])
                K = np.matmul(pg_new.camera_mat, pg_new.r_mat)[0]
                height_tmp = 1.4 + gound_height
                head_3d_tmp = pg_new.pose_3d[_iter][10, :]
                head_2d_tmp = pg_new.pose_2d[_iter][0, :]
                head_2d_tmp = head_2d_tmp[[1, 0]]

                linalg_a = K
                linalg_b = height_tmp * K[:, 1]
                linalg_a[0, 1] = -head_2d_tmp[0]
                linalg_a[1, 1] = -head_2d_tmp[1]
                linalg_a[2, 1] = -1

                res = np.linalg.solve(linalg_a, linalg_b)
                res[1] = res[2]
                res[2] = height_tmp
                if res[0] < 0 and res[1] < 0 and res[2] < 0:
                    res = -res

                pose_3d_des = pg_new.pose_3d[_iter] - head_3d_tmp + res
            elif init_method == 'obj':
                # init pose by placing in the center of layout
                res = np.zeros((1, 3))
                for obj in pg_new.bbx_3d:
                    res += np.mean(obj, axis=0)
                res = res/float(len(pg_new.bbx_3d)+1.0)
                pose_3d_des = pg_new.pose_3d[_iter] - pg_new.pose_3d[_iter][0, :] + res

            pg_new.pose_3d[_iter] = pose_3d_des

        self.pg_current = copy.deepcopy(pg_new)
        self.initialize_support_relation()
        self.energy_current = self.compute_total_likelihood(self.pg_current, show_energy=False, count_pose=True,
                                                            count_obj=True, count_layout=True, count_hoi=True)
        self.pg_best = copy.deepcopy(self.pg_current)
        self.pg_ini = copy.deepcopy(self.pg_current)
        self.energy_best = self.energy_current
        self.energy_ini = self.energy_current

        for _ in range(4):
            self.floor_adjust(1)

    def initialize_support_relation(self):
        # initialize support relation
        support_all = []
        '''
        wall: -2
        floor: -1
        '''
        with open('support_relation.json', 'r') as f:
            support_relation = json.load(f)
        for _iter in range(len(self.pg_current.bbx_2d)):
            obj_type = OBJ_CATEGORY_CLEAN[self.pg_current.bbx_label[_iter]]
            if obj_type == 'vanity':
                supporting = support_relation['bathroomvanity']
            else:
                supporting = support_relation[obj_type]
            num_total = 0
            for _, value in supporting.items():
                num_total += value
            prob_max = 0
            prob_max_type = None
            for key, value in supporting.items():
                supporting[key] = value / float(num_total)
                if supporting[key] > prob_max:
                    prob_max = supporting[key]
                    prob_max_type = key
            distance_floor = np.min(self.pg_current.bbx_3d[_iter][:, 2]) - np.min(self.pg_current.layout[:, 2])
            if prob_max_type == 'wall' or np.abs(distance_floor) > 2.0:
                continue
            support_energy = list()
            for _jter in range(len(self.pg_current.bbx_2d)):
                target_obj_type = OBJ_CATEGORY_CLEAN[self.pg_current.bbx_label[_jter]]
                if obj_type == target_obj_type:
                    support_energy.append(0)
                elif self.pg_current.bbx_2d[_jter][3] < self.pg_current.bbx_2d[_iter][1]:
                    support_energy.append(0)
                else:
                    if target_obj_type not in supporting.keys() \
                            or supporting[OBJ_CATEGORY_CLEAN[self.pg_current.bbx_label[_jter]]] < 1 / 10.0:
                        prior_energy = 1 / 10.0
                    else:
                        prior_energy = supporting[OBJ_CATEGORY_CLEAN[self.pg_current.bbx_label[_jter]]]
                    intersect_ratio = 1 - vertical_intersect_ratio(self.pg_current.bbx_2d[_jter],
                                                                   self.pg_current.bbx_2d[_iter])
                    if intersect_ratio == 1:
                        support_energy.append(0)
                    else:
                        height_distance = np.max([0,
                                                  self.pg_current.bbx_2d[_jter][1] -
                                                  self.pg_current.bbx_2d[_iter][3]])
                        likelihood_energy = np.exp(-height_distance/100.0 - intersect_ratio)
                        support_energy.append(prior_energy * likelihood_energy)
            # compute the supporting relation with the floor
            prior_energy = supporting['floor']
            likelihood_energy = np.exp(-np.abs(distance_floor) * 3)
            support_energy.append(prior_energy * likelihood_energy)

            # compute the real supporting relation
            supported_index = support_energy.index(max(support_energy))
            if supported_index != len(support_energy) - 1:
                support_all.append([_iter, supported_index])
            else:
                if distance_floor > 0.75:
                    support_all.append([_iter, -2])
                elif OBJ_CATEGORY_CLEAN[self.pg_current.bbx_label[_iter]] == 'whiteboard':
                    support_all.append([_iter, -2])
                else:
                    support_all.append([_iter, -1])
        self.pg_current.support_rel = support_all

    def one_step_pose_adjust(self, T, index, move_type, move_face=None):
        for _iter in range(len(self.pg_current.pose_3d)):
            if move_type == 'translation':
                size_ori = np.max(self.pg_current.pose_3d[_iter][:, 2]) - np.min(self.pg_current.pose_3d[_iter][:, 2])
                pg_des = copy.deepcopy(self.pg_current)
                delta = None
                if move_face == 'x':
                    delta = 0.1 * size_ori * self.scale
                    pg_des.pose_3d[_iter][:, 0] += delta
                elif move_face == 'y':
                    delta = 0.1 * size_ori * self.scale
                    pg_des.pose_3d[_iter][:, 1] += delta
                elif move_face == 'z':
                    delta = 0.1 * size_ori * self.scale
                    pg_des.pose_3d[_iter][:, 2] += delta
                elif move_face == 'depth':
                    pose_center = self.pg_current.pose_3d[_iter][8, :].copy()
                    depth_direction = pose_center / np.linalg.norm(pose_center)
                    delta = 0.1 * size_ori * self.scale * depth_direction
                    # z axis stays the same
                    delta[2] = 0
                    pg_des.pose_3d[_iter] += delta

                e_total_des = self.compute_total_likelihood(pg_des, show_energy=False, count_pose=True,
                                                            count_obj=False, count_layout=False, count_hoi=True)
                gradient = (self.energy_current - e_total_des)
                if gradient == 0:
                    return
                gradient_type, move_prob = self.q_moving_proposal()
                delta *= np.sign(gradient)
                if gradient_type == 1:
                    delta *= -1
                pg_new = copy.deepcopy(self.pg_current)
                if move_face == 'x':
                    pg_new.pose_3d[_iter][:, 0] += delta
                elif move_face == 'y':
                    pg_new.pose_3d[_iter][:, 1] += delta
                elif move_face == 'z':
                    pg_new.pose_3d[_iter][:, 2] += delta
                elif move_face == 'depth':
                    pg_new.pose_3d[_iter] += delta

                e_total_new = self.compute_total_likelihood(pg_new, show_energy=False, count_pose=True,
                                                            count_obj=False, count_layout=False, count_hoi=True)
                accept = metropolis_hasting(self.energy_current, e_total_new, move_prob, 1 - move_prob, T)
                if accept:
                    self.pg_current = copy.deepcopy(pg_new)
                    self.energy_current = e_total_new
                    self.energy_landscape.append(e_total_new)
                    if e_total_new < self.energy_best:
                        self.pg_best = copy.deepcopy(pg_new)
                        self.energy_best = e_total_new
                    self.record.append(1)
                    self.stagnate_step_pose = 0
                    self.save_count += 1
                    if self.save_history and (self.save_count % self.save_rate) == 0:
                        self.pg_history.append(pg_new)
                else:
                    self.record.append(0)
                    self.stagnate_step_pose += 1

            if move_type == 'rotation':
                assert move_face == 'z'
                pose_des = None
                pose_center = None
                if move_face == 'z':
                    rotate_angle = 5.625 / 180 * np.pi * self.scale
                    pose_center = self.pg_current.pose_3d[_iter][0]
                    # pose_center[2] = 0
                    pose_des = rotation_matrix_3d_z(rotate_angle).dot(
                        np.array(self.pg_current.pose_3d[_iter] - pose_center).T).T + pose_center
                pg_des = copy.deepcopy(self.pg_current)
                pg_des.pose_3d[_iter] = pose_des
                e_total_des = self.compute_total_likelihood(pg_des, show_energy=False, count_pose=True,
                                                            count_obj=False, count_layout=False, count_hoi=True)
                gradient = self.energy_current - e_total_des
                if gradient == 0:
                    return
                gradient_type, move_prob = self.q_moving_proposal()
                if gradient_type == 0:  # do gradient descent
                    rotate_angle *= np.sign(gradient)
                else:
                    rotate_angle *= -np.sign(gradient)
                pg_new = copy.deepcopy(self.pg_current)
                pg_new.pose_3d[_iter] = rotation_matrix_3d_z(rotate_angle).dot(
                        np.array(self.pg_current.pose_3d[_iter] - pose_center).T).T + pose_center
                e_total_new = self.compute_total_likelihood(pg_new, show_energy=False, count_pose=True,
                                                            count_obj=False, count_layout=False, count_hoi=True)
                accept = metropolis_hasting(self.energy_current, e_total_new, move_prob, 1 - move_prob, T)
                if accept:
                    self.pg_current = copy.deepcopy(pg_new)
                    self.energy_current = e_total_new
                    self.energy_landscape.append(e_total_new)
                    if e_total_new < self.energy_best:
                        self.pg_best = copy.deepcopy(pg_new)
                        self.energy_best = e_total_new
                    self.record.append(1)
                    self.stagnate_step_pose = 0
                    self.save_count += 1
                    if self.save_history and (self.save_count % self.save_rate) == 0:
                        self.pg_history.append(pg_new)
                else:
                    self.record.append(0)
                    self.stagnate_step_pose += 1

            if move_type == 'scale':
                scale_change = 0.1 * self.scale
                assert move_face in ['upscale', 'downscale']
                pose_center = self.pg_current.pose_3d[_iter][0]
                if move_face == 'upscale':
                    pose_des = (self.pg_current.pose_3d[_iter] - pose_center) * (1 + scale_change) + pose_center
                else:
                    pose_des = (self.pg_current.pose_3d[_iter] - pose_center) * (1 - scale_change) + pose_center
                pg_des = copy.deepcopy(self.pg_current)
                pg_des.pose_3d[_iter] = pose_des
                e_total_des = self.compute_total_likelihood(pg_des, show_energy=False, count_pose=True,
                                                            count_obj=False, count_layout=False, count_hoi=True)
                gradient = (self.energy_current - e_total_des) / np.abs(scale_change)
                if gradient == 0:
                    return
                gradient_type, move_prob = self.q_moving_proposal()
                scale_change *= np.sign(gradient)
                if gradient_type == 1:
                    scale_change *= -1
                pg_new = copy.deepcopy(self.pg_current)
                if move_face == 'upscale':
                    pg_new.pose_3d[_iter] = (self.pg_current.pose_3d[_iter] - pose_center) * (1 + scale_change) \
                                            + pose_center
                else:
                    pg_new.pose_3d[_iter] = (self.pg_current.pose_3d[_iter] - pose_center) * (1 - scale_change) \
                                            + pose_center

                e_total_new = self.compute_total_likelihood(pg_new, show_energy=False, count_pose=True,
                                                            count_obj=False, count_layout=False, count_hoi=True)
                accept = metropolis_hasting(self.energy_current, e_total_new, move_prob, 1 - move_prob, T)
                if np.max(pg_new.pose_3d[_iter][:, 2]) - np.min(pg_new.pose_3d[_iter][:, 2]) > 1.8:
                    accept = False
                if accept:
                    self.pg_current = copy.deepcopy(pg_new)
                    self.energy_current = e_total_new
                    self.energy_landscape.append(e_total_new)
                    if e_total_new < self.energy_best:
                        self.pg_best = copy.deepcopy(pg_new)
                        self.energy_best = e_total_new
                    self.record.append(1)
                    self.stagnate_step_pose = 0
                    self.save_count += 1
                    if self.save_history and (self.save_count % self.save_rate) == 0:
                        self.pg_history.append(pg_new)
                else:
                    self.record.append(0)
                    self.stagnate_step_pose += 1

    def multi_step_translation_adjust_pose(self, T, index, move_type):
        # Adjust translation in all directions
        for _ in range(20):
            r = random.random()
            if 0 < r <= 0.3:
                self.one_step_pose_adjust(T, index, move_type, move_face='z')
            elif 0.3 < r <= 0.6:
                self.one_step_pose_adjust(T, index, move_type, move_face='depth')
            elif 0.6 < r <= 0.8:
                self.one_step_pose_adjust(T, index, move_type, move_face='x')
            elif 0.8 < r <= 1.0:
                self.one_step_pose_adjust(T, index, move_type, move_face='y')

    def multi_step_rotation_adjust_pose(self, T, index, move_type):
        self.one_step_pose_adjust(T, index, move_type, move_face='z')

    def multi_step_scale_adjust_pose(self, T, index, move_type):
        r = random.random()
        if 0 < r <= 0.5:
            self.one_step_pose_adjust(T, index, move_type, move_face='upscale')
        else:
            self.one_step_pose_adjust(T, index, move_type, move_face='downscale')

    def pose_adjust(self, T):
        r = random.random()
        if 0 < r <= 0.7:
            move_type = 'translation'
            self.multi_step_translation_adjust_pose(T, 0, move_type)
        elif 0.7 < r <= 0.85:
            move_type = 'rotation'
            self.multi_step_rotation_adjust_pose(T, 0, move_type)
        elif 0.95 < r <= 1.0:
            move_type = 'scale'
            self.multi_step_scale_adjust_pose(T, 0, move_type)

    def one_step_obj_adjust(self, T, index, move_type, move_face):
        if move_type == 'translation':
            size_ori = np.max(self.pg_current.bbx_3d[index][:, 2]) - np.min(self.pg_current.bbx_3d[index][:, 2])
            pg_des = copy.deepcopy(self.pg_current)
            delta = None
            if move_face == 'depth':
                obj_center = np.mean(self.pg_current.bbx_3d[index], axis=0)
                depth_direction = obj_center / np.linalg.norm(obj_center)
                delta = 0.2 * size_ori * self.scale * depth_direction
                # z axis stays the same
                delta[2] = 0
                pg_des.bbx_3d[index] += delta
            elif move_face == 'z':
                delta = 0.2 * size_ori * self.scale
                pg_des.bbx_3d[index][:, 2] += delta
            elif move_face == 'y':
                delta = 0.2 * size_ori * self.scale
                pg_des.bbx_3d[index][:, 1] += delta
            elif move_face == 'x':
                delta = 0.2 * size_ori * self.scale
                pg_des.bbx_3d[index][:, 0] += delta

            e_total_des = self.compute_total_likelihood(pg_des, show_energy=False, count_pose=False,
                                                        count_obj=True, count_layout=False, count_hoi=True)
            gradient = (self.energy_current - e_total_des)
            if gradient == 0:
                return
            gradient_type, move_prob = self.q_moving_proposal()
            delta *= np.sign(gradient)
            if gradient_type == 1:
                delta *= -1
            pg_new = copy.deepcopy(self.pg_current)
            if move_face == 'depth':
                pg_new.bbx_3d[index] += delta
            elif move_face == 'z':
                pg_new.bbx_3d[index][:, 2] += delta
            elif move_face == 'y':
                pg_new.bbx_3d[index][:, 1] += delta
            elif move_face == 'x':
                pg_new.bbx_3d[index][:, 0] += delta
            e_total_new = self.compute_total_likelihood(pg_new, show_energy=False, count_pose=False,
                                                        count_obj=True, count_layout=False, count_hoi=True)
            accept = metropolis_hasting(self.energy_current, e_total_new, move_prob, 1 - move_prob, T)
            if accept:
                self.pg_current = copy.deepcopy(pg_new)
                self.energy_current = e_total_new
                self.energy_landscape.append(e_total_new)
                if e_total_new < self.energy_best:
                    self.pg_best = copy.deepcopy(pg_new)
                    self.energy_best = e_total_new
                self.record.append(1)
                self.stagnate_step_obj = 0
                self.save_count += 1
                if self.save_history and (self.save_count % self.save_rate) == 0:
                    self.pg_history.append(pg_new)
            else:
                self.record.append(0)
                self.stagnate_step_obj += 1

        if move_type == 'rotation':
            assert move_face == 'z'
            obj_des = None
            obj_center = None
            if move_face == 'z':
                rotate_angle = 5.625 / 180 * np.pi * self.scale
                obj_center = np.mean(self.pg_current.bbx_3d[index], axis=0)
                obj_des = rotation_matrix_3d_z(rotate_angle).dot(
                    np.array(self.pg_current.bbx_3d[index] - obj_center).T).T + obj_center
            pg_des = copy.deepcopy(self.pg_current)
            pg_des.bbx_3d[index] = obj_des
            e_total_des = self.compute_total_likelihood(pg_des, show_energy=False, count_pose=False,
                                                        count_obj=True, count_layout=False, count_hoi=True)
            gradient = self.energy_current - e_total_des
            if gradient == 0:
                return
            gradient_type, move_prob = self.q_moving_proposal()
            if gradient_type == 0:  # do gradient descent
                rotate_angle *= np.sign(gradient)
            else:
                rotate_angle *= -np.sign(gradient)
            pg_new = copy.deepcopy(self.pg_current)
            pg_new.bbx_3d[index] = rotation_matrix_3d_z(rotate_angle).dot(
                np.array(self.pg_current.bbx_3d[index] - obj_center).T).T + obj_center
            e_total_new = self.compute_total_likelihood(pg_new, show_energy=False, count_pose=False,
                                                        count_obj=True, count_layout=False, count_hoi=True)
            accept = metropolis_hasting(self.energy_current, e_total_new, move_prob, 1 - move_prob, T)
            if accept:
                self.pg_current = copy.deepcopy(pg_new)
                self.energy_current = e_total_new
                self.energy_landscape.append(e_total_new)
                if e_total_new < self.energy_best:
                    self.pg_best = copy.deepcopy(pg_new)
                    self.energy_best = e_total_new
                self.record.append(1)
                self.stagnate_step_obj = 0
                self.save_count += 1
                if self.save_history and (self.save_count % self.save_rate) == 0:
                    self.pg_history.append(pg_new)
            else:
                self.record.append(0)
                self.stagnate_step_obj += 1

        if move_type == 'scale':
            scale_change = 0.05 * random.random() * self.scale
            assert move_face in ['upscale', 'downscale']
            obj_center = np.mean(self.pg_current.bbx_3d[index], axis=0)
            if move_face == 'upscale':
                obj_des = (self.pg_current.bbx_3d[index] - obj_center) * (1 + scale_change) + obj_center
            else:
                obj_des = (self.pg_current.bbx_3d[index] - obj_center) * (1 - scale_change) + obj_center
            pg_des = copy.deepcopy(self.pg_current)
            pg_des.bbx_3d[index] = obj_des
            e_total_des = self.compute_total_likelihood(pg_des, show_energy=False, count_pose=False,
                                                        count_obj=True, count_layout=False, count_hoi=True)
            gradient = (self.energy_current - e_total_des) / np.abs(scale_change)
            if gradient == 0:
                return
            gradient_type, move_prob = self.q_moving_proposal()
            scale_change *= np.sign(gradient)
            if gradient_type == 1:
                scale_change *= -1
            pg_new = copy.deepcopy(self.pg_current)
            if move_face == 'upscale':
                pg_new.bbx_3d[index] = (self.pg_current.bbx_3d[index] - obj_center) * (1 + scale_change) + obj_center
            else:
                pg_new.bbx_3d[index] = (self.pg_current.bbx_3d[index] - obj_center) * (1 - scale_change) + obj_center

            e_total_new = self.compute_total_likelihood(pg_new, show_energy=False, count_pose=False,
                                                        count_obj=True, count_layout=False, count_hoi=True)
            accept = metropolis_hasting(self.energy_current, e_total_new, move_prob, 1 - move_prob, T)
            if accept:
                self.pg_current = copy.deepcopy(pg_new)
                self.energy_current = e_total_new
                self.energy_landscape.append(e_total_new)
                if e_total_new < self.energy_best:
                    self.pg_best = copy.deepcopy(pg_new)
                    self.energy_best = e_total_new
                self.record.append(1)
                self.stagnate_step_obj = 0
                self.save_count += 1
                if self.save_history and (self.save_count % self.save_rate) == 0:
                    self.pg_history.append(pg_new)
            else:
                self.record.append(0)
                self.stagnate_step_obj += 1

    def multi_step_translation_adjust_obj(self, T, move_type):
        for _index in range(len(self.pg_current.bbx_2d)):
            for _ in range(20):
                r = random.random()
                if 0 < r <= 0.3:
                    self.one_step_obj_adjust(T, _index, move_type, move_face='depth')
                elif 0.3 < r <= 0.6:
                    self.one_step_obj_adjust(T, _index, move_type, move_face='z')
                elif 0.6 < r <= 0.8:
                    self.one_step_obj_adjust(T, _index, move_type, move_face='x')
                elif 0.8 < r <= 1.0:
                    self.one_step_obj_adjust(T, _index, move_type, move_face='y')

    def multi_step_rotation_adjust_obj(self, T, move_type):
        for _index in range(len(self.pg_current.bbx_2d)):
            self.one_step_obj_adjust(T, _index, move_type, move_face='z')

    def multi_step_scale_adjust_obj(self, T, move_type):
        for _index in range(len(self.pg_current.bbx_2d)):
            r = random.random()
            if 0 < r <= 0.5:
                self.one_step_obj_adjust(T, _index, move_type, move_face='upscale')
            else:
                self.one_step_obj_adjust(T, _index, move_type, move_face='downscale')

    def object_adjust(self, T):
        r = random.random()
        if 0 < r <= 0.8:
            move_type = 'translation'
            self.multi_step_translation_adjust_obj(T, move_type)
        elif 0.8 < r <= 0.9:
            move_type = 'rotation'
            self.multi_step_rotation_adjust_obj(T, move_type)
        elif 0.9 < r <= 1.0:
            move_type = 'scale'
            self.multi_step_scale_adjust_obj(T, move_type)

    def one_step_layout_adjust(self, T, move_ind, direction):
        pg_des = copy.deepcopy(self.pg_current)
        delta = 0.3 * direction
        pg_des.layout[move_ind] += delta
        e_total_des = self.compute_total_likelihood(pg_des, show_energy=False, count_pose=False,
                                                    count_obj=False, count_layout=True, count_hoi=False)
        gradient = (self.energy_current - e_total_des)
        if gradient == 0:
            return
        gradient_type, move_prob = self.q_moving_proposal()
        delta *= np.sign(gradient)
        if gradient_type == 1:
            delta *= -1
        pg_new = copy.deepcopy(self.pg_current)
        pg_new.layout[move_ind] += delta

        e_total_new = self.compute_total_likelihood(pg_new, show_energy=False, count_pose=False,
                                                    count_obj=False, count_layout=True, count_hoi=False)
        # accept = metropolis_hasting(self.energy_current, e_total_new, move_prob, 1 - move_prob, T)
        if e_total_new < self.energy_current:
            accept = True
        else:
            accept = False
        if accept:
            self.pg_current = copy.deepcopy(pg_new)
            self.energy_current = e_total_new
            self.energy_landscape.append(self.energy_current)
            if e_total_new <= self.energy_best:
                self.pg_best = copy.deepcopy(pg_new)
                self.energy_best = e_total_new
            self.record.append(1)
            self.save_count += 1
            if self.save_history and (self.save_count % self.save_rate) == 0:
                self.pg_history.append(pg_new)
        else:
            self.record.append(0)

    def layout_adjust(self, T):
        for _iter in range(4):
            if _iter == 3:
                target = 0
            else:
                target = _iter + 1
            # find x-y surface
            totalpoints = self.pg_current.layout.shape[0]
            bottompoints = self.pg_current.layout[0: int(totalpoints / 2), :]

            adjust_ind = [_iter, target, _iter+4, target+4]
            if _iter != 0:
                direction = bottompoints[_iter - 1] - bottompoints[_iter]
            else:
                direction = bottompoints[3] - bottompoints[_iter]

            direction = direction / np.linalg.norm(direction)
            self.one_step_layout_adjust(T, adjust_ind, direction)

    def floor_adjust(self, T):
        pg_des = copy.deepcopy(self.pg_current)
        delta = 0.05 * self.scale
        pg_des.layout[4:, 2] += delta
        e_total_des = self.compute_total_likelihood(pg_des, show_energy=False, count_pose=True,
                                                    count_obj=True, count_layout=False, count_hoi=True)
        gradient = (self.energy_current - e_total_des)
        if gradient == 0:
            return
        gradient_type, move_prob = self.q_moving_proposal()
        delta *= np.sign(gradient)
        if gradient_type == 1:
            delta *= -1
        pg_new = copy.deepcopy(self.pg_current)
        pg_new.layout[4:, 2] += delta

        e_total_new = self.compute_total_likelihood(pg_new, show_energy=False, count_pose=True,
                                                    count_obj=True, count_layout=False, count_hoi=True)
        accept = metropolis_hasting(self.energy_current, e_total_new, move_prob, 1 - move_prob, T)
        if accept:
            self.pg_current = copy.deepcopy(pg_new)
            self.energy_current = e_total_new
            self.energy_landscape.append(self.energy_current)
            if e_total_new <= self.energy_best:
                self.pg_best = copy.deepcopy(pg_new)
                self.energy_best = e_total_new
            self.record.append(1)
            self.save_count += 1
            if self.save_history and (self.save_count % self.save_rate) == 0:
                self.pg_history.append(pg_new)
        else:
            self.record.append(0)

    def infer(self):
        start = time.time()
        # initialization
        self.init_all_3d()
        if self.save_history:
            self.pg_history.append(self.pg_current)
        self.energy_landscape.append(self.energy_current)
        self.record.append(0)

        for scale in [1, 0.75, 0.5, 0.25]:
            self.scale = scale
            for T in [1, 0.01, 0.001]:
                self.stagnate_step_obj = 0
                self.stagnate_step_pose = 0
                for _ in range(self.max_step):
                    r = random.random()
                    adjust_type = None
                    if 0 < r <= 0.7:
                        adjust_type = 'pose'
                    elif 0.7 < r < 0.9:
                        adjust_type = 'object'
                    elif 0.9 < r <= 1:
                        adjust_type = 'layout'

                    if adjust_type == 'pose':
                        if self.stagnate_step_pose > self.early_end_step:
                            continue
                        else:
                            self.pose_adjust(T)
                    elif adjust_type == 'object':
                        if self.stagnate_step_obj > self.early_end_step:
                            continue
                        else:
                            self.object_adjust(T)
                    elif adjust_type == 'layout':
                        self.floor_adjust(T)

        # init hoi relation
        adjust_hoi = self.detect_hoi_load(self.pg_best)

        self.energy_current = self.compute_total_likelihood(self.pg_best, show_energy=False, count_pose=True,
                                                            count_obj=True, count_layout=True, count_hoi=True)
        self.energy_best = self.energy_current
        self.pg_current = copy.deepcopy(self.pg_best)

        if adjust_hoi:
            for scale in [1, 0.5, 0.25]:
                self.scale = scale
                for T in [1, 0.01, 0.001]:
                    self.stagnate_step_obj = 0
                    self.stagnate_step_pose = 0
                    for _ in range(self.max_step):
                        r = random.random()

                        if 0 < r <= 0.6:
                            adjust_type = 'pose'
                        elif 0.6 < r < 0.9:
                            adjust_type = 'object'
                        else:
                            adjust_type = 'layout'

                        if adjust_type == 'pose':
                            if self.stagnate_step_pose > self.early_end_step:
                                continue
                            else:
                                self.pose_adjust(T)
                        elif adjust_type == 'object':
                            if self.stagnate_step_obj > self.early_end_step:
                                continue
                            else:
                                self.object_adjust(T)

        self.finalize_hoi(self.pg_best)

        self.pg_current = copy.deepcopy(self.pg_best)
        self.energy_current = self.energy_best

        T = 1
        counter = 0
        while self.pg_best.e_layout > 0:
            counter += 1
            if counter > 200:
                break
            for _ in range(self.max_step):
                self.layout_adjust(T)
        end = time.time()
        print('time elapsed:', end-start)


def main():
    pass


if __name__ == '__main__':
    main()
