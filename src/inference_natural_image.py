"""
    Created on Jan 20, 2019
    @author: Yixin Chen
    Joint inference of 3D scene and human pose for natural image.
"""
import os
import argparse
import numpy as np
import datetime
import time
import pickle
import scipy.io
import matplotlib.pyplot as plt
from pg import PG_MULTIPERSON, HOI
from utils import rotation_matrix_3d_z, world2camera, pose3d_all2partial, flip_toward_viewer, get_corners_of_bb3d
from visulization import vis_all_multi
from inference_update import Sampler
import scipy.io
import cv2


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OBJ_CATEGORY_CLEAN = ['recycle_bin', 'cpu', 'paper', 'toilet', 'stool', 'whiteboard', 'coffee_table', 'picture',
                      'keyboard', 'dresser', 'painting', 'bookshelf', 'night_stand', 'endtable', 'drawer', 'sink',
                      'monitor', 'computer', 'cabinet', 'shelf', 'lamp', 'garbage_bin', 'box', 'bed', 'sofa',
                      'sofa_chair', 'pillow', 'desk', 'table', 'chair']


def load_data(image_path, result_path, pg):
    if not os.path.isfile(image_path):
        print('Cannot find image file.')
        return
    if not os.path.isdir(result_path):
        print('Cannot find preliminary results.')
        return

    if os.path.isfile(os.path.join(result_path, 'bdb_2d.pickle')):
        with open(os.path.join(result_path, 'bdb_2d.pickle'), 'rb') as f:
            pg.bbx_2d = pickle.load(f, encoding='latin1')
    else:
        print('Cannot find 2d bounding box result.')
        return

    if os.path.isfile(os.path.join(result_path, 'hoi.pickle')):
        with open(os.path.join(result_path, 'hoi.pickle'), 'rb') as f:
            content = pickle.load(f, encoding='latin1')
            for _iter in range(len(pg.pose_2d)):
                pg.hoi_type.append(content[_iter]['type'])
                pg.hoi_obj_label.append(content[_iter]['obj_label'])
    else:
        print('Cannot find Human-Object interaction result.')
        return

    if os.path.isfile(os.path.join(result_path, 'bdb_label.pickle')):
        with open(os.path.join(result_path, 'bdb_label.pickle'), 'rb') as f:
            pg.bbx_label = pickle.load(f, encoding='latin1')
    else:
        print('Cannot find bounding box labels.')
        return

    if os.path.isfile(os.path.join(result_path, 'pose_2d.pickle')):
        with open(os.path.join(result_path, 'pose_2d.pickle'), 'rb') as f:
            pose_2d = pickle.load(f, encoding='latin1')
            pg.pose_2d = [single_pose[:, [1, 0]] for single_pose in pose_2d]  # natural image
            # pg.pose_2d = [single_pose for single_pose in pose_2d]  # sunrgbd
    else:
        print('Cannot find 2d pose data.')
        return

    if os.path.isfile(os.path.join(result_path, 'camera.pickle')):
        with open(os.path.join(result_path, 'camera.pickle'), 'rb') as f:
            pg.camera_mat = pickle.load(f, encoding='latin1')
    else:
        print('Cannot find camera parameters.')
        return

    if os.path.isfile(os.path.join(result_path, 'bdb_3d.pickle')):
        with open(os.path.join(result_path, 'bdb_3d.pickle'), 'rb') as f:
            bbx_3d_tmp = pickle.load(f, encoding='latin1')
            bdb3d_all = []
            for bdb3d in bbx_3d_tmp:
                bdb3d_tmp = get_corners_of_bb3d(bdb3d)
                bdb3d_all.append(bdb3d_tmp)
            pg.bbx_3d = bdb3d_all
    else:
        print('Cannot find 3d bounding box data.')
        return

    if os.path.isfile(os.path.join(result_path, 'pose_3d.pickle')):
        with open(os.path.join(result_path, 'pose_3d.pickle'), 'rb') as f:
            pg.pose_3d = pickle.load(f, encoding='latin1')
    else:
        print('Cannot find 3D pose data.')
        return

    if os.path.isfile(os.path.join(result_path, 'r_ex.pickle')):
        with open(os.path.join(result_path, 'r_ex.pickle'), 'rb') as f:
            pg.r_mat = pickle.load(f, encoding='latin1')
    else:
        print('Cannot find camera extrinsic parameters.')
        return

    if os.path.isfile(os.path.join(result_path, 'layout.pickle')):
        with open(os.path.join(result_path, 'layout.pickle'), 'rb') as f:
            pg.layout = pickle.load(f, encoding='latin1')
    else:
        print('Cannot find layout data.')
        return


def parse_args():
    parser = argparse.ArgumentParser('3D Scene and Pose Sampler')

    # previous results
    parser.add_argument('--image_name', type=str, help='name of images to be processed')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='root directory of image and preliminary results')
    # Sampler behavior
    parser.add_argument('--max_step_len', type=int, default=50, help='maximum sampling step length')
    parser.add_argument('--sample_pose', action='store_false', help='whether sample pose')
    parser.add_argument('--sample_obj', action='store_false', help='whether sample object')
    parser.add_argument('--sample_layout', action='store_false', help='whether sample layout')

    # Checkpoint
    parser.add_argument('--save_model', action='store_true', help='whether store sampler by end of inference')
    parser.add_argument('--save_history', action='store_true', help='whether store sampling history')
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(PROJ_ROOT, '../experiment', str(datetime.datetime.now())[5:19], 'history'),
                        help='directory in which state of sampling steps are saved')
    parser.add_argument('--save_rate', type=int, default=100,
                        help='save model once every time this many sampling steps are completed')

    arglist = parser.parse_args()
    if arglist.save_history:
        if not os.path.exists(arglist.save_dir):
            os.makedirs(arglist.save_dir)

    return arglist


def infer_image(arglist):
    image_path = os.path.join(arglist.data_dir, 'image', arglist.image_name)
    result_path = os.path.join(arglist.data_dir, 'result', arglist.image_name)
    pg = PG_MULTIPERSON()
    '''load previous result'''
    load_data(image_path, result_path, pg)

    sampler = Sampler(arglist, pg)
    sampler.infer()

    if arglist.save_model:
        with open(os.path.join(result_path, 'sampler.pickle'), 'wb') as handle:
            pickle.dump(sampler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    vis_all_multi(sampler, image_path, result_path, save_image=True)


def main():
    arglist = parse_args()
    infer_image(arglist)


if __name__ == '__main__':
    main()
    pass
