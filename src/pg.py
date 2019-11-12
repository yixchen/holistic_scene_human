"""
    Created on Jan 20, 2019
    @author: Yixin Chen
    Data structure.
"""


class PG(object):
    def __init__(self):
        self.camera_mat = None
        self.bbx_2d = None
        self.pose_2d = None
        self.bbx_label = None

        self.bbx_3d = None
        self.pose_3d = None
        self.r_mat = None
        self.layout = None

        self.support_rel = None

        self.hoi = HOI()
        self.hoi_type = None
        self.hoi_obj_label = None

        self.pose_2d_pred = None

        # energy for this pg
        self.e_total = 0
        self.e_2d3d_pose = 0
        self.e_2d3d_obj = 0
        self.e_4dhoi = 0
        self.e_physical = 0
        self.e_layout = 0
        self.e_pose_ground = 0
        self.e_support = 0


class PG_MULTIPERSON(object):
    def __init__(self):
        self.camera_mat = None
        self.bbx_2d = None
        self.pose_2d = None
        self.bbx_label = None

        self.bbx_3d = None
        self.pose_3d = None
        self.r_mat = None
        self.layout = None

        self.support_rel = None

        self.hoi = list()
        self.hoi_type = list()
        self.hoi_obj_label = list()

        self.pose_2d_pred = None

        # energy for this pg
        self.e_total = 0
        self.e_2d3d_pose = 0
        self.e_2d3d_obj = 0
        self.e_4dhoi = 0
        self.e_physical = 0
        self.e_layout = 0
        self.e_pose_ground = 0
        self.e_support = 0


class HOI(object):
    def __init__(self):
        self.type = list()
        self.obj_label = list()
        self.bbx_ind = list()


class SUNRGBD_PG(object):
    def __init__(self):
        self.sequence_id = None
        self.camera_mat = None
        self.boxes = list()
        self.new_boxes = list()
        self.r_mat = None
        self.layout = None

        self.image_w = None
        self.image_h = None

        self.pose_2d = list()
        self.pose_3d = list()
        self.pose_dis = list()
        self.action_label = list()

        # energy for this pg
        self.e_total = 0
        self.e_2d3d_pose = 0
        self.e_2d3d_obj = 0
        self.e_4dhoi = 0
        self.e_physical = 0
        self.e_layout = 0
        self.e_pose_ground = 0
        self.e_support = 0