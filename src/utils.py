"""
    Created on Jan 20, 2019
    @author: Yixin Chen
    Utilization functions.
"""
import numpy as np
import scipy.io
from shapely.geometry.polygon import Polygon
from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


def rotation_matrix_3d_z(angle):
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(angle)
    R[1, 1] = np.cos(angle)
    R[0, 1] = -np.sin(angle)
    R[1, 0] = np.sin(angle)
    R[2, 2] = 1
    return R


def world2camera(world_coor):
    coor = world_coor.copy()
    coor = coor[:, [0, 2, 1]]
    coor[:, 1] = - coor[:, 1]
    return coor


# pose_3d_all is 17*3 and pose_3d_cam is 14*3
def pose3d_all2partial(pose_3d_all):
    pose_3d_cam = np.zeros([14, 3])
    pose_3d_cam[0, :] = pose_3d_all[10, :]
    pose_3d_cam[1, :] = pose_3d_all[8, :]
    pose_3d_cam[2, :] = pose_3d_all[14, :]
    pose_3d_cam[3, :] = pose_3d_all[15, :]
    pose_3d_cam[4, :] = pose_3d_all[16, :]
    pose_3d_cam[5, :] = pose_3d_all[11, :]
    pose_3d_cam[6, :] = pose_3d_all[12, :]
    pose_3d_cam[7, :] = pose_3d_all[13, :]
    pose_3d_cam[8, :] = pose_3d_all[1, :]
    pose_3d_cam[9, :] = pose_3d_all[2, :]
    pose_3d_cam[10, :] = pose_3d_all[3, :]
    pose_3d_cam[11, :] = pose_3d_all[4, :]
    pose_3d_cam[12, :] = pose_3d_all[5, :]
    pose_3d_cam[13, :] = pose_3d_all[6, :]
    return pose_3d_cam


def flip_toward_viewer(normals, points):

    points = np.divide(points, np.tile(np.sqrt(np.sum(np.square(points), axis=1)), (3, 1)))
    proj = np.sum(np.multiply(points, normals), axis=1)
    flip = proj > 0
    normals[flip, :] = -normals[flip, :]

    return normals


def get_corners_of_bb3d(bb3d):
    corners = np.zeros((8, 3))
    # order the basis
    basis_tmp = bb3d['basis']
    inds = np.argsort(np.abs(basis_tmp[:, 0]))[::-1]
    basis = basis_tmp[inds, :]
    coeffs = bb3d['coeffs'].T[inds]

    inds = np.argsort(np.abs(basis[1:3, 1]))[::-1]

    if inds[0] == 1:
        basis[1:3, :] = np.flip(basis[1:3, :], 0)
        coeffs[1:3] = np.flip(coeffs[1:3], 1)
    centroid = bb3d['centroid']

    basis = flip_toward_viewer(basis, np.tile(centroid, (3, 1)))
    coeffs = np.abs(coeffs)

    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]

    corners = corners + np.tile(centroid, (8, 1))
    return corners


# in 2d, return the intersection vertial length of two bbx / length of bbx1 #1
def vertical_intersect_ratio(bbx1, bbx2):
    if bbx2[0] > bbx1[2] or bbx2[2] < bbx1[0]:
        return 0
    else:
        return (np.min([bbx1[2], bbx2[2]]) - np.max([bbx1[0], bbx2[0]])) / float(bbx1[2] - bbx1[0])


# return the intersection area of two cuboid in x-y coordinates / area of cuboid #1
def intersection_2d_ratio(cu1, cu2):
    polygon_1 = Polygon([(cu1[0][0], cu1[0][1]), (cu1[1][0], cu1[1][1]), (cu1[2][0], cu1[2][1]), (cu1[3][0], cu1[3][1])])
    polygon_2 = Polygon([(cu2[0][0], cu2[0][1]), (cu2[1][0], cu2[1][1]), (cu2[2][0], cu2[2][1]), (cu2[3][0], cu2[3][1])])
    intersection_ratio = polygon_1.intersection(polygon_2).area / polygon_1.area
    return intersection_ratio


def iou_2dobj(bbx_pred, bbx_gt):
    '''
    A error metric in terms of 2d object
    :param bbx_pred: 8*2
    :param bbx_gt: 4*2
    :return: the iou of bbx_pred and bbx_gt
    '''
    hull_pred = ConvexHull(bbx_pred)
    bbx_gt = np.array([[bbx_gt[0], bbx_gt[1]],
                       [bbx_gt[2], bbx_gt[1]],
                       [bbx_gt[0], bbx_gt[3]],
                       [bbx_gt[2], bbx_gt[3]]])
    hull_gt = ConvexHull(bbx_gt)
    polygon_gt = Polygon([(bbx_gt[ind, 0], bbx_gt[ind, 1]) for ind in hull_gt.vertices])
    polygon_pred = Polygon([(bbx_pred[ind, 0], bbx_pred[ind, 1]) for ind in hull_pred.vertices])
    visualize = False
    if visualize:
        plt.plot(bbx_pred[:, 0], bbx_pred[:, 1], 'o')
        in_area = polygon_gt.intersection(polygon_pred)
        x, y = in_area.exterior.xy
        plt.plot(x, y, color='#6699cc', alpha=0.7,
        linewidth=3, solid_capstyle='round', zorder=2)
        un_area = polygon_gt.union(polygon_pred)
        x, y = un_area.exterior.xy
        plt.plot(x, y, color='#6699cc', alpha=0.7,
                 linewidth=3, solid_capstyle='round', zorder=2)
        for simplex in hull_gt.simplices:
            plt.plot(bbx_gt[simplex, 0], bbx_gt[simplex, 1], 'k-')
        for simplex in hull_pred.simplices:
            plt.plot(bbx_pred[simplex, 0], bbx_pred[simplex, 1], 'k-')
        plt.show()
    iou = polygon_gt.intersection(polygon_pred).area / polygon_gt.union(polygon_pred).area

    return iou


def get_iou_cuboid(cu1, cu2):
    """
        Calculate the Intersection over Union (IoU) of two 3D cuboid.

        Parameters
        ----------
        cu1 : numpy array, 8x3
        cu2 : numpy array, 8x3

        Returns
        -------
        float
            in [0, 1]
    """
    polygon_1 = Polygon([(cu1[0][0], cu1[0][1]), (cu1[1][0], cu1[1][1]), (cu1[2][0], cu1[2][1]), (cu1[3][0], cu1[3][1])])
    polygon_2 = Polygon([(cu2[0][0], cu2[0][1]), (cu2[1][0], cu2[1][1]), (cu2[2][0], cu2[2][1]), (cu2[3][0], cu2[3][1])])
    intersect_2d = polygon_1.intersection(polygon_2).area
    inter_vol = intersect_2d * max(0.0, min(cu1[0][2], cu2[0][2]) - max(cu1[4][2], cu2[4][2]))
    vol1 = polygon_1.area * (cu1[0][2] - cu1[4][2])
    vol2 = polygon_2.area * (cu2[0][2] - cu2[4][2])
    return inter_vol / (vol1 + vol2 - inter_vol)


class hoi_model():
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.energy_model = []
        self.energy_model.append(multivariate_normal(mean=mean[0], cov=cov[0]))
        self.energy_model.append(multivariate_normal(mean=mean[1], cov=cov[1]))
        self.energy_model.append(multivariate_normal(mean=mean[2], cov=cov[2]))

    def get_energy(self, value, axis):
        energy = self.energy_model[axis].pdf(value) / \
                 self.energy_model[axis].pdf(self.mean[axis])
        return energy


class sit_desk_hoi_model():
    def __init__(self, mean, cov, weights):
        self.mean = mean
        self.cov = cov
        self.weights = weights
        self.energy_model = []
        self.energy_model.append(multivariate_normal(mean=mean[0], cov=cov[0]))
        self.energy_model.append(multivariate_normal(mean=mean[1], cov=cov[1]))
        self.energy_model.append(multivariate_normal(mean=mean[2], cov=cov[2]))

    def get_energy(self, value):
        energy = 0
        for _iter in range(3):
            energy += self.weights[_iter]*self.energy_model[_iter].pdf(value) / \
                      self.energy_model[_iter].pdf(self.mean[_iter])
        return energy


if __name__ == '__main__':
    points = np.random.rand(8, 2)  # 30 random points in 2-D
    plt.plot(points[:, 0], points[:, 1], 'o')
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        a = simplex
        pass
    b = hull.vertices
    c = points[b]
    pass
