import numpy as np
import yaml
import os


def load_calibration(calib_path):
    calib = {}
    
    # Get calibrations
    calib['extrinsics'] = yaml.load(open(os.path.join(calib_path, 'extrinsics.yaml')), yaml.SafeLoader)
    calib['CAM00'] = yaml.load(open(os.path.join(calib_path, '00.yaml')), yaml.SafeLoader)
    calib['CAM01'] = yaml.load(open(os.path.join(calib_path, '01.yaml')), yaml.SafeLoader)
    calib['CAM02'] = yaml.load(open(os.path.join(calib_path, '02.yaml')), yaml.SafeLoader)
    calib['CAM03'] = yaml.load(open(os.path.join(calib_path, '03.yaml')), yaml.SafeLoader)
    calib['CAM04'] = yaml.load(open(os.path.join(calib_path, '04.yaml')), yaml.SafeLoader)
    calib['CAM05'] = yaml.load(open(os.path.join(calib_path, '05.yaml')), yaml.SafeLoader)
    calib['CAM06'] = yaml.load(open(os.path.join(calib_path, '06.yaml')), yaml.SafeLoader)
    calib['CAM07'] = yaml.load(open(os.path.join(calib_path, '07.yaml')), yaml.SafeLoader)

    return calib

class Calibration(object):
    def __init__(self, calib_file):
        if isinstance(calib_file, str):
            calib = load_calibration(calib_file)
        else:
            calib = calib_file
    
        # Projection matrix from camera to image
        self.t_img_cam = []
        # Projection matrix from camera to lidar
        self.t_cam_lidar = []
        for cam in range(8):
            t_img_cam = np.eye(4)
            t_img_cam[0:3, 0:3] = np.array(calib['CAM0' + str(cam)]['camera_matrix']['data']).reshape(-1, 3)
            t_img_cam = t_img_cam[0:3, 0:4]  # remove last row
            t_cam_lidar = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM0' + str(cam)]))

            self.t_img_cam.append(t_img_cam)
            self.t_cam_lidar.append(t_cam_lidar)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        Note: Using camera 0 as rectified frame
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        t_lidar_cam = np.linalg.inv(self.t_cam_lidar[0])
        return np.matmul(t_lidar_cam,pts_rect_hom.T).T[:,0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        return np.matmul(self.t_cam_lidar[0],pts_lidar_hom.T).T[:,0:3]

    def rect_to_img(self, pts_rect, cam=0):
        """
        :param pts_rect: (N, 3)
        :param cam: Int, camera number to project onto
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.t_img_cam[cam].T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.t_img_cam[cam].T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar, cam=0):
        """
        :param pts_lidar: (N, 3)
        :param cam: Int, camera number to project onto
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect, cam=cam)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        raise NotImplementedError

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        raise NotImplementedError
