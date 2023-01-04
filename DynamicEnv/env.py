import sys
import os
import numpy as np
from numpy import newaxis as na
import pandas
import pybullet as p
import pybullet_data
import gym
from gym import spaces
import time
import math
from random import choice
from typing import Tuple, List
import logging
import copy
from collections import deque
from sklearn.metrics.pairwise import euclidean_distances

CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(os.path.dirname(CURRENT_PATH))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.dirname(CURRENT_PATH))
from pybullet_util import MotionExecute
from math_util import quaternion_matrix, euler_from_matrix, euler_from_quaternion
from rays_to_indicator import RaysCauculator


class Env(gym.Env):
    def __init__(
            self,
            is_render: bool = False,
            is_good_view: bool = False,
            is_train: bool = True,
            show_boundary: bool = True,
            add_moving_obstacle: bool = False,
            moving_obstacle_speed: float = 0.15,
            moving_init_direction: int = -1,
            moving_init_axis: int = 0,
            workspace: list = [-0.4, 0.4, 0.3, 0.7, 0.2, 0.4],
            max_steps_one_episode: int = 1024,
            num_obstacles: int = 3,
            prob_obstacles: float = 0.8,
            obstacle_box_size: list = [0.04, 0.04, 0.002],
            obstacle_sphere_radius: float = 0.04
    ):
        '''
        is_render: start GUI
        is_good_view: slow down the motion to have a better look
        is_tarin: training or testing
        '''
        self.is_render = is_render
        self.is_good_view = is_good_view
        self.is_train = is_train
        self.DISPLAY_BOUNDARY = show_boundary
        self.extra_obst = add_moving_obstacle
        if self.is_render:
            self.physicsClient = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # set the area of the workspace
        self.x_low_obs = workspace[0]
        self.x_high_obs = workspace[1]
        self.y_low_obs = workspace[2]
        self.y_high_obs = workspace[3]
        self.z_low_obs = workspace[4]
        self.z_high_obs = workspace[5]

        # for the moving
        self.direction = moving_init_direction
        self.moving_xy = moving_init_axis  # 0 for x, 1 for y
        self.moving_obstacle_speed = moving_obstacle_speed
        # action sapce
        self.action = None
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)  # angular velocities

        # set DataFrame for transforming segmentation mask to an RGB image
        translation_dic_red = pandas.DataFrame.from_dict(
            {-1: 230, 0: 60, 1: 255, 2: 0, 3: 245, 4: 145, 5: 70, 6: 240, 7: 210, 8: 250, 9: 0,
             10: 220, 11: 170}, orient="index", columns=["r"])
        translation_dic_green = pandas.DataFrame.from_dict(
            {-1: 25, 0: 180, 1: 225, 2: 130, 3: 130, 4: 30, 5: 240, 6: 50, 7: 245, 8: 190, 9: 128,
             10: 190, 11: 110}, orient="index", columns=["g"])
        translation_dic_blue = pandas.DataFrame.from_dict(
            {-1: 75, 0: 75, 1: 25, 2: 200, 3: 48, 4: 180, 5: 240, 6: 230, 7: 60, 8: 212, 9: 128,
             10: 255, 11: 40}, orient="index", columns=["b"])
        self.df_seg_to_rgb = pandas.concat([translation_dic_red, translation_dic_green, translation_dic_blue], axis=1)

        # parameters for spatial infomation
        self.home = [0, np.pi / 2, -np.pi / 6, -2 * np.pi / 3, -4 * np.pi / 9, np.pi / 2, 0.0]
        self.target_position = None
        self.obsts = None
        self.current_pos = None
        self.current_orn = None
        self.current_joint_position = None
        self.vel_checker = 0
        self.past_distance = deque([])

        # observation space
        self.state = np.zeros((14,), dtype=np.float32)
        # self.obs_rays = np.zeros(shape=(129,), dtype=np.float32)
        # self.indicator = np.zeros((10,), dtype=np.int8)
        self.distances_to_obstacles = np.zeros((8,), dtype=np.float32)
        self.link_center_of_mass_coordinates = np.zeros((8, 3), dtype=np.float32)

        obs_spaces = {
            'position': spaces.Box(low=-2, high=2, shape=(14,), dtype=np.float32),
            #'indicator': spaces.Box(low=0, high=2, shape=(10,), dtype=np.int8),
            "distances_to_obstacles": spaces.Box(low=0, high=100, shape=(8,), dtype=np.float32),
            "link_center_of_mass_coordinates": spaces.Box(low=-100, high=100, shape=(8, 3), dtype=np.float32)
        }

        self.observation_space = spaces.Dict(obs_spaces)

        # step counter
        self.step_counter = 0
        # max steps in one episode
        self.max_steps_one_episode = max_steps_one_episode
        # whether collision
        self.collided = None
        # path to urdf of robot arm
        self.urdf_root_path = os.path.join(BASE, 'ur5_description/urdf/ur5.urdf')
        # link indexes
        self.base_link = 1
        self.effector_link = 7
        # obstacles
        self.num_obstacles = num_obstacles
        self.prob_obstacles = prob_obstacles
        self.obstacle_box_size = obstacle_box_size
        self.obstacle_sphere_radius = obstacle_sphere_radius

        # set image width and height
        self._set_camera_matrices()
        self.img_width = 176
        self.img_height = 120

        # parameters of augmented targets for training
        if self.is_train:
            self.distance_threshold = 0.4
            self.distance_threshold_last = 0.4
            self.distance_threshold_increment_p = 0.001
            self.distance_threshold_increment_m = 0.01
            self.distance_threshold_max = 0.4
            self.distance_threshold_min = 0.01
        # parameters of augmented targets for testing
        else:
            self.distance_threshold = 0.03
            self.distance_threshold_last = 0.03
            self.distance_threshold_increment_p = 0.0
            self.distance_threshold_increment_m = 0.0
            self.distance_threshold_max = 0.03
            self.distance_threshold_min = 0.03

        self.episode_counter = 0
        self.episode_interval = 50
        self.success_counter = 0

        # # for camera debugging
        # self.x_offset = p.addUserDebugParameter("_x_offset", -2, 2, 0)
        # self.y_offset = p.addUserDebugParameter("_y_offset", -2, 2, 0.4)
        # self.z_offset = p.addUserDebugParameter("_z_offset", -2, 2, 0.5)

    def _get_image(self) -> Tuple[List[float], List[int]]:
        """
        Retrieve a depth image and a segmentation mask from the camera sensor.
        :return: The depth image and the segmentation mask
        :rtype: Tuple[List[float], List[int]]
        """
        depthImg, segImg = p.getCameraImage(
            width=self.img_width,
            height=self.img_height,
            viewMatrix=self.camera_view_matrix,
            projectionMatrix=self.camera_proj_matrix)[3:]

        return depthImg, segImg

    def _depth_img_to_point_cloud(self, depth: np.array) -> np.array:
        """
        Compute a point cloud from a given depth image. The computation is done according to this stackoverflow post:
        https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
        :param depth: input depth image;
        the amount of points in the image should equal the product of the camera sensors pixel width and height
        :type depth: np.array
        :return: The point cloud in the shape [width x height, 3]
        :rtype: np.array
        """
        # set width and height
        W = np.arange(0, self.img_width)
        H = np.arange(0, self.img_height)

        # compute pixel coordinates
        X = ((2 * W - self.img_width) / self.img_width)[na, :].repeat(self.img_height, axis=0).flatten()[:, na]
        Y = (-1 * (2 * H - self.img_height) / self.img_height)[:, na].repeat(self.img_width, axis=1).flatten()[:, na]
        Z = (2 * depth - 1).flatten()[:, na]

        # transform pixel coordinates into real world ones
        num_of_pixels = self.img_width * self.img_height
        PixPos = np.concatenate([X, Y, Z, np.ones(num_of_pixels)[:, na]], axis=1)
        points = np.tensordot(self.tran_pix_world, PixPos, axes=(1, 1)).swapaxes(0, 1)
        points = (points / points[:, 3][:, na])[:, 0:3]

        return points

    def _prepreprocess_point_cloud(self, points: np.array, segImg: np.array) -> Tuple[np.array, np.array]:
        """
        Preprocess a point cloud by removing its points for the background, the points for the target and
        the points for the robot arm
        :param points: an array containing the x, y and z coordinates
        of the point cloud in the shape [width x height, 3]
        :type points: np.array
        :param segImg: an array containing the segmentation mask given by pybullet; number of entries needs to equal
        width x height
        :type segImg: np.array
        :return: the points of the point cloud with the points for the background, robot arm and target removed
        :rtype: np.array
        """
        # Points that have the same color as the first point in the point cloud are removed
        # Points that have the color [60, 180, 75] are removed, as this is the color used for the target point
        segImg = segImg.flatten()
        select_mask = segImg > 0
        points = points[select_mask]
        segImg = segImg[select_mask]

        # get the index in the segmentation mask of the robot arm
        robot_arm_index = np.bincount(segImg).argmax()

        # remove robot arm
        points = points[segImg != robot_arm_index]

        return points

    def _set_min_distances_of_robot_to_obstacle(self) -> np.array:
        """
        Compute and set the minimal euclidean distance between the coordinates of the centers of mass of the robot links
        to the obstacles points given by the scene point cloud.
        :return: obstacle points
        :rtype: np.array
        """
        # get depth image and segmentation mask
        depth, seg = self._get_image()

        # compute point cloud of obstacles
        points = self._depth_img_to_point_cloud(depth)

        # preprocess point cloud
        points = self._prepreprocess_point_cloud(points, seg)

        # p.addUserDebugPoints(points, np.tile([0, 0, 255], points.shape[0]).reshape(points.shape), pointSize=3, lifeTime=0.00001)

        # get coordinates of every links center of mass
        link_center_of_mass_coordinates = []
        for i in range(p.getNumJoints(self.RobotUid)):
            link_center_of_mass_coordinates.append(p.getLinkState(self.RobotUid, i)[0])
        self.link_center_of_mass_coordinates = np.asarray(link_center_of_mass_coordinates, dtype=np.float32)
        # Compute minimal euclidean distance from the coordinates of every links center of mass to every obstacle point
        if points.shape[0] == 0:
            distances_to_obstacles = euclidean_distances(self.link_center_of_mass_coordinates,
                                                         np.array([100, 100, 100])[na, :]).min(axis=1).round(10)
        else:
            distances_to_obstacles = euclidean_distances(self.link_center_of_mass_coordinates, points).min(axis=1).round(10)

        self.distances_to_obstacles = distances_to_obstacles.astype(np.float32)

        return points

    def _seg_mask_to_rgb(self, segImg: np.array) -> np.array:
        """
        Transform a pybullet segmentation mask into an RGB image by assigning a unqiue color to each class.
        :param segImg: a pybullet segmentation mask; its amount of entries should equal
        the product of the camera sensors pixel width and height
        :type segImg: np.array
        """
        segImg = segImg.flatten()
        return np.asarray(self.df_seg_to_rgb.loc[segImg])

    def _set_camera_matrices(self) -> None:
        """
        Set the camera sensors view and projection matrices by hardcoding the values within this method.
        """
        cameraEyePosition = np.array([0,
                                      1,
                                      0.5])

        # link_position = p.getLinkState(self.RobotUid, 4)[0]
        # cameraEyePosition = np.array([link_position[0] + p.readUserDebugParameter(self.x_offset),
        #                               link_position[1] + p.readUserDebugParameter(self.y_offset),
        #                               p.readUserDebugParameter(self.z_offset)])
        # The target of the camera sensor is the middle point of the work space
        cameraTargetPosition = np.array([(self.x_high_obs + self.x_low_obs) / 2,
                                         (self.y_high_obs + self.y_low_obs) / 2,
                                         (self.z_high_obs + self.z_low_obs) / 2])

        # The up-vector is chosen so that it is orthogonal to the vector between the camera eye and the camera target
        cameraUpVector = np.array([0, -0.6, 1])

        # Set the view matrix with hardcoded values
        self.camera_view_matrix = p.computeViewMatrix(
            cameraEyePosition, cameraTargetPosition, cameraUpVector
        )

        # # Add a debug line between the camera eye and the up vector in green and one between the eye and the target
        # # in red
        p.addUserDebugLine(cameraEyePosition, cameraTargetPosition, lineColorRGB=[1, 0, 0], lifeTime=0.2)
        p.addUserDebugLine(cameraEyePosition, cameraEyePosition + cameraUpVector, lineColorRGB=[0, 1, 0], lifeTime=0.2)
        # p.addUserDebugLine(cameraEyePosition, cameraEyePosition - np.array([0, 0, 2]), lineColorRGB=[0, 0, 1], lifeTime=0.2)
        # Set the projection matrix with hardcoded values
        self.camera_proj_matrix = p.computeProjectionMatrixFOV(
            fov=100,
            aspect=1,
            nearVal=0.1,
            farVal=1)

        # set transformation matrix to retrieve real world coordinates from pixel coordinates
        self.tran_pix_world = np.linalg.inv(np.matmul(np.asarray(self.camera_proj_matrix).reshape([4, 4], order='F'),
                                                      np.asarray(self.camera_view_matrix).reshape([4, 4], order='F')))

    def _set_home(self):

        rand = np.float32(np.random.rand(3, ))
        init_x = (self.x_low_obs + self.x_high_obs) / 2 + 0.5 * (rand[0] - 0.5) * (self.x_high_obs - self.x_low_obs)
        init_y = (self.y_low_obs + self.y_high_obs) / 2 + 0.5 * (rand[1] - 0.5) * (self.y_high_obs - self.y_low_obs)
        init_z = (self.z_low_obs + self.z_high_obs) / 2 + 0.5 * (rand[2] - 0.5) * (self.z_high_obs - self.z_low_obs)
        init_home = [init_x, init_y, init_z]

        rand_orn = np.float32(np.random.uniform(low=-np.pi, high=np.pi, size=(3,)))
        init_orn = np.array([np.pi, 0, np.pi] + 0.1 * rand_orn)
        return init_home, init_orn

    def _create_visual_box(self, halfExtents):
        visual_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0.5, 0.5, 0.5, 1])
        return visual_id

    def _create_collision_box(self, halfExtents):
        collision_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=halfExtents)
        return collision_id

    def _create_visual_sphere(self, radius):
        visual_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=[0.5, 0.5, 0.5, 1])
        return visual_id

    def _create_collision_sphere(self, radius):
        collision_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius)
        return collision_id

    def _add_obstacles(self):
        rand = np.float32(np.random.rand(3, ))
        target_x = self.x_low_obs + rand[0] * (self.x_high_obs - self.x_low_obs)
        target_y = self.y_low_obs + rand[1] * (self.y_high_obs - self.y_low_obs)
        target_z = self.z_low_obs + rand[2] * (self.z_high_obs - self.z_low_obs)
        target_position = [target_x, target_y, target_z]
        # print (target_position)
        target = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1]),
            basePosition=target_position,
        )
        obsts = []
        for i in range(self.num_obstacles):
            obst_position = target_position
            val = False
            type = np.random.random()
            rate = np.random.random()
            if (rate < self.prob_obstacles) and (type > 0.5):
                cc = 0
                while not val:
                    cc += 1
                    rand = np.float32(np.random.rand(3, ))
                    obst_x = self.x_high_obs - rand[0] * (self.x_high_obs - self.x_low_obs)
                    obst_y = self.y_high_obs - rand[1] * (self.y_high_obs - self.y_low_obs)
                    obst_z = self.z_high_obs - rand[2] * (self.z_high_obs - self.z_low_obs)
                    obst_position = [obst_x, obst_y, obst_z]
                    diff = abs(np.asarray(target_position) - np.asarray(obst_position))
                    diff_2 = abs(np.asarray(self.init_home) - np.asarray(obst_position))
                    val = (diff > 0.05).all() and (np.linalg.norm(diff) < 0.5) and (diff_2 > 0.05).all() and (
                            np.linalg.norm(diff_2) < 0.5)
                    if cc > 100:
                        val = True
                if cc <= 100:
                    halfExtents = list(np.float32(np.random.uniform(0.8, 1.2) * np.array(self.obstacle_box_size)))
                    obst_orientation = [[0.707, 0, 0, 0.707], [0, 0.707, 0, 0.707], [0, 0, 0.707, 0.707]]
                    obst_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=self._create_visual_box(halfExtents),
                        baseCollisionShapeIndex=self._create_collision_box(halfExtents),
                        basePosition=obst_position,
                        baseOrientation=choice(obst_orientation)
                    )
                    obsts.append(obst_id)
            if (rate < self.prob_obstacles) and (type <= 0.5):
                cc = 0
                while not val:
                    cc += 1
                    rand = np.float32(np.random.rand(3, ))
                    obst_x = self.x_high_obs - rand[0] * (self.x_high_obs - self.x_low_obs)
                    obst_y = self.y_high_obs - rand[1] * 0.5 * (self.y_high_obs - self.y_low_obs)
                    obst_z = self.z_high_obs - rand[2] * (self.z_high_obs - self.z_low_obs)
                    obst_position = [obst_x, obst_y, obst_z]
                    diff = abs(np.asarray(target_position) - np.asarray(obst_position))
                    diff_2 = abs(np.asarray(self.init_home) - np.asarray(obst_position))
                    val = (diff > 0.05).all() and (np.linalg.norm(diff) < 0.4) and (diff_2 > 0.05).all() and (
                            np.linalg.norm(diff_2) < 0.4)
                    if cc > 100:
                        val = True
                if cc <= 100:
                    radius = np.float32(np.random.uniform(0.8, 1.2)) * self.obstacle_sphere_radius
                    obst_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=self._create_visual_sphere(radius),
                        baseCollisionShapeIndex=self._create_collision_sphere(radius),
                        basePosition=obst_position,
                    )
                    obsts.append(obst_id)
        return target_position, obsts

    def _add_moving_plate(self):
        pos = copy.copy(self.target_position)
        if self.moving_xy == 0:
            pos[0] = self.x_high_obs - np.random.random() * (self.x_high_obs - self.x_low_obs)
        if self.moving_xy == 1:
            pos[1] = self.y_high_obs - np.random.random() * (self.y_high_obs - self.y_low_obs)

        pos[2] += 0.05
        obst_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._create_visual_box([0.05, 0.05, 0.002]),
            baseCollisionShapeIndex=self._create_collision_box([0.05, 0.05, 0.002]),
            basePosition=pos
        )
        return obst_id

    def reset(self):
        p.resetSimulation()

        self.init_home, self.init_orn = self._set_home()

        self.target_position, self.obsts = self._add_obstacles()

        if self.extra_obst:
            self.moving_xy = choice([0, 1])
            self.barrier = self._add_moving_plate()
            self.obsts.append(self.barrier)

        # reset
        self.step_counter = 0
        self.collided = False

        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated = False
        p.setGravity(0, 0, 0)

        # display boundary
        if self.DISPLAY_BOUNDARY:
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_low_obs],
                               lineToXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_low_obs],
                               lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_low_obs],
                               lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_high_obs, self.y_high_obs, self.z_low_obs],
                               lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
                               lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs],
                               lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
                               lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs],
                               lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_low_obs],
                               lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_low_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_low_obs],
                               lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_low_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_low_obs],
                               lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_low_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_low_obs],
                               lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_low_obs])

        # load the robot arm
        baseorn = p.getQuaternionFromEuler([0, 0, 0])
        self.RobotUid = p.loadURDF(self.urdf_root_path, basePosition=[0.0, -0.12, 0.5], baseOrientation=baseorn,
                                   useFixedBase=True)
        self.motionexec = MotionExecute(self.RobotUid, self.base_link, self.effector_link)
        # robot goes to the initial position
        self.motionexec.go_to_target(self.init_home, self.init_orn)

        # get position observation
        self.current_pos = p.getLinkState(self.RobotUid, self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid, self.effector_link)[5]

        self.current_joint_position = [0]

        # # get lidar observation
        # lidar_results = self._set_lidar_cylinder()
        # for i, ray in enumerate(lidar_results):
        #     self.obs_rays[i] = ray[2]
        # rc = RaysCauculator(self.obs_rays)
        # self.indicator = rc.get_indicator()

        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])

        self.episode_counter += 1
        if self.episode_counter % self.episode_interval == 0:
            self.distance_threshold_last = self.distance_threshold
            success_rate = self.success_counter / self.episode_interval
            self.success_counter = 0
            if success_rate < 0.8 and self.distance_threshold < self.distance_threshold_max:
                self.distance_threshold += self.distance_threshold_increment_p
            elif success_rate >= 0.8 and self.distance_threshold > self.distance_threshold_min:
                self.distance_threshold -= self.distance_threshold_increment_m
            elif success_rate == 1 and self.distance_threshold == self.distance_threshold_min:
                self.distance_threshold == self.distance_threshold_min
            else:
                self.distance_threshold = self.distance_threshold_last
            if self.distance_threshold <= self.distance_threshold_min:
                self.distance_threshold = self.distance_threshold_min
            print('current distance threshold: ', self.distance_threshold)

        # do this step in pybullet
        p.stepSimulation()

        # if the simulation starts with the robot arm being inside an obstacle, reset again
        if p.getContactPoints(self.RobotUid) != ():
            self.reset()

        # input("Press ENTER")

        return self._get_obs()

    def step(self, action):

        # set a coefficient to prevent the action from being too large
        self.action = action
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        droll = action[3] * dv
        dpitch = action[4] * dv
        dyaw = action[5] * dv

        self.current_pos = p.getLinkState(self.RobotUid, self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid, self.effector_link)[5]
        current_rpy = euler_from_quaternion(self.current_orn)
        new_robot_pos = [self.current_pos[0] + dx,
                         self.current_pos[1] + dy,
                         self.current_pos[2] + dz]
        new_robot_rpy = [current_rpy[0] + droll,
                         current_rpy[1] + dpitch,
                         current_rpy[2] + dyaw]
        self.motionexec.go_to_target(new_robot_pos, new_robot_rpy)

        if self.extra_obst:
            barr_pos = np.asarray(p.getBasePositionAndOrientation(self.barrier)[0])
            if self.moving_xy == 0:
                if barr_pos[0] > self.x_high_obs or barr_pos[0] < self.x_low_obs:
                    self.direction = -self.direction
                barr_pos[0] += self.direction * self.moving_obstacle_speed * dv
                p.resetBasePositionAndOrientation(self.barrier, barr_pos,
                                                  p.getBasePositionAndOrientation(self.barrier)[1])
            if self.moving_xy == 1:
                if barr_pos[1] > self.y_high_obs or barr_pos[1] < self.y_low_obs:
                    self.direction = -self.direction
                barr_pos[1] += self.direction * self.moving_obstacle_speed * dv
                p.resetBasePositionAndOrientation(self.barrier, barr_pos,
                                                  p.getBasePositionAndOrientation(self.barrier)[1])

        # update current pose
        self.current_pos = p.getLinkState(self.RobotUid, self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid, self.effector_link)[5]
        self.current_joint_position = [0]
        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])

        # logging.debug("self.current_pos={}\n".format(self.current_pos))
        #
        # # get lidar observation
        # lidar_results = self._set_lidar_cylinder()
        # for i, ray in enumerate(lidar_results):
        #     self.obs_rays[i] = ray[2]
        # # print (self.obs_rays)
        # rc = RaysCauculator(self.obs_rays)
        # self.indicator = rc.get_indicator()

        # print (self.indicator)    
        # check collision
        for i in range(len(self.obsts)):
            contacts = p.getContactPoints(bodyA=self.RobotUid, bodyB=self.obsts[i])
            if len(contacts) > 0:
                self.collided = True

        p.stepSimulation()
        if self.is_good_view:
            time.sleep(0.02)

        self._set_camera_matrices()
        self.step_counter += 1
        # input("Press ENTER")
        return self._reward()

    def _reward(self):
        reward = 0

        # distance between torch head and target postion
        self.distance = np.linalg.norm(np.asarray(list(self.current_pos)) - np.asarray(self.target_position), ord=None)
        # print(self.distance)
        # check if out of boundary      
        x = self.current_pos[0]
        y = self.current_pos[1]
        z = self.current_pos[2]
        out = bool(
            x < self.x_low_obs
            or x > self.x_high_obs
            or y < self.y_low_obs
            or y > self.y_high_obs
            or z < self.z_low_obs
            or z > self.z_high_obs
        )
        # check shaking
        shaking = 0
        if len(self.past_distance) >= 10:
            arrow = []
            for i in range(0, 9):
                arrow.append(0) if self.past_distance[i + 1] - self.past_distance[i] >= 0 else arrow.append(1)
            for j in range(0, 8):
                if arrow[j] != arrow[j + 1]:
                    shaking += 1
        reward -= shaking * 0.005
        # success
        is_success = False
        if out:
            self.terminated = True
            reward += -5
        elif self.collided:
            self.terminated = True
            reward += -10
        elif self.distance < self.distance_threshold:
            self.terminated = True
            is_success = True
            self.success_counter += 1
            reward += 10
        # not finish when reaches max steps
        elif self.step_counter >= self.max_steps_one_episode:
            self.terminated = True
            reward += -1
        # this episode goes on
        else:
            self.terminated = False
            reward += -0.01 * self.distance

            # add reward for distance to obstacles
            reward += -1 * np.exp(-15 * self.distances_to_obstacles.min())

        info = {'step': self.step_counter,
                'out': out,
                'distance': self.distance,
                'reward': reward,
                'collided': self.collided,
                'shaking': shaking,
                'is_success': is_success,
                "min_distance_to_obstacles": self.distances_to_obstacles.min()}

        if self.terminated:
            print(info)
            # logger.debug(info)
        return self._get_obs(), reward, self.terminated, info

    def _get_obs(self):
        # set camera matrices
       # self._set_camera_matrices()

        # set distance between robot arm and obstacles points
        self._set_min_distances_of_robot_to_obstacle()

        # Set other observations
        self.state[0:6] = self.current_joint_position[1:]
        self.state[6:9] = np.asarray(self.target_position) - np.asarray(self.current_pos)
        self.state[9:13] = self.current_orn
        self.distance = np.linalg.norm(np.asarray(list(self.current_pos)) - np.asarray(self.target_position), ord=None)
        self.past_distance.append(self.distance)

        if len(self.past_distance) > 10:
            self.past_distance.popleft()
        self.state[13] = self.distance

        return {
            'position': self.state,
            # 'indicator': self.indicator,
            "distances_to_obstacles": self.distances_to_obstacles,
            "link_center_of_mass_coordinates": self.link_center_of_mass_coordinates
        }

    def _set_lidar_cylinder(self, ray_min=0.02, ray_max=0.2, ray_num_ver=10, ray_num_hor=12, render=False):
        global x_start
        ray_froms = []
        ray_tops = []
        frame = quaternion_matrix(self.current_orn)
        frame[0:3, 3] = self.current_pos
        ray_froms.append(list(self.current_pos))
        ray_tops.append(np.matmul(np.asarray(frame), np.array([0.0, 0.0, ray_max, 1]).T)[0:3].tolist())

        for angle in range(230, 270, 20):
            for i in range(ray_num_hor):
                z = -ray_max * math.sin(angle * np.pi / 180)
                l = ray_max * math.cos(angle * np.pi / 180)
                x_end = l * math.cos(2 * math.pi * float(i) / ray_num_hor)
                y_end = l * math.sin(2 * math.pi * float(i) / ray_num_hor)
                start = list(self.current_pos)
                end = np.matmul(np.asarray(frame), np.array([x_end, y_end, z, 1]).T)[0:3].tolist()
                ray_froms.append(start)
                ray_tops.append(end)

        # set the angle of rays
        interval = -0.005

        for i in range(8):
            ai = i * np.pi / 4
            for angle in range(ray_num_ver):
                z_start = (angle) * interval
                x_start = ray_min * math.cos(ai)
                y_start = ray_min * math.sin(ai)
                start = np.matmul(np.asarray(frame), np.array([x_start, y_start, z_start, 1]).T)[0:3].tolist()
                z_end = (angle) * interval
                x_end = ray_max * math.cos(ai)
                y_end = ray_max * math.sin(ai)
                end = np.matmul(np.asarray(frame), np.array([x_end, y_end, z_end, 1]).T)[0:3].tolist()
                ray_froms.append(start)
                ray_tops.append(end)

        for angle in range(230, 270, 20):
            for i in range(ray_num_hor):
                z = -0.15 + ray_max * math.sin(angle * np.pi / 180)
                l = ray_max * math.cos(angle * np.pi / 180)
                x_end = l * math.cos(2 * math.pi * float(i) / ray_num_hor)
                y_end = l * math.sin(2 * math.pi * float(i) / ray_num_hor)

                start = np.matmul(np.asarray(frame), np.array([x_start, y_start, z_start - 0.15, 1]).T)[0:3].tolist()
                end = np.matmul(np.asarray(frame), np.array([x_end, y_end, z, 1]).T)[0:3].tolist()
                ray_froms.append(start)
                ray_tops.append(end)
        results = p.rayTestBatch(ray_froms, ray_tops)

        if render:
            hitRayColor = [0, 1, 0]
            missRayColor = [1, 0, 0]

            p.removeAllUserDebugItems()

            for index, result in enumerate(results):
                if result[0] == -1:
                    p.addUserDebugLine(ray_froms[index], ray_tops[index], missRayColor)
                else:
                    p.addUserDebugLine(ray_froms[index], ray_tops[index], hitRayColor)
        return results


if __name__ == '__main__':
    env = Env(is_render=False, is_good_view=False, add_moving_obstacle=False)
    episodes = 100
    for episode in range(episodes):
        state = env.reset()
        done = False
        i = 0
        time_for_distance_to_obstacles = []
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            # print(info)
