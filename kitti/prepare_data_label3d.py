''' Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'viz'))
import kitti.kitti_util as utils
import _pickle as pickle
from kitti.kitti_object import *
import argparse

# Label3d Addition
from kitti.prepare_data import *
import random as random
     
def demo():
    import mayavi.mlab as mlab
    from viz.viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'))
    data_idx = 0

    # Load data from dataset
    objects = dataset.get_label_objects(data_idx)
    objects[0].print_object()
    img = dataset.get_image(data_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_height, img_width, img_channel = img.shape
    print(('Image shape: ', img.shape))
    pc_velo = dataset.get_lidar(data_idx)[:,0:3]
    calib = dataset.get_calibration(data_idx)

    ## Draw lidar in rect camera coord
    #print(' -------- LiDAR points in rect camera coordination --------')
    #pc_rect = calib.project_velo_to_rect(pc_velo)
    #fig = draw_lidar_simple(pc_rect)
    #input()

    # # Draw 2d and 3d boxes on image
    # print(' -------- 2D/3D bounding boxes in images --------')
    # show_image_with_boxes(img, objects, calib)
    # input()
    #
    # # Show all LiDAR points. Draw 3d box in LiDAR point cloud
    # print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
    # #show_lidar_with_boxes(pc_velo, objects, calib)
    # #input()
    # show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
    # input()
    #
    # # Visualize LiDAR points on images
    # print(' -------- LiDAR points projected to image plane --------')
    # show_lidar_on_image(pc_velo, img, calib, img_width, img_height)
    # input()
    #
    # # Show LiDAR points that are in the 3d box
    # print(' -------- LiDAR points in a 3D bounding box --------')
    # box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P)
    # box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    # box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo)
    # print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))
    #
    # fig = mlab.figure(figure=None, bgcolor=(0,0,0),
    #     fgcolor=None, engine=None, size=(1000, 500))
    # draw_lidar(box3droi_pc_velo, fig=fig)
    # draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
    # mlab.show(1)
    # input()
    #
    # # UVDepth Image and its backprojection to point clouds
    # print(' -------- LiDAR points in a frustum from a 2D box --------')
    # imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
    #     calib, 0, 0, img_width, img_height, True)
    # imgfov_pts_2d = pts_2d[fov_inds,:]
    # imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    #
    # cameraUVDepth = np.zeros_like(imgfov_pc_rect)
    # cameraUVDepth[:,0:2] = imgfov_pts_2d
    # cameraUVDepth[:,2] = imgfov_pc_rect[:,2]
    #
    # # Show that the points are exactly the same
    # backprojected_pc_velo = calib.project_image_to_velo(cameraUVDepth)
    # print(imgfov_pc_velo[0:20])
    # print(backprojected_pc_velo[0:20])
    #
    # fig = mlab.figure(figure=None, bgcolor=(0,0,0),
    #     fgcolor=None, engine=None, size=(1000, 500))
    # draw_lidar(backprojected_pc_velo, fig=fig)
    # input()
    #
    # # Only display those points that fall into 2d box
    # print(' -------- LiDAR points in a frustum from a 2D box --------')
    # xmin,ymin,xmax,ymax = \
    #     objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax
    # boxfov_pc_velo = \
    #     get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax)
    # print(('2d box FOV point num: ', boxfov_pc_velo.shape[0]))
    #
    # fig = mlab.figure(figure=None, bgcolor=(0,0,0),
    #     fgcolor=None, engine=None, size=(1000, 500))
    # draw_lidar(boxfov_pc_velo, fig=fig)
    # mlab.show(1)
    # input()

def extract_frustum_data(idx_filename, split, output_filename, viz=False,
                       perturb_box=False, augmentX=1, type_whitelist=['Car']):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)
        
    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR,'dataset/KITTI/object'), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = [] # int number
    box2d_list = [] # [xmin,ymin,xmax,ymax]
    box3d_list = [] # (8,3) array in rect camera coord
    input_list = [] # channel number = 4, xyz,intensity in rect camera coord
    label_list = [] # 1 for roi object, 0 for clutter
    type_list = [] # string e.g. Car
    heading_list = [] # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = [] # array of l,w,h
    frustum_angle_list = [] # angle of 2d box center from pos x-axis

    pos_cnt = 0
    all_cnt = 0

    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
            calib, 0, 0, img_width, img_height, True)

        cubic_size = 1.75

        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist:
                continue

            for _ in range(augmentX):
                xmin, ymin, xmax, ymax = 0, 0, 0, 0
                # Augment data by box2d perturbation
                # if perturb_box2d:
                #     xmin,ymin,xmax,ymax = random_shift_box2d(box2d)
                #     print(box2d)
                #     print(xmin,ymin,xmax,ymax)
                # else:
                #     xmin,ymin,xmax,ymax = box2d

                # filter points from the image space
                # box_fov_inds = (pc_image_coord[:,0]<xmax) & \
                #     (pc_image_coord[:,0]>=xmin) & \
                #     (pc_image_coord[:,1]<ymax) & \
                #     (pc_image_coord[:,1]>=ymin)
                # box_fov_inds = box_fov_inds & img_fov_inds
                # pc_in_box_fov = pc_rect[box_fov_inds,:]

                # First get a point in ground truth
                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                _, inds = extract_pc_in_box3d(pc_rect, box3d_pts_3d)

                # Find a random point
                if np.sum(inds) == 0:
                    continue
                pos_ind_list = [i for i in range(len(inds)) if i]
                while True:
                    index = random.choice(pos_ind_list)
                    if inds[index]:
                        click_pts_id = index
                        break

                click_pt = pc_rect[click_pts_id, 0:3]
                # print(click_pt)

                # filter points from a cubic area
                # obj_center = objects[obj_idx].t
                # obj_h = objects[obj_idx].h
                # obj_center_x_max = obj_center[0] + cubic_size
                # obj_center_x_min = obj_center[0] - cubic_size
                #
                # box_center_y = obj_center[1] - (obj_h / 2.0)
                # obj_center_y_max = box_center_y - cubic_size
                # obj_center_y_min = box_center_y + cubic_size
                #
                # obj_center_z_max = obj_center[2] + cubic_size
                # obj_center_z_min = obj_center[2] - cubic_size
                #
                # box_cube_inds = (pc_rect[:, 0] < obj_center_x_max) & \
                #                 (pc_rect[:, 0] >= obj_center_x_min) & \
                #                 (pc_rect[:, 1] < obj_center_y_min) & \
                #                 (pc_rect[:, 1] >= obj_center_y_max) & \
                #                 (pc_rect[:, 2] < obj_center_z_max) & \
                #                 (pc_rect[:, 2] >= obj_center_z_min)

                # Filter points from a random click
                obj_center_x_max = click_pt[0] + cubic_size
                obj_center_x_min = click_pt[0] - cubic_size
                obj_center_y_max = click_pt[1] - cubic_size
                obj_center_y_min = click_pt[1] + cubic_size
                obj_center_z_max = click_pt[2] + cubic_size
                obj_center_z_min = click_pt[2] - cubic_size

                box_cube_inds = (pc_rect[:, 0] < obj_center_x_max) & \
                                (pc_rect[:, 0] >= obj_center_x_min) & \
                                (pc_rect[:, 1] < obj_center_y_min) & \
                                (pc_rect[:, 1] >= obj_center_y_max) & \
                                (pc_rect[:, 2] < obj_center_z_max) & \
                                (pc_rect[:, 2] >= obj_center_z_min)

                box_cube_inds = box_cube_inds & img_fov_inds
                pc_in_cube = pc_rect[box_cube_inds, :]
                # print(pc_rect.shape)
                # print(pc_in_cube.shape)

                # Get frustum angle (according to center pixel in 2D BOX)
                # box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
                # box2d_center = np.array([0, 0])
                # uvdepth = np.zeros((1, 3))
                # uvdepth[0,0:2] = box2d_center
                # uvdepth[0,2] = 20 # some random depth
                # box2d_center_rect = calib.project_image_to_rect(uvdepth)
                # frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
                #     box2d_center_rect[0,0])

                # gt_centroid = obj.obj_center
                frustum_angle = -1 * np.arctan2(click_pt[2],
                                                click_pt[0])
                # print(frustum_angle)

                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P) 
                _, inds = extract_pc_in_box3d(pc_in_cube, box3d_pts_3d)
                label = np.zeros((pc_in_cube.shape[0]))
                label[inds] = 1

                # Get 3D BOX heading
                heading_angle = obj.ry

                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])

                # Reject too far away object or object without points
                if np.sum(label) == 0:
                    continue

                id_list.append(data_idx)
                box2d_list.append(np.array([xmin, ymin, xmax, ymax]))
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_cube)
                label_list.append(label)
                type_list.append(objects[obj_idx].type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)
    
                # collect statistics
                pos_cnt += np.sum(label)
                # print(pc_in_cube.shape)
                all_cnt += pc_in_cube.shape[0]

    # print(all_cnt)
    print('Average pos ratio: %f' % (pos_cnt/float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt)/len(id_list)))
    
    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list, fp)
        pickle.dump(box3d_list, fp)
        pickle.dump(input_list, fp)
        pickle.dump(label_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(heading_list, fp)
        pickle.dump(box3d_size_list, fp)
        pickle.dump(frustum_angle_list, fp)
    
    if viz:
        import mayavi.mlab as mlab
        for i in range(10):
            p1 = input_list[i]
            seg = label_list[i] 
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            input()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo.')
    parser.add_argument('--gen_train', action='store_true', help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data with GT 2D boxes')
    parser.add_argument('--gen_val_rgb_detection', action='store_true', help='Generate val split frustum data with RGB detection 2D boxes')
    parser.add_argument('--car_only', action='store_true', help='Only generate cars; otherwise cars, peds and cycs')
    args = parser.parse_args()

    if args.demo:
        demo()
        exit()

    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_caronly_'
    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'frustum_carpedcyc_'

    if args.gen_train:
        extract_frustum_data(\
            os.path.join(BASE_DIR, 'image_sets/train.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix+'train.pickle'), 
            viz=False, perturb_box=True, augmentX=5,
            type_whitelist=type_whitelist)

    if args.gen_val:
        extract_frustum_data(\
            os.path.join(BASE_DIR, 'image_sets/val.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix+'val.pickle'),
            viz=False, perturb_box=False, augmentX=1,
            type_whitelist=type_whitelist)
    #
    # if args.gen_val_rgb_detection:
    #     extract_frustum_data_rgb_detection(\
    #         os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_val.txt'),
    #         'training',
    #         os.path.join(BASE_DIR, output_prefix+'val_rgb_detection.pickle'),
    #         viz=False,
    #         type_whitelist=type_whitelist)
