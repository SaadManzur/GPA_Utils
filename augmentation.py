from __future__ import print_function

import numpy as np
import cv2 as opencv
from constants import HIP, RUPLEG, PARENTS
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def local_vectors(joints):

    rhip = joints[RUPLEG, :]
    hip = joints[HIP, :]

    up = np.array([0, 1, 0], dtype=np.float32)
    right = np.array([
        rhip[0] - hip[0],
        0.0,
        rhip[2] - hip[2]
    ])
    right /= np.linalg.norm(right)
    forward = np.cross(up, right)
    forward /= np.linalg.norm(forward)

    return up, right, forward


def draw_skeleton(joints, subplot):

    subplot.scatter(joints[:, 0], joints[:, 1], zs=joints[:, 2], color='k')

    max_, min_ = 5, -5

    for i in range(joints.shape[0]):
        parent = PARENTS[i]

        if parent != -1:
            subplot.plot(
                [joints[i, 0], joints[parent, 0]],
                [joints[i, 1], joints[parent, 1]],
                zs=[joints[i, 2], joints[parent, 2]],
                color='k'
            )

    subplot.set_xlim((min_, max_))
    subplot.set_ylim((min_, max_))
    subplot.set_zlim((min_, max_))

    subplot.set_xlabel("X")
    subplot.set_ylabel("Y")
    subplot.set_zlabel("Z")


def plot_vector(origin, direction, subplot, color='g'):

    end = origin + direction*0.5

    subplot.plot(
        [origin[0], end[0]],
        [origin[1], end[1]],
        zs=[origin[2], end[2]],
        color=color
    )


def process_frame(subject, extrinsic):

    cam_pos = extrinsic[:3, 3]
    cam_rot = extrinsic[:3, :3]

    up, right, forward = local_vectors(subject)
    body_matrix = np.array([right, up, forward]).T

    cam_to_body = np.matmul(body_matrix.T, cam_rot)
    cam_pos_body = cam_pos - subject[HIP, :]

    return body_matrix, cam_to_body, cam_pos_body


def transform(filename, save_filename):

    data = np.load(filename, allow_pickle=True)['data'].item()
    calculated_data = {}

    exclusion = ['size']

    for key in data.keys():

        if key in exclusion:
            continue

        pos_3d_world = []
        body_matrix = []
        cam_to_body = []
        cam_pos_body = []
        cam_extrinsics_org = []
        fnames = []
        intrinsics = {}

        print("Processing ", key)

        for i in tqdm(range(data[key]['3d_world'].shape[0])):
            frame = data[key]['3d_world'][i]

            instance = frame.reshape((34, 3))
            cam_rot = opencv.Rodrigues(data[key]['orientation'][i, :, 0])[0]
            cam_rot = cam_rot.reshape((3, 3))
            cam_tx = data[key]['translation'][i, :, 0]*10
            cam_extrinsic = np.zeros((4, 4))
            cam_extrinsic[:3, :3] = cam_rot
            cam_extrinsic[:3, 3] = cam_tx
            cam_extrinsic[3, 3] = 1

            body_mat, tx_mat, tx_dist = process_frame(instance, cam_extrinsic)

            pos_3d_world.append(instance)
            body_matrix.append(body_mat)
            cam_to_body.append(tx_mat)
            cam_pos_body.append(tx_dist)
            cam_extrinsics_org.append(cam_extrinsic)
            fnames.append(key)

        intrinsics['center'] = data[key]['center']
        intrinsics['focus'] = data[key]['focus']
        intrinsics['distortion'] = data[key]['distortion']

        calculated_data[key] = {
            'pos_3d_world': pos_3d_world,
            'body_matrix': body_matrix,
            'cam_to_body': cam_to_body,
            'cam_pos_body': cam_pos_body,
            'cam_extrinsics_org': cam_extrinsics_org,
            'fnames': fnames,
            'cam_intrinsics': intrinsics
        }

    np.savez_compressed(save_filename, data=calculated_data)


def test_camera(dataset):

    data = np.load(dataset, allow_pickle=True)['data'].item()

    rand_idx = np.random.randint(0, len(data['train']['pos_3d_world']))

    fig = plt.figure(figsize=(10, 10))
    subplot = fig.add_subplot(111, projection='3d')

    for i in tqdm(range(350)):
        rand_idx = i

        skel_instance = data['train']['pos_3d_world'][rand_idx]/1000

        cam_to_body = data['train']['cam_to_body'][rand_idx]
        cam_pos_body = data['train']['cam_pos_body'][rand_idx]/1000
        body_matrix = data['train']['body_matrix'][rand_idx]
        cam_rot = data['train']['cam_extrinsics_org'][rand_idx][:3, :3]
        cam_pos = data['train']['cam_extrinsics_org'][rand_idx][:3, 3]/1000

        draw_skeleton(skel_instance, subplot)

        new_center = skel_instance[HIP, :] + cam_pos_body
        new_cam = np.matmul(cam_to_body.T, body_matrix.T).T

        plot_vector(skel_instance[HIP, :], body_matrix[:, 1], subplot)
        plot_vector(skel_instance[HIP, :], body_matrix[:, 0], subplot, 'r')
        plot_vector(skel_instance[HIP, :], body_matrix[:, 2], subplot, 'b')

        plot_vector(cam_pos, cam_rot[:, 0], subplot, 'r')
        plot_vector(cam_pos, cam_rot[:, 1], subplot, 'g')
        plot_vector(cam_pos, cam_rot[:, 2], subplot, 'b')

        plot_vector(new_center, new_cam[:, 0], subplot, 'k')
        plot_vector(new_center, new_cam[:, 1], subplot, 'k')
        plot_vector(new_center, new_cam[:, 2], subplot, 'k')

        plt.pause(0.05)
        subplot.clear()

    plt.show()
