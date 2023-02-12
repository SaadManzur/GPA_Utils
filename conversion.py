"""
Convert the data to different formats. Structure:
{
    Width
    Height
    Depth [ignore if 2d data]
    Cameras: [
        Camera {
            Id
            Center
            Focus
            Translation
            Rotation
        }
    ]
    Train: {
        3d: [
            [x, y, z]
        ],
        2d: [
            [x, y]
        ],
        2d_c: [
           [x, y] 
        ]
    },
    Test: {
        3d: [
            [x, y, z]
        ],
        2d: [
            [x, y]
        ],
        2d_c: [
            [x, y]
        ]
    } 
}
"""
from __future__ import print_function

import numpy as np
from tqdm import tqdm
import ijson
import cv2 as opencv
from semgcn.camera import project_to_2d, image_coordinates, normalize_screen_coordinates, world_to_camera
from semgcn.utils import wrap
from pyquaternion import Quaternion
from visualize import Visualize

TOTAL_DATAPOINTS = 304892

def read_data(filename, field, shape, data_points=TOTAL_DATAPOINTS):

    data = np.zeros(shape, dtype=np.float32)

    with open(filename, 'r') as file:

        print("Gathering ", field)
        json_data = ijson.items(file, field + ".item", shape)

        print("Compiling ", field)
        for entry in tqdm(json_data, total=data_points):
            joint = np.array(entry, dtype=np.float32).reshape((1, shape[1], shape[2]))
            data = np.vstack((data, joint))

        file.close()

    return data

def convert_json_to_npz(dataset_path, filename):

    data_2d_train = read_data(dataset_path, "gpa_train_2d_origin", (0, 34, 2), 222514)
    data_2d_c_train = read_data(dataset_path, "gpa_train_2d_crop", (0, 34, 2), 222514)
    data_3d_train = read_data(dataset_path, "gpa_train_3d", (0, 34, 3), 222514)

    data_2d_test = read_data(dataset_path, "gpa_test_2d_origin", (0, 34, 2), 82378)
    data_2d_c_test = read_data(dataset_path, "gpa_test_2d_crop", (0, 34, 2), 82378)
    data_3d_test = read_data(dataset_path, "gpa_test_3d", (0, 34, 3), 82378)

    data = {
        "width": 1920,
        "height": 1080,
        "width_cropped": 256,
        "height_cropped": 256,
        "train": {
            "2d": data_2d_train,
            "2d_c": data_2d_c_train,
            "3d": data_3d_train
        },
        "test": {
            "2d": data_2d_test,
            "2d_c": data_2d_c_test,
            "3d": data_3d_test
        }
    }

    np.savez_compressed(filename, data=data)

def convert_json_to_npz_v2(dataset_path, save_filename, total=None):

    """
    data_2d_train = np.zeros((0, 34, 2), dtype=np.float32)
    data_3d_train = np.zeros((0, 34, 3), dtype=np.float32)
    data_2d_test = np.zeros((0, 34, 2), dtype=np.float32)
    data_3d_test = np.zeros((0, 34, 3), dtype=np.float32)
    data_2d_c_train = np.zeros((0, 34, 2), dtype=np.float32)
    data_2d_c_test = np.zeros((0, 34, 2), dtype=np.float32)
    data_cam_center_train = np.zeros((0, 2), dtype=np.float32)
    data_cam_focal_train = np.zeros((0, 2), dtype=np.float32)
    data_cam_center_test = np.zeros((0, 2), dtype=np.float32)
    data_cam_focal_test = np.zeros((0, 2), dtype=np.float32)
    data_bbox_train = np.zeros((0, 4), dtype=np.float32)
    data_bbox_test = np.zeros((0, 4), dtype=np.float32)
    """

    data_2d_train = []
    data_3d_train = []
    data_2d_test = []
    data_3d_test = []
    data_2d_c_train = []
    data_2d_c_test = []
    data_cam_center_train = []
    data_cam_focal_train = []
    data_cam_center_test = []
    data_cam_focal_test = []
    data_bbox_train = []
    data_bbox_test = []
    

    with open(dataset_path, 'r') as file:

        json_data = ijson.items(file, 'annotations.item')

        for entry in tqdm(json_data, total=total):
            joint_2d = np.array(entry['joint_imgs_uncrop'], dtype=np.float32).reshape((1, 34, 2))
            joint_2d_c = np.array(entry['joint_imgs'], dtype=np.float32).reshape((1, 34, 2))
            joint_3d = np.array(entry['joint_cams'], dtype=np.float32).T.reshape((1, 34, 3))
            cam_center = np.array(entry['c'], dtype=np.float32).reshape((1, 2))
            cam_focal = np.array(entry['f'], dtype=np.float32).reshape((1, 2))
            bbox = np.array(entry['bboxes'], dtype=np.float32)

            if entry['istrains']:
                data_2d_train.append(joint_2d)
                data_2d_c_train.append(joint_2d_c)
                data_3d_train.append(joint_3d)
                data_cam_center_train.append(cam_center)
                data_cam_focal_train.append(cam_focal)
                data_bbox_train.append(bbox)
            elif entry['istests']:
                data_2d_test.append(joint_2d)
                data_2d_c_test.append(joint_2d_c)
                data_3d_test.append(joint_3d)
                data_cam_center_test.append(cam_center)
                data_cam_focal_test.append(cam_focal)
                data_bbox_test.append(bbox)

        file.close()

    print(np.vstack(data_2d_train).shape)

    data = {
        "size": [1920, 1080],
        "size_cropped": [256, 256],
        "train": {
            "2d": np.vstack(data_2d_train),
            "2d_c": np.vstack(data_2d_c_train),
            "3d": np.vstack(data_3d_train),
            "center": np.vstack(data_cam_center_train),
            "focus": np.vstack(data_cam_focal_train),
            "bbox": np.vstack(data_bbox_train)
        },
        "test": {
            "2d": np.vstack(data_2d_test),
            "2d_c": np.vstack(data_2d_c_test),
            "3d": np.vstack(data_3d_test),
            "center": np.vstack(data_cam_center_test),
            "focus": np.vstack(data_cam_focal_test),
            "bbox": np.vstack(data_bbox_test)
        }   
    }

    np.savez_compressed(save_filename, data=data)

def convert_json_to_npz_world_cam(dataset_path, save_filename, total=None):
    
    data_3d_wc_train = [] # np.zeros((0, 34, 3), dtype=np.float32)
    data_3d_wc_test = [] # np.zeros((0, 34, 3), dtype=np.float32)
    data_cam_center_train = []
    data_cam_center_test = []
    data_cam_focus_train = []
    data_cam_focus_test = []
    data_cam_rotations_train = [] # np.zeros((0, 3, 1), dtype=np.float32)
    data_cam_rotations_test = [] # np.zeros((0, 3, 1), dtype=np.float32)
    data_cam_translations_train = [] # np.zeros((0, 3, 1), dtype=np.float32)
    data_cam_translations_test = [] # np.zeros((0, 3, 1), dtype=np.float32)
    data_cam_distortions_train = [] # np.zeros((0, 5, 1), dtype=np.float32)
    data_cam_distortions_test = [] # np.zeros((0, 5, 1), dtype=np.float32)

    with open(dataset_path, 'r') as file:
        json_data = ijson.items(file, 'annotations.item')

        for entry in tqdm(json_data, total=total):

            joint_3d_wc = np.array(entry['joint_world_mm'], dtype=np.float32).reshape((1, 34, 3))
            cam_intrinsic = np.array(entry['src_cam0'], dtype=np.float32).reshape((3, 3))
            cam_distortion = np.array(entry['src_cam1'], dtype=np.float32).reshape((1, 5, 1))
            cam_rotation = np.array(entry['src_cam2'], dtype=np.float32).reshape((1, 3, 1))
            cam_translation = np.array(entry['src_cam3'], dtype=np.float32).reshape((1, 3, 1))

            if entry['istrains']:
                data_3d_wc_train.append(joint_3d_wc)
                data_cam_center_train.append([cam_intrinsic[0, 2], cam_intrinsic[1, 2]])
                data_cam_focus_train.append([cam_intrinsic[0, 0], cam_intrinsic[1, 1]])
                data_cam_rotations_train.append(cam_rotation)
                data_cam_distortions_train.append(cam_distortion)
                data_cam_translations_train.append(cam_translation)

            elif entry['istests']:
                data_3d_wc_test.append(joint_3d_wc)
                data_cam_center_test.append([cam_intrinsic[0, 2], cam_intrinsic[1, 2]])
                data_cam_focus_test.append([cam_intrinsic[0, 0], cam_intrinsic[1, 1]])
                data_cam_rotations_test.append(cam_rotation)
                data_cam_distortions_test.append(cam_distortion)
                data_cam_translations_test.append(cam_translation)

        file.close()

        data_3d_wc_train = np.vstack(data_3d_wc_train)
        data_3d_wc_test = np.vstack(data_3d_wc_test)
        data_cam_center_train = np.vstack(data_cam_center_train)
        data_cam_center_test = np.vstack(data_cam_center_test)
        data_cam_focus_train = np.vstack(data_cam_focus_train)
        data_cam_focus_test = np.vstack(data_cam_focus_test)
        data_cam_rotations_train = np.vstack(data_cam_rotations_train)
        data_cam_rotations_test = np.vstack(data_cam_rotations_test)
        data_cam_translations_train = np.vstack(data_cam_translations_train) # np.zeros((0, 3, 1), dtype=np.float32)
        data_cam_translations_test = np.vstack(data_cam_translations_test) # np.zeros((0, 3, 1), dtype=np.float32)
        data_cam_distortions_train = np.vstack(data_cam_distortions_train) # np.zeros((0, 5, 1), dtype=np.float32)
        data_cam_distortions_test = np.vstack(data_cam_distortions_test) # np.zeros((0, 5, 1), dtype=np.float32)

        print(f"WC: {data_3d_wc_train.shape}, {data_3d_wc_test.shape}")
        print(f"Cam Center: {data_cam_center_train.shape}, {data_cam_center_test.shape}")
        print(f"Focus: {data_cam_focus_train.shape}, {data_cam_focus_test.shape}")
        print(f"Rotations: {data_cam_rotations_train.shape}, {data_cam_rotations_test.shape}")
        print(f"Translations: {data_cam_translations_train.shape}, {data_cam_translations_test.shape}")
        print(f"Distortions: {data_cam_distortions_train.shape}, {data_cam_distortions_test.shape}")

        data = {
            "train": {
                "3d": data_3d_wc_train,
                "center": data_cam_center_train,
                "focus": data_cam_focus_train,
                "distortion": data_cam_distortions_train,
                "orientation": data_cam_rotations_train,
                "translation": data_cam_translations_train
            },
            "test": {
                "3d": data_3d_wc_test,
                "center": data_cam_center_test,
                "focus": data_cam_focus_test,
                "distortion": data_cam_distortions_test,
                "orientation": data_cam_rotations_test,
                "translation": data_cam_translations_test
            }   
        }

        np.savez_compressed(save_filename, data=data)


def convert_dataset_and_project_from_3d(dataset_file, save_filename):

    data = np.load(dataset_file, allow_pickle=True)['data'].item()

    data_3d_train = data["train"]["3d"]
    data_3d_test = data["test"]["3d"]
    centers_train = data["train"]["center"]
    centers_test = data["test"]["center"]
    focus_train = data["train"]["focus"]
    focus_test = data["test"]["focus"]
    orientation_train = data["train"]["orientation"]
    orientation_test = data["test"]["orientation"]
    translation_train = data["train"]["translation"]
    translation_test = data["test"]["translation"]
    distortion_train = data["train"]["distortion"]
    distortion_test = data["test"]["distortion"]

    data_2d_train = np.zeros((data_3d_train.shape[0], data_3d_train.shape[1], 2))
    data_2d_test = np.zeros((data_3d_test.shape[0], data_3d_test.shape[1], 2))

    data_3d_train_cam = np.zeros((data_3d_train.shape[0], data_3d_train.shape[1], 3))
    data_3d_test_cam = np.zeros((data_3d_test.shape[0], data_3d_test.shape[1], 3))

    width, height = 1920, 1080

    data['size'] = [1920, 1080]

    for i in tqdm(range(data_3d_train.shape[0])):
        orientation = opencv.Rodrigues(orientation_train[i, :, 0])[0].reshape((3, 3))
        translation = translation_train[i, :, :]/100 #-np.matmul(orientation.T, translation_train[i, :, :]/100)
        
        pos_3d_cam = (np.matmul(orientation.T, data_3d_train[i, :, :].T/1000) + translation).T
        c = normalize_screen_coordinates(np.array(centers_train[i]), w=width, h=height)
        f = np.array(focus_train[i]) / width * 2.0
        intrinsic = np.concatenate((f, c, distortion_train[[0, 1, 4], 0][:, 0], distortion_train[[2, 3], 0][:, 0]))
        pos_2d = wrap(project_to_2d, True, pos_3d_cam[:, :], intrinsic)
        pos_2d_pixel_space = image_coordinates(pos_2d, w=width, h=height)
        data_2d_train[i, :] = pos_2d_pixel_space
        data_3d_train_cam[i, :] = pos_3d_cam

    for i in tqdm(range(data_3d_test.shape[0])):
        orientation = opencv.Rodrigues(orientation_test[i, :, 0])[0].reshape((3, 3))
        translation = translation_test[i, :, :]/100 #-np.matmul(orientation.T, translation_test[i, :, :]/100)

        pos_3d_cam = (np.matmul(orientation.T, data_3d_test[i, :, :].T/1000) + translation).T
        c = normalize_screen_coordinates(np.array(centers_test[i]), width, height)
        f = np.array(focus_test[i]) / width * 2.0
        intrinsic = np.concatenate((f, c, distortion_test[[0, 1, 4], 0][:, 0], distortion_test[[2, 3], 0][:, 0]))
        pos_2d = wrap(project_to_2d, True, pos_3d_cam[:, :], intrinsic)
        pos_2d_pixel_space = image_coordinates(pos_2d, w=width, h=height)
        data_2d_test[i, :] = pos_2d_pixel_space
        data_3d_test_cam[i, :] = pos_3d_cam

    data['train']['2d_projected'] = data_2d_train
    data['test']['2d_projected'] = data_2d_test

    data['train']['3d_world'] = data['train']['3d']
    data['test']['3d_world'] = data['test']['3d']
    data['train']['3d'] = data_3d_train_cam
    data['test']['3d'] = data_3d_test_cam

    np.savez_compressed(save_filename, data=data)


def convert_dataset_and_project_from_3d_using_cam(dataset_file, save_filename, camera_params=None, distance_threshold=None, extrinsic=None):

    data = np.load(dataset_file, allow_pickle=True)['data'].item()

    data_3d_train = data["train"]["3d"]
    data_3d_test = data["test"]["3d"]
    centers_train = data["train"]["center"]
    centers_test = data["test"]["center"]
    focus_train = data["train"]["focus"]
    focus_test = data["test"]["focus"]
    orientation_train = data["train"]["orientation"]
    orientation_test = data["test"]["orientation"]
    translation_train = data["train"]["translation"]
    translation_test = data["test"]["translation"]
    distortion_train = data["train"]["distortion"]
    distortion_test = data["test"]["distortion"]

    if camera_params is not None:
        cam_center = camera_params["center"]
        cam_focus = camera_params["focus"]
        cam_radial = camera_params["radial"]
        cam_tangential = camera_params["tangential"]
    
    data_2d_train = np.empty((0, data_3d_train.shape[1], 2))
    data_2d_test = np.empty((0, data_3d_test.shape[1], 2))

    data_3d_train_cam = np.empty((0, data_3d_train.shape[1], 3))
    data_3d_test_cam = np.empty((0, data_3d_test.shape[1], 3))

    if camera_params is not None:
        width, height = camera_params["res_w"], camera_params["res_h"]
    else:
        width, height = 1920, 1080

    data['size'] = [width, height]

    for i in tqdm(range(data_3d_train.shape[0])):

        if camera_params is None:
            cam_center = centers_train[i]
            cam_focus = focus_train[i]
            cam_radial = distortion_train[[0, 1, 4], 0][:, 0]
            cam_tangential = distortion_train[[2, 3], 0][:, 0]

        if extrinsic is None:
            orientation = opencv.Rodrigues(orientation_train[i, :, 0])[0].reshape((3, 3))
            translation = translation_train[i, :, :]/100 #-np.matmul(orientation.T, translation_train[i, :, :]/100)
            pos_3d_cam = (np.matmul(orientation.T, data_3d_train[i, :, :].T/1000) + translation).T
        else:
            translation = translation_train[i, :, :].astype(np.float32)/100 # np.array(extrinsic['translation'])/1000
            orientation = np.array(extrinsic['orientation'], dtype=np.float32)
            pos_3d_cam = world_to_camera(data_3d_train[i, :, :]/1000, R=orientation, t=translation[:, 0])
        
        c = normalize_screen_coordinates(np.array(cam_center), w=width, h=height)
        f = np.array(cam_focus) / width * 2.0
        intrinsic = np.concatenate((f, c, cam_radial, cam_tangential))
        print(orientation)
        print(translation)
        print(intrinsic)
        
        print(pos_3d_cam)
        pos_2d = wrap(project_to_2d, True, pos_3d_cam[:, :], intrinsic)
        pos_2d_pixel_space = image_coordinates(pos_2d, w=width, h=height)
        print(pos_2d)

        distance_from_cam = np.linalg.norm(translation*1000 - data_3d_train[i, 0, :])

        if distance_threshold is not None and distance_from_cam > distance_threshold:
            continue
        
        if np.any(pos_2d_pixel_space < 0) or np.any(pos_2d_pixel_space > max(width, height)):
            continue
        
        pos_2d_pixel_space = pos_2d_pixel_space.reshape((1, pos_2d_pixel_space.shape[0], pos_2d_pixel_space.shape[1]))
        print(pos_2d_pixel_space)
        pos_3d_cam = pos_3d_cam.reshape((1, pos_3d_cam.shape[0], pos_3d_cam.shape[1]))

        data_2d_train = np.vstack((data_2d_train, pos_2d_pixel_space))
        data_3d_train_cam = np.vstack((data_3d_train_cam, pos_3d_cam))

        break


    for i in tqdm(range(data_3d_test.shape[0])):
        if camera_params is None:
            cam_center = centers_test[i]
            cam_focus = focus_test[i]
            cam_radial = distortion_test[[0, 1, 4], 0][:, 0]
            cam_tangential = distortion_test[[2, 3], 0][:, 0]

        if extrinsic is None:
            orientation = opencv.Rodrigues(orientation_test[i, :, 0])[0].reshape((3, 3))
            translation = translation_test[i, :, :]/100 #-np.matmul(orientation.T, translation_test[i, :, :]/100)

            pos_3d_cam = (np.matmul(orientation.T, data_3d_test[i, :, :].T/1000) + translation).T
        else:
            translation = translation_test[i, :, :].astype(np.float32)/100 # np.array(extrinsic['translation'])/1000
            orientation = np.array(extrinsic['orientation'], dtype=np.float32)
            pos_3d_cam = world_to_camera(data_3d_test[i, :, :]/1000, R=orientation, t=translation[:, 0])

        c = normalize_screen_coordinates(np.array(cam_center), width, height)
        f = np.array(cam_focus) / width * 2.0
        intrinsic = np.concatenate((f, c, cam_radial, cam_tangential))

        pos_2d = wrap(project_to_2d, True, pos_3d_cam[:, :], intrinsic)
        pos_2d_pixel_space = image_coordinates(pos_2d, w=width, h=height)

        distance_from_cam = np.linalg.norm(translation*1000 - data_3d_test[i, 0, :])

        if distance_threshold is not None and distance_from_cam > distance_threshold:
            continue

        if np.any(pos_2d_pixel_space < 0) or np.any(pos_2d_pixel_space > max(width, height)):
            continue

        pos_2d_pixel_space = pos_2d_pixel_space.reshape((1, pos_2d_pixel_space.shape[0], pos_2d_pixel_space.shape[1]))
        pos_3d_cam = pos_3d_cam.reshape((1, pos_3d_cam.shape[0], pos_3d_cam.shape[1]))

        data_2d_test = np.vstack((data_2d_test, pos_2d_pixel_space))
        data_3d_test_cam = np.vstack((data_3d_test_cam, pos_3d_cam))

        break

    data['train']['2d_projected'] = data_2d_train
    data['test']['2d_projected'] = data_2d_test

    data['train']['3d_world'] = data['train']['3d']
    data['test']['3d_world'] = data['test']['3d']
    data['train']['3d'] = data_3d_train_cam
    data['test']['3d'] = data_3d_test_cam

    np.savez_compressed(save_filename, data=data)


def compute_distance(dataset_file, save_filename):

    data = np.load(dataset_file, allow_pickle=True)['data'].item()

    measurements = {
        'mean': 0.0,
        'distances': []
    }

    count = data['train']['3d_world'].shape[0] + data['test']['3d_world'].shape[0]

    for i in range(data['train']['3d_world'].shape[0]):
        distance = np.linalg.norm(data['train']['translation'][i, :, 0]*10 - data['train']['3d_world'][i, 0, :])
        measurements['mean'] += distance
        measurements['distances'].append(distance)

    measurements['mean'] /= count
    np.savez_compressed(save_filename, data=measurements)
