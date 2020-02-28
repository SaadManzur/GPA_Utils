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
from semgcn.camera import project_to_2d, image_coordinates, normalize_screen_coordinates
from semgcn.utils import wrap

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
                data_2d_train = np.vstack((data_2d_train, joint_2d))
                data_2d_c_train = np.vstack((data_2d_train, joint_2d_c))
                data_3d_train = np.vstack((data_3d_train, joint_3d))
                data_cam_center_train = np.vstack((data_cam_center_train, cam_center))
                data_cam_focal_train = np.vstack((data_cam_focal_train, cam_focal))
                data_bbox_train = np.vstack((data_bbox_train, bbox))
            elif entry['istests']:
                data_2d_test = np.vstack((data_2d_test, joint_2d))
                data_2d_c_test = np.vstack((data_2d_c_test, joint_2d_c))
                data_3d_test = np.vstack((data_3d_test, joint_3d))
                data_cam_center_test = np.vstack((data_cam_center_test, cam_center))
                data_cam_focal_test = np.vstack((data_cam_focal_test, cam_focal))
                data_bbox_test = np.vstack((data_bbox_test, bbox))

        file.close()

    data = {
        "size": [1920, 1080],
        "size_cropped": [256, 256],
        "train": {
            "2d": data_2d_train,
            "2d_c": data_2d_c_train,
            "3d": data_3d_train,
            "center": data_cam_center_train,
            "focus": data_cam_focal_train,
            "bbox": data_bbox_train
        },
        "test": {
            "2d": data_2d_test,
            "2d_c": data_2d_c_test,
            "3d": data_3d_test,
            "center": data_cam_center_test,
            "focus": data_cam_focal_test,
            "bbox": data_bbox_test
        }   
    }

    np.savez_compressed(save_filename, data=data)


def convert_dataset_and_project_from_3d(dataset_file, save_filename):

    data = np.load(dataset_file, allow_pickle=True)['data'].item()

    data_3d_train = data["train"]["3d"]/1000
    data_3d_test = data["test"]["3d"]/1000
    centers_train = data["train"]["center"]
    centers_test = data["test"]["center"]
    focus_train = data["train"]["focus"]
    focus_test = data["test"]["focus"]

    data_2d_train = np.zeros((data_3d_train.shape[0], data_3d_train.shape[1], 2))
    data_2d_test = np.zeros((data_3d_test.shape[0], data_3d_test.shape[1], 2))

    for i in tqdm(range(data_3d_train.shape[0])):
        c = normalize_screen_coordinates(centers_train[i, :], data['size'][0], data['size'][1])
        f = focus_train[i, :] / data['size'][0] * 2.0
        intrinsic = np.concatenate((f, c, np.zeros(3), np.zeros(2)))
        pos_2d = wrap(project_to_2d, True, data_3d_train[i, :], intrinsic)
        pos_2d_pixel_space = image_coordinates(pos_2d, w=data['size'][0], h=data['size'][1])
        data_2d_train[i, :] = pos_2d_pixel_space

    for i in tqdm(range(data_3d_test.shape[0])):
        c = normalize_screen_coordinates(centers_test[i, :], data['size'][0], data['size'][1])
        f = focus_test[i, :] / data['size'][0] * 2.0
        intrinsic = np.concatenate((f, c, np.zeros(3), np.zeros(2)))
        pos_2d = wrap(project_to_2d, True, data_3d_test[i, :], intrinsic)
        pos_2d_pixel_space = image_coordinates(pos_2d, w=data['size'][0], h=data['size'][1])
        data_2d_test[i, :] = pos_2d_pixel_space

    data['train']['2d_projected'] = data_2d_train
    data['test']['2d_projected'] = data_2d_test

    np.savez_compressed(save_filename, data=data)