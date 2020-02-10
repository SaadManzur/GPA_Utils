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

import numpy as np
from tqdm import tqdm
import psutil
import ijson

TOTAL_DATAPOINTS = 304892

def convert_json_to_npz(dataset_path, filename):

    with open(dataset_path, 'r') as json_file:
        json_data = ijson.items(json_file, 'annotations.item')

        data_2d_train = np.zeros((0, 34, 2), dtype=np.float32)
        data_2d_c_train = np.zeros((0, 34, 2), dtype=np.float32)

        data_2d_test = np.zeros((0, 34, 2), dtype=np.float32)
        data_2d_c_test = np.zeros((0, 34, 2), dtype=np.float32)

        data_3d_train = np.zeros((0, 34, 3), dtype=np.float32)
        data_3d_test = np.zeros((0, 34, 3), dtype=np.float32)

        for entry in tqdm(json_data, total=TOTAL_DATAPOINTS):
            joint_3d = np.array(entry['joint_cams'], dtype=np.float32).T.reshape(1, 34, 3)
            joint_2d = np.array(entry['joint_imgs_uncrop'], dtype=np.float32).reshape(1, 34, 2)
            joint_2d_c = np.array(entry['joint_imgs'], dtype=np.float32).reshape(1, 34, 2)
            if entry['is_trains']:
                data_2d_train = np.vstack((data_2d_train, joint_2d))
                data_2d_c_train = np.vstack((data_2d_c_train, joint_2d_c))
                data_3d_train = np.vstack((data_3d_train, joint_3d))
            else:
                data_2d_test = np.vstack((data_2d_test, joint_2d))
                data_2d_c_test = np.vstack((data_2d_c_test, joint_2d_c))
                data_3d_test = np.vstack((data_3d_test, joint_3d))

        data = {
            "width": 1920,
            "height": 1080,
            "width_cropped": 256,
            "height_cropped": 256,
            "train": {
                "2d": data_2d_train,
                "2d_c": data_2d_c_train
                "3d": data_3d_train
            },
            "test": {
                "2d": data_2d_test,
                "2d_c": data_2d_c_test
                "3d": data_3d_test
            }
        }

        np.savez_compressed(filename, data=data)