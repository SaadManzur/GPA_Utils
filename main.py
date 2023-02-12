from __future__ import print_function

import numpy as np
from joint_set import JointSet
import constants as jnt
from conversion import *
from utils import read_npz
from visualize import Visualize
from semgcn.camera import normalize_screen_coordinates
from augmentation import *

if __name__ == "__main__":
    # new_joint_set = JointSet()

    # new_joint_set.build([26, 25, 24, 29, 30, 31, 0, 5, 11, 10, 9, 17, 18,19])

    # new_joint_set.draw_skeleton()

    # print(new_joint_set.get_skeleton())

    # convert_json_to_npz("../gpa_cross_dataset.json", "gpa_cross_dataset_p2")

    # convert_json_to_npz_v2("../xyz_gpa12_cntind_2saad.json", "gpa_xyz", 304892)

    # convert_dataset_and_project_from_3d('gpa_xyz.npz', 'gpa_xyz_projected')

    convert_json_to_npz_world_cam("../xyz_gpa12_cntind_2saad_wolrd_cams.json", "gpa_xyz_raw", 304892)

    # convert_dataset_and_project_from_3d('gpa_xyz_wc.npz', 'gpa_xyz_projected_wc_v3')

    """
    CAMERA = {
        'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focus': [1145.0494384765625, 1143.7811279296875],
        'radial': [
                -0.20709891617298126,
                0.24777518212795258,
                -0.0030751503072679043
            ],
        'tangential': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70,  # Only used for visualization
    }

    EXTRINSIC = {
        'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
        'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125]
    }

    # convert_dataset_and_project_from_3d_using_cam('gpa_xyz_wc.npz', 'dummy')

    visualize = Visualize(10)

    data = read_npz('gpa_with_3dpw_xtrinsic.npz')

    visualize.dump_all_images(
        data['test']['2d_projected'],
        data['size'][0], data['size'][1],
        "out_xtrinsic_3dpw"
    )

    # compute_distance('../gpa_xyz_projected_wc_v3.npz', 'distances')

    distance_data = read_npz('distances.npz')
    print("Mean = ", distance_data['mean'])
    
    for i in range(len(distance_data['distances'])):
        if distance_data['distances'][i]+1000 > distance_data['mean']:
            print(distance_data['distances'][i]) """

    # transform('../gpa_xyz_projected_wc_v3.npz', 'gpa_body_relative')

    # test_camera(dataset='gpa_body_relative.npz')
