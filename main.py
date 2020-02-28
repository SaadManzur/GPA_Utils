from __future__ import print_function

from joint_set import JointSet
import constants as jnt
from conversion import convert_json_to_npz_v2, convert_dataset_and_project_from_3d
from utils import read_npz

if __name__ == "__main__":
    
    #new_joint_set = JointSet()

    #new_joint_set.build([26, 25, 24, 29, 30, 31, 0, 5, 11, 10, 9, 17, 18,19])

    #new_joint_set.draw_skeleton()
    
    #print(new_joint_set.get_skeleton())

    #convert_json_to_npz("../gpa_cross_dataset.json", "gpa_cross_dataset_p2")

    #convert_json_to_npz_v2("../xyz_gpa12_cntind_2saad.json", "gpa_xyz", 304892)

    convert_dataset_and_project_from_3d('gpa_xyz.npz', 'gpa_xyz_projected')