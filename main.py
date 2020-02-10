from joint_set import JointSet
import constants as jnt

if __name__ == "__main__":
    new_joint_set = JointSet()

    new_joint_set.build(
        [
            jnt.HIP,
            jnt.RUPLEG,
            jnt.RLEG,
            jnt.RFOOT,
            jnt.LUPLEG,
            jnt.LLEG,
            jnt.LFOOT,
            jnt.SPINE1,
            jnt.NECK,
            jnt.NOSE,
            jnt.HEAD,
            jnt.LSHOULDER,
            jnt.LFOREARM,
            jnt.LHANDEND,
            jnt.RSHOULDER,
            jnt.RFOREARM,
            jnt.RHANDEND
        ]
    )

    new_joint_set.draw_skeleton()