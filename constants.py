'''
Constants related to the GPA dataset
'''

HIP = 0
SPINE = 1
SPINE1 = 2
SPINE2 = 3
SPINE3 = 4
NECK = 5
HEAD = 6
NOSE = 7
RSHOULDER = 8
RARM = 9
RFOREARM = 10
RHAND = 11
RHANDEND = 12
LSHOULDER = 16
LARM = 17
LFOREARM = 18
LHAND = 19
LHANDEND = 20
RUPLEG = 24
RLEG = 25
RFOOT = 26
RTOEBASE = 27
LUPLEG = 29
LLEG = 30
LFOOT = 31
LTOEBASE = 32

JOINTS_NAME = [
    'Hips', #0
    'Spine', #1
    'Spine1', #2
    'Spine2', #3
    'Spine3', #4
    'Neck', #5
    'Head', #6
    'Site', #7
    'RightShoulder', #8
    'RightArm', #9
    'RightForeArm', #10
    'RightHand', #11
    'RightHandEnd', #12
    'Site', #13
    'RightHandThumb1', #14
    'Site', #15
    'LeftShoulder', #16
    'LeftArm', #17
    'LeftForeArm', #18
    'LeftHand', #19
    'LeftHandEnd', #20
    'Site', #21
    'LeftHandThumb1', #22
    'Site', #23
    'RightUpLeg', #24
    'RightLeg', #25
    'RightFoot', #26
    'RightToeBase', #27
    'Site', #28
    'LeftUpLeg', #29
    'LeftLeg', #30
    'LeftFoot', #31
    'LeftToeBase', #32
    'Site' #33
]

ADJACENCY_LIST = {
    0: [1, 24, 29], #Hip
    1: [2], #Spine
    2: [3], #Spine1
    3: [4], #Spine2
    4: [5], #Spine3
    5: [7, 8, 16], #Neck
    6: [], #Head
    7: [6], #Nose?
    8: [9], #RightShoulder
    9: [10], #RightArm
    10: [11], #RightForeArm
    11: [12], #RightHand
    12: [14], #RightHandEnd
    14: [], #RightHandThumb1
    16: [17], #LeftShoulder
    17: [18], #LeftArm
    18: [19], #LeftForeArm
    19: [20], #LeftHand
    20: [22], #LeftHandEnd
    22: [], #LeftHandThumb1
    24: [25], #RightUpLeg
    25: [26], #RightLeg
    26: [27], #RightFoot
    27: [], #RightFootToeBase
    29: [30], #RightUpLeg
    30: [31], #RightLeg
    31: [32], #RightFoot
    32: [], #RightFootToeBase
}

PARENTS = [
    -1, 0, 1, 2, 3,
    4, 5, 6, 5, 8,
    9, 10, 11, 12, 12,
    13, 5, 16, 17, 18,
    19, 20, 20, 21, 0,
    24, 25, 26, 27, 0,
    29, 30, 31, 32
]

LEFT_JOINTS = [16, 17, 18, 19, 20, 22, 29, 30, 31, 32]

RIGHT_JOINTS = [8, 9, 10, 11, 12, 14, 24, 25, 26, 27]

DUMMY_LOCATIONS = [

    [0, 0], #Hip
    [0, 0.5], #Spine
    [0, 1], #Spine1
    [0, 1.5], #Spine2
    [0, 2], #Spine3
    [0, 3], #Neck
    [0, 3.8], #Head
    [0, 3.5], #Nose?
    [1, 3], #RShoulder
    [1.2, 2.5], #RArm
    [1.4, 2], #RForeArm
    [1.2, 1.5], #RHand
    [1.2, 1], #RHandEnd
    [1.2, 1.2], #Site
    [1.2, 1.3], #RHThumb1
    [1.2, 1.4], #Site
    [-1, 3], #LShoulder
    [-1.2, 2.5], #LArm
    [-1.4, 2], #LForeArm
    [-1.2, 1.5], #LHand
    [-1.2, 1], #LHandEnd
    [-1.2, 1.2], #Site
    [-1.2, 1.3], #LHThumb1
    [-1.2, 1.4], #Site
    [.7, 0], #RUpLeg
    [.7, -2], #RLeg
    [.7, -4], #RFoot
    [.7, -4.5], #RToeBase
    [.7, -4.6], #Site
    [-.7, 0], #LUpLeg
    [-.7, -2], #LLeg
    [-.7, -4], #LFoot
    [-.7, -4.5], #LToeBase
    [-.7, -4.6], #Site 
]