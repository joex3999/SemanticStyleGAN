
'''
0: unlabled out of roi
1: Flat : road sidewalk parking rail track
2: human : person rider
3: vehicle : car truck bus on rails motorcycle bicycle caravan trailer
4: construction : building wall fence guard rail bridge tunnel
5: object : pole pole group rtaffic sign traffic light
6: nature : vegetation terrain
7: sky : sky
8:void : ground dynamic static
'''
cut_down_mapping_v1 = {
    0: 0,
    1: 0,
    2: 5,
    3: 0,
    4: 8,
    5: 8,
    6: 8,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 4,
    12: 4,
    13: 4,
    14: 4,
    15: 4,
    16: 4,
    17: 5,
    18: 5,
    19: 5,
    20: 5,
    21: 6,
    22: 6,
    23: 7,
    24: 2,
    25: 2,
    26: 3,
    27: 3,
    28: 3,
    29: 3,
    30: 3,
    31: 3,
    32: 3,
    33: 3,
}
labels = {
    'unlabeled':  0 ,
    'ego vehicle':  1 ,
    'rectification border':  2 ,
    'out of roi' :  3 ,
    'static' :  4 ,
    'dynamic' :  5 ,
    'ground'    :  6 ,
    'road'      :  7 ,
    'sidewalk'  :  8 ,
    'parking'   :  9 ,
    'rail track': 10 ,
    'building'  : 11 ,
    'wall'      : 12 ,
    'fence'     : 13 ,
    'guard rail': 14 ,
    'bridge'    : 15 ,
    'tunnel'    : 16 ,
    'pole'      : 17 ,
    'polegroup' : 18 ,
    'traffic light': 19 ,
    'traffic sign'  : 20 ,
    'vegetation'    : 21 ,
    'terrain'       : 22 ,
    'sky'           : 23 ,
    'person'        : 24 ,
    'rider'         : 25 ,
    'car'           : 26 ,
    'truck'         : 27 ,
    'bus'           : 28 ,
    'caravan'       : 29 ,
    'trailer'       : 30 ,
    'train'         : 31 ,
    'motorcycle'    : 32 ,
    'bicycle'       : 33 ,
    'license plate' : -1 
}


