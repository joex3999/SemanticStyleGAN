## TODO: Change this entire file.
"""
0: unlabled out of roi
1: Flat : road sidewalk parking rail track
2: human : person rider
3: vehicle : car truck bus on rails motorcycle bicycle caravan trailer
4: construction : building wall fence guard rail bridge tunnel
5: object : pole pole group rtaffic sign traffic light
6: nature : vegetation terrain
7: sky : sky
8:void : ground dynamic static
"""
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


"""
Version 2 
0: void: unlabeled/ego vehicle/ rectif. border/ out of roi /static/ dynamic/ ground
1: Road :  roadq/parking/rail track
2: S.Walk: Swalk
3: Build:building/bridge/tunnel
4: Wall: wall/ guard rail
5: Fence: fence
6: Pole: pole / pole group
7: Tr.Light: traffic light
8: Sign:  rtraffic sign
9: Veget.: Vegetation terrain
10: Sky: Sky 
11: Person:Person
12:Rider: rider
13: Car:car
14: Other vehicles: caravan/ trailer/ bus/ truck / train
15: M.Bike/Bike: motorcycle/ bicycle
"""

cut_down_mapping_v2 = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
    8: 2,
    9: 1,
    10: 1,
    11: 3,
    12: 4,
    13: 5,
    14: 4,
    15: 3,
    16: 3,
    17: 6,
    18: 6,
    19: 7,
    20: 8,
    21: 9,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 14,
    29: 14,
    30: 14,
    31: 14,
    32: 15,
    33: 15,
}

"""
Version 3
0: void: unlabeled/ego vehicle/ rectif. border/ out of roi /static/ dynamic/ ground
1: Road :  roadq/parking/rail track
2: S.Walk: Swalk
3: Build:building
4: Wall: wall
5: Fence: fence
6: Pole: pole
7: Tr.Light: traffic light
8: Sign:  rtraffic sign
9: Veget.: Vegetation
10: Sky: Sky 
11: Person:Person
12:Rider: rider
13: Car:car
14: Caravan & trailer: caravan/ trailer
15: M.Bike/Bike: motorcycle/ bicycle
16:Pole Group: Pole Group
17:terrain: Terrain
18:Bridge: bridge
19:Tunnel:tunnel
20:Guard rail: Guard Rail.
21:Train:Train
22:Bus & truck : bus/ truck
23: Guard rail : Guard rail
"""
cut_down_mapping_v3_24C = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
    8: 2,
    9: 1,
    10: 1,
    11: 3,
    12: 4,
    13: 5,
    14: 23,
    15: 18,
    16: 19,
    17: 6,
    18: 16,
    19: 7,
    20: 8,
    21: 9,
    22: 17,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 22,
    28: 22,
    29: 14,
    30: 14,
    31: 21,
    32: 15,
    33: 15,
}


"""
Version 4 (12 classes )
0: void: unlabeled/ego vehicle/ rectif. border/ out of roi /static/ dynamic/ ground
1: Road :  roadq/parking/rail track/swalk
2: Build:building/bridge/tunnel
3: Wall: wall/ guard rail /fence 
4: Pole: pole / pole group
5: Tr.Light: traffic light / traffic sign
6: Veget.: Vegetation terrain
7: Sky: Sky 
8: Person/Rider:Person/Rider
9: Car:car
10: Other vehicles: caravan/ trailer/ bus/ truck / train
11: M.Bike/Bike: motorcycle/ bicycle
"""
cut_down_mapping_v4 = {
    0: 0,  # unlabeled
    1: 0,  # ego v
    2: 0,  # recti border
    3: 0,  # out of roi
    4: 0,  # static
    5: 0,  # dynamic
    6: 0,  # ground
    7: 1,  # road
    8: 1,  # swalk
    9: 1,  # parking
    10: 1,  # railtrack
    11: 2,  # building
    12: 3,  # wall
    13: 3,  # fence
    14: 3,  # guard rail
    15: 2,  # Bridge
    16: 2,  # tunnel
    17: 4,  # Pole
    18: 4,  # polegroup
    19: 5,  # Traffic light
    20: 5,  # traffic sign
    21: 6,  # Vegi
    22: 6,  # Terrain
    23: 7,  # Sky
    24: 8,  # Person
    25: 8,  # Rider
    26: 9,  # Car
    27: 10,  # Truck
    28: 10,  # Bus
    29: 10,  # Caravan
    30: 10,  # Trailer
    31: 10,  # Train
    32: 11,  # Motorcycle
    33: 11,  # Bicycle
}

labels = {
    "unlabeled": 0,
    "ego vehicle": 1,
    "rectification border": 2,
    "out of roi": 3,
    "static": 4,
    "dynamic": 5,
    "ground": 6,
    "road": 7,
    "sidewalk": 8,
    "parking": 9,
    "rail track": 10,
    "building": 11,
    "wall": 12,
    "fence": 13,
    "guard rail": 14,
    "bridge": 15,
    "tunnel": 16,
    "pole": 17,
    "polegroup": 18,
    "traffic light": 19,
    "traffic sign": 20,
    "vegetation": 21,
    "terrain": 22,
    "sky": 23,
    "person": 24,
    "rider": 25,
    "car": 26,
    "truck": 27,
    "bus": 28,
    "caravan": 29,
    "trailer": 30,
    "train": 31,
    "motorcycle": 32,
    "bicycle": 33,
    "license plate": -1,
}
