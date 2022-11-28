"""
Same mapping as V2 City-Scapes.
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

cut_down_mapping = {
    0: 1,
    1: 1,
    2: 1,
    3: 2,
    4: 1,
    5: 2,
    6: 11,
    7: 0,  # ??? ##Animal No Images that contains animals where found whatsover in val and train setting them to unlabeled for now
    8: 12,
    9: 15,
    10: 15,
    11: 13,
    12: 13,
    13: 14,
    14: 14,
    15: 14,
    16: 14,
    17: 14,
    18: 14,
    19: 2,
    20: 4,
    21: 5,
    22: 4,
    23: 8,
    24: 8,
    25: 7,
    26: 6,
    27: 6,
    28: 4,  # ??Could also be building #Obstruction fallback Setting as wall for now
    29: 3,
    30: 3,
    31: 3,
    32: 9,
    33: 10,
    34: 3,  # ??Either sky or vegit or building evenunlabled? #fallback background setting as building for now
    35: 0,
}

labels = {
    "road": 0,
    "parking": 1,
    "drivable_fallback": 2,
    "sidewalk": 3,
    "railtrack": 4,
    "non_drivable_fallback": 5,
    "person": 6,
    "animal": 7,
    "rider": 8,
    "motorcycle": 9,
    "bicycle": 10,
    "autorickshaw": 11,
    "car": 12,
    "truck": 13,
    "bus": 14,
    "caravan": 15,
    "trailer": 16,
    "train": 17,
    "vehiclefallback": 18,
    "curb": 19,
    "wall": 20,
    "fence": 21,
    "guardrail": 22,
    "billboard": 23,
    "trafficsign": 24,
    "trafficlight": 25,
    "pole": 26,
    "polegroup": 27,
    "obstructionfallback": 28,
    "building": 29,
    "bridge": 30,
    "tunnel": 31,
    "vegitation": 32,
    "sky": 33,
    "fallbackbackground": 34,
    "unlabeled": 35,
}
