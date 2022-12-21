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

cut_down_mapping_v1_level1 = {
    0: 1,
    1: 1,
    2: 1,
    3: 2,
    4: 1,
    5: 2,
    6: 11,
    7: 0,  # ??? ##Animals set to unlabeled
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
##17 Local generators instead of 16, just split rickshaw in it's own seperate entity.
##This for lmdb v5
cut_down_mapping_v5_level1_split_rickshaw = {
    0: 1,
    1: 1,
    2: 1,
    3: 2,
    4: 1,
    5: 2,
    6: 11,
    7: 0,  # ??? ##Animals set to unlabeled
    8: 12,
    9: 15,
    10: 15,
    11: 16,  # Rickshaw is Set to label 16.
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

"""
Same mapping as V2 City-Scapes.
0: void: unlabeled/ego vehicle/ rectif. border/ out of roi /static/ dynamic/ ground
1: Road :  roadq/parking/rail track
2: S.Walk: Swalk
3: Build:building/bridge/tunnel
4: Wall: wall/ guard rail / curv
5: Fence: fence
6: Pole: pole / pole group//Builboard
7: Tr.Light: traffic light
8: Sign:  rtraffic sign
9: Veget.: Vegetation terrain
10: Sky: Sky 
11: Person:Person
12:Rider: rider
13: Car:car
14: Other vehicles: caravan/ trailer/ bus/ truck / train/ Auto ricksaw
15: M.Bike/Bike: motorcycle/ bicycle
"""
# This is an abstraction above level3IDs
cut_down_mapping_v2_level3 = {
    0: 1,
    1: 1,
    2: 2,  # Side walk// railtrack//non drivable fallback
    3: 2,
    4: 11,  # Person and animal
    5: 12,  # Rider
    6: 15,  # Motorcycle
    7: 15,  # Bicycle
    8: 14,
    9: 13,
    10: 14,
    11: 14,
    12: 14,
    13: 4,  # Curb
    14: 4,  # Wall
    15: 4,  # Fence
    16: 4,  # Guard rail
    17: 6,  # billboard
    18: 8,
    19: 7,
    20: 6,
    21: 4,  # Obst fall back
    22: 3,  # Building
    23: 3,  # Bridge and tunnel
    24: 9,
    25: 10,  # Sky
    26: 10,  # Object fall back
    255: 0,  # All else void.
}


"""
Completely new Mapping/ Advatnage would be splitting autorickshaw in it's own LG, and 3 lgs less
0: void: unlabeled/ego vehicle/ rectif. border/ out of roi /static/ dynamic/ ground
1: Drivable :
2: Non-Drivable:
3: Living-things:
4: 2-wheeler:
5: autorickshaw:
6: Car:
7: Large-Vehicle:
8: Barrier:
9: Structures:
10: Construction:
11: Vegetation:
12: Sky:
"""
# This is an abstraction above level3IDs
cut_down_mapping_v3_level3 = {
    0: 1,
    1: 1,
    2: 2,  # Side walk// railtrack//non drivable fallback
    3: 2,
    4: 3,  # Person and animal
    5: 3,  # Rider
    6: 4,  # Motorcycle
    7: 4,  # Bicycle
    8: 5,
    9: 6,
    10: 7,
    11: 7,
    12: 7,
    13: 8,  # Curb
    14: 8,  # Wall
    15: 8,  # Fence
    16: 8,  # Guard rail
    17: 9,  # billboard
    18: 9,
    19: 9,
    20: 9,
    21: 9,  # Obst fall back
    22: 10,  # Building
    23: 10,  # Bridge and tunnel
    24: 11,
    25: 12,  # Sky Object fall back ##PS: Object fallback is an LG on it's own in the original split.
    255: 0,  # All else void.
}
"""
Completely new Mapping/ Advantage would be:
                Other than splitting some classes on it's own (like the previous rickshaw class.),
                We would also increase the number of local generators to see if the overall result
                is improved, We are increasing the LGs to 21 LG.
0: void: unlabeled/ego vehicle/ rectif. border/ out of roi /static/ dynamic/ ground
1: Drivable : Road / Parking / Drivable fallback
2: Non-Drivable:SideWalk/RailTrack/ND Fallback
3: Person/Rider
4: Animal
5: autorickshaw:
6: Car:
7: Motorcycle/Bicycle
8: Truck
9:Bus
10:Other-Vehicles:Caravan/Trailer/Train/ Vehicle fallback
11: Curb
12:Wall
13:Fence//Guardrail
14: TrafficSign//Bilboard
15: Traffic Light
16: Pole/PoleGroup/obst-str-bar-fallback
17:Building
18:Bridge/Tunnel
19:Vegitation
20:sky/fallbackground
"""

##Mapped to lmdb v4
cut_down_mapping_v4_level1 = {
    0: 1,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 3,
    7: 4,  # ??? ##Animals set to unlabeled
    8: 3,
    9: 7,
    10: 7,
    11: 5,  ##Autorickshaw
    12: 6,
    13: 8,
    14: 9,
    15: 10,
    16: 10,
    17: 10,
    18: 10,
    19: 11,
    20: 12,
    21: 13,
    22: 13,
    23: 14,
    24: 14,
    25: 15,
    26: 16,
    27: 16,
    28: 16,  # ??Could also be building #Obstruction fallback Setting as wall for now
    29: 17,
    30: 18,
    31: 18,
    32: 19,
    33: 20,
    34: 20,  # ??Either sky or vegit or building evenunlabled? #fallback background setting as building for now
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
