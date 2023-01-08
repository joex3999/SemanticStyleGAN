labels = {
0 :"Bird", #Animal
1 :"Ground Animal", #Animal
2 :"Curb", #Fence 5
3 :"Fence", #Fence 5
4 :"Guard Rail", # Wall 4
5 :"Barrier",#Fence 5
6 :"Wall",# Wall 4
7 :"Bike Lane",# Road 1
8 :"Crosswalk - Plain",# Road 1 
9 :"Curb Cut",#Road 1
10 :"Parking",# Road 1 
11 :"Pedestrian Area",# Road 1
12 :"Rail Track",# Road 1
13 :"Road",#Road 1 
14 :"Service Lane",#Road 1
15 :"Sidewalk",#Side walk 2 
16 :"Bridge",#Building 3
17 :"Building",#Building 3 
18 :"Tunnel",#building 3 
19 :"Person",# Person 11
20 :"Bicyclist", # Rider 12
21 :"Motorcyclist",# Rider 12
22 :"Other Rider",# Rider 12
23 :"Lane Marking - Crosswalk",#Road 1
24 :"Lane Marking - General",#Road 1 
25 :"Mountain",# Terrain/Vegitation 9
26 :"Sand",# Terrain/Vegitation 9 
27 :"Sky",#Sky 10
28 :"Snow",#Terrain/vegitation 9 
29 :"Terrain",#Terrain/vegitation 9
30 :"Vegetation",#Terrain/vegitation 9
31 :"Water",#Terrain/vegitation 9 
32 :"Banner",#Pole/PoleGroup 6
33 :"Bench",#side walk 2
34 :"Bike Rack",#Side walk2
35 :"Billboard",#Pole/PoleGroup 6
36 :"Catch Basin",#Sidewalk 2
37 :"CCTV Camera",#Building 3
38 :"Fire Hydrant",#Sidewalk2
39 :"Junction Box",#Building 3
40 :"Mailbox",#Sidewalk 2
41 :"Manhole",#Road 1
42 :"Phone Booth",#Building 3
43 :"Pothole",#Road 1
44 :"Street Light",#Tr.light 7
45 :"Pole",#Pole/PoleGroup 6
46 :"Traffic Sign Frame",# Traffic sign 8
47 :"Utility Pole",#Pole/PoleGroup 6
48 :"Traffic Light",# Tr.Light 7
49 :"Traffic Sign (Back)",#Tr.Sign 8
50 :"Traffic Sign (Front)",#Tr.Sign 8
51 :"Trash Can",#Building 3 
52 :"Bicycle",#Bike 15
53 :"Boat",#Other vehicles 14
54 :"Bus",#Other vehicles 14
55 :"Car",#Cars 13 
56 :"Caravan",#Other vehicles 14
57 :"Motorcycle",#Bike 15
58 :"On Rails",#Other vehicles 14
59 :"Other Vehicle",#Other vehicles 14
60 :"Trailer",#Other vehicles 14
61 :"Truck",#Other vehicles 14
62 :"Wheeled Slow",#Other vehicles 14
63 :"Car Mount",#Void 0 
64 :"Ego Vehicle",# void 0 
65 :"Unlabeled",#Void 0
}
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

cut_down_mapping_v1 = {
0 : 0,
1 : 0,
2 : 5,
3 : 5,
4 : 4,
5 : 5,
6 : 4,
7 : 1,
8 : 1,
9 : 1,
10 : 1,
11 : 1,
12 : 1,
13 :1,
14 : 1,
15 : 2,
16 :3 ,
17 : 3,
18 :3 ,
19 : 11,
20 :12 ,
21 : 12,
22 : 12,
23 :1 ,
24 : 1,
25 : 9,
26 : 9,
27 : 10,
28 : 9,
29 : 9,
30 : 9,
31 : 9,
32 : 6,
33 :2 ,
34 : 2,
35 : 6,
36 : 2,
37 : 3,
38 : 2,
39 : 3,
40 : 2,
41 :1 ,
42 : 3,
43 : 1,
44 : 7,
45 : 6,
46 : 8,
47 : 6,
48 : 7,
49 : 8,
50 : 8,
51 : 3,
52 : 15,
53 : 14,
54 : 14,
55 : 13,
56 : 14,
57 : 15,
58 : 14,
59 : 14,
60 : 14,
61 : 14,
62 : 14,
63 : 0,
64 : 0,
65 : 0,
}
