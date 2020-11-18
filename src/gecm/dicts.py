# --------------------------------------------------
# Dictionaries describing the Category - ID Mapping
# --------------------------------------------------
# Source: NFI woodland map 2018
# 2 Categories: Woodland & Non Woodland
# Woodland categories run from 1-16
# Non Woodland categories run from 1-11,
# but the number 2 precludes them to
# differentiate them from woodland categories
# --------------------------------------------------


# TODO [low]: copy this dict into a new excel file to be parsed when needed
#  --> potentially also include long category descriptors
original_lulc_mapping = {
    "Conifer": 1,  # start of woodland categories
    "Broadleaved": 2,
    "Mixed mainly conifer": 3,
    "Mixed mainly broadleaved": 4,
    "Coppice with standards": 6,
    "Shrub": 7,
    "Young trees": 8,
    "Felled": 9,
    "Ground prep": 10,
    "Low density": 13,
    "Assumed woodland": 14,
    "Failed": 15,
    "Windblow": 16,
    "Open water": 21,  # start of non-woodland categories
    "Grassland": 22,
    "Agriculture land": 23,
    "Urban": 24,
    "Road": 25,
    "River": 26,
    "Quarry": 28,
    "Bare area": 29,
    "Windfarm": 210,
    "Other vegetation": 211,
    "Grazing": 240,  # defined by team
}

# describes the simplified LULC types
simplified_lulc_mapping = {
    "Sheep Farming": 1,
    "Native Forest": 2,
    "Commercial Forest": 3,
    "Cattle Farming": 4,
}

# describes the relationship between the simplified & original LULC identifiers
lulc_remapping = {
    1: [7, 13, 22, 23, 211, 240],
    2: [2, 4, 6, 9, 21, 24, 25, 26, 28, 29, 210],
    3: [1, 3, 8, 10, 14, 15, 16],
}

# colormap source: https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=5
simplified_lulc_mapping_colors = {
    1: "#fee090",
    2: "#33a02c",
    3: "#b2df8a",
    #4: "#fdbf6f",
}
