import rasterio
#from constants import *
#from skimage import io
#from openpiv import tools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# needs to import some kind of database where the former decisions are saved?

def roles(players):
    if players == 3:
        print('There are 3 roles for each player, the Farmer, the Forester and a Business cartel. Please distribute each role among yourselves')
    if players == 4:
        print('There are 4 roles for each player, Farmer 1 and 2, a Forester and a Business cartel. Please distribute each role among yourselves')
    if players == 5:
        print('There are 5 roles for each player, Farmer 1 and 2, Forester 1 and 2 and a Business cartel. Please distribute each role among yourselves')
    if players == 6:
        print('There are 6 roles for each player, Farmer 1 and 2, Forester 1 and 2 and Business representatives 1 and 2. Please distribute each role among yourselves')
    else:
        print(f'The number_of_players is /{players} but should be between 3 and 6')


def game(round=0):
    # ideally just the map itself imported, without the legend and so on, so that we know how many pixel is each field?
    
    # base_map = rasterio.open(f'NFI_map_30_30_{0}.png')
    # map = rasterio.open(f'NFI_map_30_30_{round}.png') #no georef
    map_name = f'NFI_map_30_30_{round}.png'
    pyplot_map = plt.imread(map_name)
    split_name = map_name.split('_')
    resolution = split_name[3]
    plt.plot()
    plt.title(f'Map_{resolution}x{resolution}_{round}')
    plt.imshow(pyplot_map)
    plt.show()
# changed