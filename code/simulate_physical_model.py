from math import exp
from pathos.multiprocessing import ProcessingPool as Pool
from random import uniform
import cv2
import imageio
import multiprocessing
import numpy as np
import os
import random

# configurations for atomic images simulation:
save_directory = '/home/feng/raid_storage/simulated_data/m35_sim_images_1_10/' # path to save images
pixel_interval = 8 # density control
range_of_columns = (75,150) # numbers of atoms
sigmas=(1.0, 10.0) # diversity of bright spots
max_intensity = 4 # peak intensity of atoms
total_threads = 32 # simulation threads
images_to_generate = 1024 # images to generate per thread

# testing configuration
#save_directory = './sim/'
#total_threads = 1
#images_to_generate = 16

def convolution( image, kernel ): # naiive 2D convolution
    assert 2 == len(image.shape)
    assert 2 == len(kernel.shape)
    return cv2.filter2D( src=image, kernel=kernel, ddepth=-1)

def calculate_kernel( sigma=10.0, dimension=(256,256) ): # gaussian kernel
    kernel = np.zeros( dimension )
    row, col = dimension
    for r in range( row ):
        for c in range( col ):
            offset_x = r - (row>>1)
            offset_y = c - (col>>1)
            kernel[r][c] = exp( - (offset_x*offset_x+offset_y*offset_y) / (sigma+sigma) )
    return kernel

def generate_random_coordinates( dimension=(512,512), columns=50, pixel_interval=32, max_intensity=5 ):
    intensities = np.random.random_integers( 1, max_intensity, columns )

    row, col = dimension
    r_, c_ = int(row/pixel_interval), int(col/pixel_interval)

    compressed_coords = np.random.random_integers( 0, r_*c_-1, r_*c_ )
    compressed_2d_coords = np.zeros( (r_, c_ ) )
    proceed = 0
    for idx in range( r_*c_ ):
        if proceed >= columns: # done
            break
        r, c = int( compressed_coords[idx]/r_), compressed_coords[idx]%r_
        if compressed_2d_coords[r][c] < 0.5:
            compressed_2d_coords[r][c] = intensities[proceed]
            proceed += 1

    random_offset = np.random.rand( columns, 2 ) * 0.25 - 0.125

    random_coords = np.zeros( (row, col) )
    proceed = 0
    for r in range(r_):
        for c in range(c_):
            if compressed_2d_coords[r][c] > 0.5:
                random_coords[int((r+random_offset[proceed][0])*pixel_interval)][int((c+random_offset[proceed][1])*pixel_interval)] = compressed_2d_coords[r][c]
                proceed += 1
    return random_coords

def remove_boundary( image, boundary=16 ):
    row, col = image.shape
    ans = np.zeros( (row, col) )
    ans[boundary:row-boundary, boundary:col-boundary] = image[boundary:row-boundary, boundary:col-boundary]
    return ans

def fake_image( dimension=(512, 512), range_of_columns=(20, 50), kernel=None, sigma=80.0, max_intensity = 5, offset=32, pixel_interval=32, save_path=None ):
    columns = int( random.uniform(*range_of_columns) )
    row, col = dimension
    row_coords = np.random.random_integers( offset, row-offset, columns )
    col_coords = np.random.random_integers( offset, col-offset, columns )
    intensities = np.random.random_integers( 1, max_intensity, columns )

    image = np.zeros( dimension )
    for idx in range(columns):
        image[row_coords[idx]][col_coords[idx]] = intensities[idx]

    if kernel is None:
        kernel = calculate_kernel( sigma=sigma, dimension=(row>>1, col>>1) )

    image = generate_random_coordinates( dimension=dimension, columns=columns, pixel_interval=pixel_interval, max_intensity=max_intensity )
    image = remove_boundary( image )
    image = convolution( image=image, kernel=kernel )

    if save_path is not None:
        imageio.imsave( save_path, image )

    return image

def fake_images( images_to_generate, dimension=(512, 512), range_of_columns=(30, 70), sigmas=(9,100), max_intensity=5, pixel_interval=32 ):
    row, col = dimension
    random_sigma = np.random.random_integers( *sigmas, images_to_generate )
    images = np.zeros( (images_to_generate, row, col) )
    for idx in range( images_to_generate ):
        images[idx,:,:] = fake_image( dimension=dimension, range_of_columns=range_of_columns, kernel=None, sigma=random_sigma[idx], max_intensity=max_intensity, offset=32, pixel_interval=pixel_interval, save_path=None )

    return images

def normalize( image ) :
    return (image-np.amin(image))/(np.amax(image)-np.amin(image)+1.0e-10)#scale to [0, 1]

def make_simulation(index):
    global save_directory
    global images_to_generate
    global pixel_interval
    global range_of_columns
    global sigmas
    global max_intensity

    directory = f'{save_directory}/{index}/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    simulated_images = fake_images( images_to_generate, dimension=(256, 256), range_of_columns=range_of_columns, sigmas=sigmas, max_intensity=max_intensity, pixel_interval=pixel_interval )
    simulated_images = np.asarray( normalize( simulated_images ) * 65535.0, dtype='uint16')
    for idx in range( images_to_generate ):
        file_name = f'{directory}/{str(idx).zfill(4)}.png'
        imageio.imsave( file_name, simulated_images[idx, 64:192, 64:192] )

if __name__ == '__main__':
    indices = [ i for i in range( total_threads ) ]
    with Pool(multiprocessing.cpu_count()) as p:
        p.map( make_simulation, indices )

