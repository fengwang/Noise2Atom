import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tqdm
import copy

training_index = 1

from message import send_message
from message import send_photo
send_message( f'cycle training {training_index} started' )

from models import build_wgan
critic_model, noisy_to_clean_generator, cycle_training_model, noisy_to_clean_model = build_wgan()

cycle_training_model_directory = f'/raid/feng/cache/train_wasserstein_28_denoising/model_cycle_training_model_{training_index}'
noisy_to_clean_model_directory = f'/raid/feng/cache/train_wasserstein_28_denoising/model_noisy_to_clean_model_{training_index}'

import glob
simulated_image_paths = glob.glob( '/home/feng/raid_storage/simulated_data/simulated_stem_images_128x128_16bits/*/*.png' ) # gray, 128x128
n_simulated_images = len( simulated_image_paths )
print( f'simulated image dataset has {n_simulated_images} entries' )

from skimage import io
import numpy as np

noisy_images_ = io.imread( '/raid/feng/experimental_data/Data sets for NN denoising-Debora/original/too_large/2019_03_26-15.35abcdf,2048px,8.8pm.tif.too_large.tif' ) #
send_message( f'noisy image of shape {noisy_images_.shape} loaded' )
n, row, col = noisy_images_.shape
noisy_images = np.zeros( (n*256, 128, 128), dtype=noisy_images_.dtype )
for r in range( 16 ):
    for c in range( 16 ):
        offset = r * 16 + c
        noisy_images[offset*n:(1+offset)*n] = noisy_images_[:,r*128:(r+1)*128, c*128:(c+1)*128]
print( 'noisy image converted' )

io.imsave( f'/raid/feng/cache/train_wasserstein_28_denoising/noisy_experimental_{training_index}.tif', noisy_images )
n_noisy_images, *_ = noisy_images.shape
print( f'{n_noisy_images} noisy images loaded' )

import numpy as np
noisy_images = np.asarray( noisy_images, dtype='float32' ) / (np.amax( noisy_images ) + 1.0e-10)
noisy_images = noisy_images.reshape( noisy_images.shape + (1,) )

batch_size = 6 # training batch
simulated_image_input = np.zeros( (batch_size, 128, 128, 1 ) )

noisy_counter = 0
simulated_counter = 0

import imageio

def scaler( array, mn=-1.0, mx=1.0 ):
    return np.asarray( (mx-mn) * (array-np.amin(array))/(np.amax(array)-np.amin(array)+1.0e-10) + mn, dtype='float32' )

def scale_simulated( array ): # uint16
    if array.dtype == np.dtype('uint16'):
        return np.asarray( 2.0 * array / 65535.0 - 1.0, dtype='float32' )
    if array.dtype == np.dtype('uint8'):
        return np.asarray( 2.0 * array / 255.0 - 1.0, dtype='float32' )
    return 2.0*(array-np.amin(array))/(np.amax(array)-np.amin(array)+1.0e-10) - 1.0

def scale_noisy( array ): # has been rescaled to [0, 1]
    return 2.0 * array - 1.0

def get_training_data_inputs( n_batch_size ):
    global noisy_counter
    global noisy_images
    global simulated_counter
    global simulated_image_input
    global simulated_image_paths

    if noisy_counter + n_batch_size >= n_noisy_images:
        noisy_counter = 0

    if simulated_counter + n_batch_size > n_simulated_images:
        simulated_counter = 0

    for idx in range( n_batch_size ):
        simulated_image_input[idx] = np.asarray( imageio.imread( simulated_image_paths[idx+simulated_counter] ) ).reshape( (128, 128, 1) )

    noisy_counter += n_batch_size
    simulated_counter += n_batch_size

    return [scale_noisy(noisy_images[noisy_counter-n_batch_size:noisy_counter]), scale_simulated(simulated_image_input)]

zero_output = np.zeros( (batch_size, 128, 128, 1) )
valid = -np.ones((batch_size, 1))
fake =  np.ones((batch_size, 1))
dummy = np.zeros((batch_size, 1))

noisy_to_clean_generator_multi_gpu_model = noisy_to_clean_generator
cycle_training_multi_gpu_model = cycle_training_model

send_message( 'data/model loaded' )

n_loops = 192*32*32
n_iterations = 64
n_critcs = 5

import tqdm
from keras_utils import read_model
from keras_utils import write_model
import tifffile


def normalize( array ):
    return ( array - np.amin(array) ) / ( np.amax(array) - np.amin(array) + 1.0e-10)

def combine( array_a, array_b ):
    a, b = normalize(array_a), normalize(array_b)
    row, col = a.shape
    ans = np.zeros( (row, col*2) )
    ans[:,:col] = a
    ans[:,col:] = b
    ans = np.asarray( ans*255.0, dtype='uint8' )
    return ans

for loop in range( n_loops ): # half hour for each

    last_loss = None
    print( f'Loop {loop+1} started' )
    for _ in tqdm.tqdm( range( n_iterations ) ):
        noisy, clear = get_training_data_inputs(batch_size)
        for nc in range( n_critcs ):
            critic_model.train_on_batch([clear, noisy], [valid, fake, dummy])
        noisy_to_clean_generator.train_on_batch(noisy, valid)
        last_loss = cycle_training_multi_gpu_model.train_on_batch( [noisy, clear], [zero_output, zero_output] )

    write_model( f'{cycle_training_model_directory}_{loop+1}', cycle_training_model )
    write_model( f'{noisy_to_clean_model_directory}_{loop+1}', noisy_to_clean_model )

    noisy_to_clean_multi_gpu_model = noisy_to_clean_model
    denoised_experimental = noisy_to_clean_multi_gpu_model.predict( noisy_images[:batch_size] )
    denoised_experimental = (denoised_experimental + 1.0) / 2.0
    denoised_experimental = np.asarray( denoised_experimental * ( (1 << 16) -1 ), dtype='uint16' )

    tifffile.imsave( f'./denoised_image-{training_index}_{loop+1}.tif', denoised_experimental, compress=6 )
    send_message( f'cycle denoising of {training_index}: {loop+1}/{n_loops} done, last loss [noisy, clear] is {last_loss}.' )

    _a, _b = copy.deepcopy( noisy_images[:4] ), denoised_experimental[:4]
    for idx in range(4):
        imageio.imsave( './tmp_denoised.png', combine( np.squeeze(_a[idx]), np.squeeze(_b[idx]) ) )
        send_photo( './tmp_denoised.png' )

import inspect
import os
file_name = inspect.getfile(inspect.currentframe())
file_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

send_message( f'training finished with file {file_directory}{file_name} of {n_training_loops} loops with batch size {batch_size}' )
send_message( f'cycle model has been saved to {cycle_training_model_directory}' )
send_message( f'noisy to clean model has been saved to {noisy_to_clean_model_directory}' )

