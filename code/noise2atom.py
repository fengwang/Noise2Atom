import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np
import tifffile
import copy
import imageio
import random
from message import send_message
from message import send_photo
from keras_utils import read_model
from keras_utils import write_model
from simulate_physical_model import simulate_atomic_images, normalize
from models import build_wgan, batch_size

def scale( array ):
    return 2.0 * array - 1.0

def combine( array_a, array_b ):
    a, b = normalize(array_a), normalize(array_b)
    row, col = a.shape
    ans = np.zeros( (row, col*2) )
    ans[:,:col] = a
    ans[:,col:] = b
    ans = np.asarray( ans*255.0, dtype='uint8' )
    return ans

# random sample a batch of training examples
def get_training_data_inputs( n_batch_size, simulated_images, noisy_images ):
    n_simulated_images, sr, sc, _ = simulated_images.shape
    random_index = random.randint(0, n_simulated_images-n_batch_size)
    random_simulated = simulated_images[random_index:random_index+n_batch_size]
    print( f'random_simulated:{random_simulated.shape}' )

    n_noisy_images, r, c, _ = noisy_images.shape
    random_index = random.randint(0, n_noisy_images-n_batch_size)
    random_row = random.randint(0, r-sr)
    random_col = random.randint(0, c-sc)
    random_noisy = noisy_images[random_index:random_index+batch_size, random_row:random_row+sr, random_col:random_col+sc]
    print( f'random_noisy:{random_noisy.shape}' )

    return [random_simulated, random_noisy]


training_identity = 1
simulated_tif_path = './simulated_images.tif'
noisy_tif_path = '/home/feng/Downloads/Trond denoise/re6-s10-crop800.tif'
noisy_to_clean_model_directory = './model_noisy2clean'
n_simulated_images = 1024
n_loops = 1024
n_sampling_interval = 64
n_critcs = 5



send_message( f'noisy2atom training {training_identity=} started' )

# loading/simulating clear atomic images
if not os.path.isfile( simulated_tif_path ):
    print( 'Simulated image file is not found, trying to simulate in place.' )
    simulated_images = simulate_atomic_images( n_simulated_images, tiff_path=simulated_tif_path, resolution=(128, 128) )
else:
    simulated_images = np.asarray( tifffile.imread( simulated_tif_path ), dtype='float32' )
    n, *_ = simulated_images.shape
    assert n >= n_simulated_images, f'Expecting more than {n_simulated_images} atomic images, but only get {n}'
    simulated_images = normalize( simulated_images[:n_simulated_images] )
simulated_images = scale( simulated_images.reshape( simulated_images.shape + (1,) ) ) # of shape [ n_simulated_images, 128, 128, 1]
send_message( f'noisy2atom training {training_identity=} clear images generated/loaded' )

# load noisy experimental data
if not os.path.isfile( noisy_tif_path ):
    assert False, f'Failed to load STEM image from {noisy_tif_path=} with {training_identity=}'
else:
    noisy_images = normalize( tifffile.imread( noisy_tif_path ) )
noisy_images = scale( noisy_images.reshape( noisy_images.shape + (1,) ) )
send_message( f'noisy2atom training {training_identity=} noisy images generated/loaded' )


# generating model
critic_model, noisy_to_clean_generator, cycle_training_model, noisy_to_clean_model = build_wgan()
send_message( f'noisy2atom training {training_identity=} model generated' )


zero_output = np.zeros( (batch_size, 128, 128, 1) )
valid = -np.ones((batch_size, 1), dtype='float32')
fake =  np.ones((batch_size, 1), dtype='float32')
dummy = np.zeros((batch_size, 1), dtype='float32')

for loop in range( n_loops ):
    last_loss = None
    print( f'Loop {loop+1} started' )
    for _ in range( n_sampling_interval ): # every half an hour for a single 1080 Ti
        noisy, clear = get_training_data_inputs(batch_size, simulated_images, noisy_images)
        for nc in range( n_critcs ):
            critic_model.train_on_batch([clear, noisy], [valid, fake, dummy])
        noisy_to_clean_generator.train_on_batch(noisy, valid)
        last_loss = cycle_training_model.train_on_batch( [noisy, clear], [zero_output, zero_output] )

    write_model( f'{noisy_to_clean_model_directory}_{loop+1}', noisy_to_clean_model )

    denoised_experimental = noisy_to_clean_model.predict( noisy_images[:batch_size] )
    denoised_experimental = (denoised_experimental + 1.0) / 2.0
    denoised_experimental = np.asarray( denoised_experimental * ( (1 << 16) -1 ), dtype='uint16' )

    tifffile.imsave( f'./denoised_image-{training_identity}_{loop+1}.tif', denoised_experimental, compress=6 )
    send_message( f'cycle denoising of {training_identity}: {loop+1}/{n_loops} done, cycle loss [noisy, clear] is {last_loss=}.' )

    _a, _b = copy.deepcopy( noisy_images[:batch_size] ), denoised_experimental[:batch_size]
    for idx in range(batch_size):
        imageio.imsave( f'./tmp_denoised_{idx}.png', combine( np.squeeze(_a[idx]), np.squeeze(_b[idx]) ) )
        send_photo( f'./tmp_denoised_{idx}.png' )

write_model( f'./noisy_to_clean_model_{training_identity}', noisy_to_clean_model )




