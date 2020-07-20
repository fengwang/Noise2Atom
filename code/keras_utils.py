from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

import os
import glob
import numpy as np
import imageio

log_flag = True

def log( message ):
    if log_flag:
        print( message )

'''
    Example:
        model = ...
        write_model('./cached_folder', model)
'''
def write_model(directory, model):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # saving weights
    weights_path = f'{directory}/weights.h5'
    if os.path.isfile(weights_path):
        os.remove(weights_path)
    model.save_weights(weights_path)

    # saving json
    json_path = f'{directory}/js.json'
    if os.path.isfile(json_path):
        os.remove(json_path)
    with open( json_path, 'w' ) as js:
        js.write( model.to_json() )

def write_model_checkpoint(directory, model):
    model_path = f'{directory}/model.h5'
    if os.path.isfile(model_path):
        os.remove(model_path)

    model.save( model_path )
    write_model(directory, model)

'''
    Example:
        model = read_model( './cached_folder' )
'''
def read_model(directory):
    weights_path = f'{directory}/weights.h5'
    if not os.path.isfile(weights_path):
        log( f'No such file {weights_path}' )
        return None

    json_path = f'{directory}/js.json'
    if not os.path.isfile(json_path):
        log( f'No such file {json_path}' )
        return None

    js_file = open( json_path, 'r' )
    model_json = js_file.read()
    js_file.close()
    model = model_from_json( model_json )
    model.load_weights( weights_path )
    return model

def read_model_checkpoint(directory):
    model_path = f'{directory}/model.h5'
    if not os.path.isfile(model_path):
        log( f'No such file {model_path}' )
        return None

    model = load_model(model_path)
    return model

'''
    Example:
        model_a, model_b, ... = generate_model_function(xxx) # <-- weights shared models
        read_weights( './pre_cache_folder', model_a ) #
'''
def read_weights(directory, model):
    weights_path = f'{directory}/weights.h5'

    if not os.path.isfile(weights_path):
        log( f'No such file {weights_path}' )
        return False

    model.load_weights( weights_path )
    return True

'''
    Example:
        all_images = make_image_set( './training_set' )

    this will return a numpy array with datatype uint8
'''
def make_image_set( paths, images_to_use=None, trimmed_shape=None ):

    def trim_to( image, new_shape ):
        r, c, *_ = image.shape
        r_, c_, *_ = new_shape
        offset_r, offset_c = ((r-r_) >> 1), ((c-c_)>>1)
        return image[offset_r:offset_r+r_, offset_c:offset_c+c_]

    # arrange image order
    image_paths = glob.glob( paths )
    image_paths.sort()

    # determine numbers of images in dataset
    images_to_use = len(image_paths)
    if images_to_use is not None:
        images_to_use = min( images_to_use, len(image_paths) )

    # case of empty directory
    if images_to_use == 0:
        return []

    # determine row, col and channels
    first_image = imageio.imread( image_paths[0] )

    row, col, *_ = first_image.shape
    if trimmed_shape is not None:
        row, col = trimmed_shape

    channels = 3
    if len(first_image.shape) == 2:
        channels = 1

    # load images one by one
    total_image_set = np.zeros( (images_to_use, row, col, channels), dtype='uint8' )
    if trimmed_shape is not None:
        for idx in range( images_to_use ):
            total_image_set[idx] = trim_to( np.asarray( imageio.imread( image_paths[idx] ), dtype='uint8' ), (row, col) ).reshape((row, col, channels))
    else:
        for idx in range( images_to_use ):
            total_image_set[idx] =  np.asarray( imageio.imread( image_paths[idx] ), dtype='uint8' ).reshape((row, col, channels))

    return total_image_set

if __name__ == '__main__':
    if 0:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        from tensorflow.keras.layers import Conv2D
        input_layer = Input( shape=(None, None, 3) )
        output_layer = Conv2D( 3, (3,3) )( input_layer )
        model = Model( input_layer, output_layer )
        model.summary()

        write_model('./tmp', model)

        new_model = read_model('./tmp' )
        new_model.summary()

        write_model_checkpoint( './tmp', new_model )
        model_se = read_model_checkpoint( './tmp' )
        model_se.summary()

    if 1:
        images = make_image_set( '/raid/feng/wallpapers/music/*.jpg', images_to_use=16, trimmed_shape=(512,512) )
        for idx in range( 16 ):
            imageio.imsave( f'./tmp/dumped_{idx}.png', images[idx] )


