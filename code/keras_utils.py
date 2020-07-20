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

if __name__ == '__main__':
    if 1:
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

