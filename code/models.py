# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# and https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py
#
from functools import partial
from instance_normalization import InstanceNormalization
from math import exp, sqrt
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import Subtract
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.python.keras.layers.merge import _Merge
import numpy as np
import tensorflow.keras.backend as K

batch_size = 6
batch_size = 3

layer_counter = 0
def unique_name():
    global layer_counter
    layer_counter += 1
    return 'Layer_'+str(layer_counter).zfill(5)

class RandomWeightedAverage(_Merge):
    def _merge_function(self,inputs):
        global batch_size
        alpha = K.random_uniform((batch_size, 128, 128, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def make_activation( input_layer, with_normalization=True ):
    if with_normalization:
        return LeakyReLU(alpha=0.2, name=unique_name())(InstanceNormalization(name=unique_name())(input_layer))
    return LeakyReLU(alpha=0.2, name=unique_name())(input_layer)

def make_pooling( input_layer, channels, with_normalization=True ):
    x = conv2d_transpose( channels, kernel_size=(3,3), activation='linear', strides=1, padding='valid')( input_layer )
    x = make_activation( x, with_normalization )
    x = conv2d( channels, kernel_size=(3,3), activation='linear', strides=2, padding='valid')( x )
    x = make_activation( x, with_normalization )
    return x

def build_critic( input_shape=(128, 128, 1) ):
    def build_scaling_block( channels, input_layer, with_bn=True ):
        return Dropout(0.25)( make_pooling( input_layer, channels, with_normalization=with_bn ) )

    l0 = Input( shape=input_shape )
    l1 = build_scaling_block( 64,  l0, with_bn=False ) # 16x64^2
    l2 = build_scaling_block( 64,  l1 ) # 32x32^2
    l3 = build_scaling_block( 128,  l2 ) # 64x16^2
    l4 = build_scaling_block( 128, l3 ) # 128x8^2

    l_last = l4
    output = Dense(1)(Flatten()(l_last))

    return Model( l0, output )

def conv2d_transpose( *args,**kwargs ):
    if 'name' in kwargs:
        return Conv2DTranspose( *args, **kwargs )
    return Conv2DTranspose( *args, **kwargs, name=unique_name() )

def conv2d( *args,**kwargs ):
    if 'name' in kwargs:
        return Conv2D( *args, **kwargs )
    return Conv2D( *args, **kwargs, name=unique_name() )

def make_block( input_layer, channels, kernel_size=(3,3), with_normalization=True ):
    x = input_layer
    x = conv2d_transpose( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid')( x )
    x = make_activation( x, with_normalization )
    x = conv2d( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid')( x )
    x = make_activation( x, with_normalization )
    return x

def make_output_block( input_layer, output_channels, kernel_size, output_activation ):
    channels = output_channels << 3
    x = input_layer
    x = conv2d_transpose( channels, kernel_size=kernel_size, activation='linear', strides=1, padding='valid')( x )
    x = make_activation( x )
    x = conv2d( output_channels, kernel_size=kernel_size, activation=output_activation, strides=1, padding='valid')( x )
    return x

def make_upsampling( input_layer, channels ):
    x = conv2d_transpose( channels, kernel_size=(4,4), activation='linear', strides=2, padding='valid')( input_layer )
    x = make_activation( x )
    x = conv2d( channels, kernel_size=(3,3), activation='linear', strides=1, padding='valid')( x )
    x = make_activation( x )
    return x

def make_xception_blocks( input_layer, channels, kernel_sizes ):
    sub_channels = int( channels/len(kernel_sizes) )
    assert sub_channels * len(kernel_sizes) == channels, 'sub-channels and channels not match, adjust the channels or the size of sub-kernels'
    layer_blocks = []
    for kernel_size in kernel_sizes:
        layer_blocks.append( make_block( input_layer, sub_channels, kernel_size ) )
    return concatenate( layer_blocks )

def add( layers ):
    return Add(name=unique_name())( layers )

def make_blocks( n_blocks, input_layer, channels, kernel_size=(3,3) ):
    x = make_block( input_layer, channels, kernel_size )
    for idx in range( n_blocks ):
        x_ = make_block( x, channels, kernel_size )
        x = add( [x_, x] )
    return x

def generator( input_channels=1, output_channels=1, transform_repeater=16, output_activation='tanh', name=None ):
    input_layer = Input( shape=(None, None, input_channels) )

    encoder_128 = make_xception_blocks( make_block( input_layer, 64, with_normalization=False ), 64, (1, 3, 5, 7) )
    encoder_64 = make_xception_blocks( make_block( make_pooling( encoder_128, 64 ), 64 ), 64, (3, 5) )
    encoder_32 = make_xception_blocks( make_block( make_pooling( encoder_64, 64 ), 64 ), 64, (3, 5) )
    encoder_16 = make_xception_blocks( make_block( make_pooling( encoder_32, 128 ), 128 ), 128, (3, 5) )

    transformer = make_blocks( transform_repeater, encoder_16, 192 )

    decoder_32 = make_xception_blocks( make_block( add( [encoder_32, make_upsampling( transformer, 64 )]), 64 ), 64, (3, 5) )
    decoder_64 = make_xception_blocks( make_block( add( [encoder_64, make_upsampling( decoder_32, 64 )]), 64 ), 64, (3, 5) )
    decoder_128= make_xception_blocks( make_block( add( [encoder_128, make_upsampling( decoder_64, 64 )]), 64 ), 64, (1, 3, 5, 7) )

    output_layer = make_output_block( decoder_128, output_channels, (9, 9), output_activation )

    if name is None:
        name = 'generator_model'

    return Model( input_layer, output_layer, name=name )

def lpf():
    def make_lpf():
        input_layer = Input( shape=(None, None, 1) )
        gaussian_30_ = conv2d( 1, kernel_size=(33,33), activation='linear', strides=(1,1), padding='same', name='gaussian_layer', use_bias=False, trainable=False )
        gaussian_30 = gaussian_30_( input_layer )
        return Model( input_layer, gaussian_30, name='low_pass_filter' )

    def kernel_gaussian( sigma ):
        kernel = np.zeros( (33, 33) )
        for r in range( 33 ):
            for c in range( 33 ):
                if (r*r + c*c) < 17*17:
                    kernel[r][c] = exp( -((r-16)^2+(c-16)^2)/(sigma+sigma) )
        kernel /= np.sum( kernel )
        return kernel.reshape( (33, 33, 1, 1) )

    model = make_lpf()
    model.get_layer('gaussian_layer').set_weights([kernel_gaussian(15.),])
    return model

from keras_utils import read_model
def classifier( path='./cluster_classifier_64' ): #xception model, pretrained, see `xception.py`
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        m = read_model( path )
        return m

def cycle_classifier(input_shape=(128,128,1)):
    noisy_to_clean_model = generator( transform_repeater=16, name='noisy2clean' )
    clean_to_noisy_model = generator( transform_repeater=16, name='clean2noisy' )

    low_pass_filter_model = lpf()

    noisy_input_layer = Input( shape=input_shape, name='noisy_input' )
    blurry_noisy_input = low_pass_filter_model( noisy_input_layer )
    noisy_lpf_output_layer = low_pass_filter_model( clean_to_noisy_model( noisy_to_clean_model( noisy_input_layer ) ) ) # LPF is applied
    noisy_to_noisy_lpf_model = Model( noisy_input_layer, noisy_lpf_output_layer, name='noisy2noisyLPF' )

    clean_input_layer = Input( shape=input_shape, name='clean_input' )
    clean_output_layer = noisy_to_clean_model( clean_to_noisy_model( clean_input_layer ) )
    clean_to_clean_model = Model( clean_input_layer, clean_output_layer, name='clean2clean' )

    should_be_zero_lpf_difference_layer = Subtract( name='lpf_difference' )( [noisy_lpf_output_layer, blurry_noisy_input] )
    should_be_zero_clear_difference_layer = Subtract( name='clear_difference' )( [clean_input_layer, clean_output_layer] )


    cycle_training_model = Model( inputs=[noisy_input_layer, clean_input_layer],
                                  outputs=[should_be_zero_lpf_difference_layer, should_be_zero_clear_difference_layer], name='cycle_training' )

    return ( cycle_training_model, noisy_to_clean_model, clean_to_noisy_model)

def build_wgan( input_shape=(128, 128, 1) ):
    optimizer = RMSprop(lr=0.00005)

    cycle_training_model, noisy_to_clean_model, clean_to_noisy_model = cycle_classifier(input_shape=input_shape)
    critic = build_critic(input_shape=input_shape)

    cycle_training_model.compile(loss='mae', optimizer=optimizer)

    noisy_to_clean_model.trainable = False
    real_image = Input(shape=input_shape)
    valid = critic( real_image )
    noisy_image = Input(shape=input_shape)
    fake_image = noisy_to_clean_model(noisy_image)
    fake = critic( fake_image )
    interpolated_img = RandomWeightedAverage()([real_image, fake_image])
    validity_interpolated = critic(interpolated_img)
    partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty'
    critic_model = Model(inputs=[real_image, noisy_image], outputs=[valid, fake, validity_interpolated])
    critic_model.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss], optimizer=optimizer, loss_weights=[1, 1, 10])

    noisy_to_clean_model.trainable = True
    critic.trainable = False
    new_noisy_image = Input(shape=input_shape)
    denoised_image = noisy_to_clean_model( new_noisy_image )
    valid = critic( denoised_image )
    noisy_to_clean_generator = Model( new_noisy_image, valid )
    noisy_to_clean_generator.compile(loss=wasserstein_loss, optimizer=optimizer)

    return critic_model, noisy_to_clean_generator, cycle_training_model, noisy_to_clean_model

from tensorflow.keras.utils import plot_model
import imageio
import numpy as np

if __name__ == '__main__':
    if 1:
        c, n2cg, ct, n2c = build_wgan()
        c.summary()
        n2c.summary()
        ct.summary()
        n2c.summary()
    if 0:
        critic = build_critic()
        critic.summary()
    if 0:
        m = generator(transform_repeater=32)
        m.summary()
    if 0: # test lpf
        m = lpf()
        #image = np.asarray(imageio.imread( '/raid/feng/pictures/768_768/37339.jpg' ), dtype='float32') # load a random image
        image = np.asarray(imageio.imread( './frame_1.tif' ), dtype='float32') # load a random image
        #image = (image[:,:,0] + image[:,:,1] + image[:,:,2]) / 3.0
        image = image / np.amax(image)
        imageio.imsave( './noisy.png', image )
        noisy = image.reshape( (1,)+image.shape+(1,) )
        lpf = np.squeeze( m.predict( noisy ) )
        lpf = lpf / np.amax(lpf)
        imageio.imsave( './lpf.png', lpf )
    if 0:
        m = lpf()
        m.summary()
    if 0:
        m = classifier()
        m.summary()
    if 0:
        c, ct, n2cc, n2c, c2n = cycle_classifier()
        ct.summary()

