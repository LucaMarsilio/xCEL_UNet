'''
Convolution Neural Network Archietecture
'''

import os
from utils import PEE
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import *
from tensorflow.keras.layers import (
    Concatenate, Dense, GlobalAveragePooling3D,
)


class xCELUNet(object):
    """
    This class provides a simple interface to create the xCEL_UNet network with custom parameters.
    Args:
        input_size: input_size: input size for the network. If the input of the network is two-dimensional, input size must be
        of type (input_dim_1, input_dim_2, 1). If the input is three-dimensional, then input size must be
        of type (input_dim_1, input_dim_2, input_dim_3, 1).
        kernel_size: size of the kernel to be used in the convolutional layers of the U-Net
        strides: stride shape to be used in the convolutional layers of the U-Net
        deconv_strides: stride shape to be used in the deconvolutional layers of the U-Net
        deconv_kernel_size: kernel size shape to be used in the deconvolutional layers of the U-Net
        pool_size: size of the pool size to be used in MaxPooling layers
        pool_strides: size of the strides to be used in MaxPooling layers
        depth: depth of the U-Net model
        activation: activation function used in the U-Net layers
        padding: padding used for the input data in all the U-Net layers
        n_inital_filters: number of feature maps in the first layer of the U-Net
        add_batch_normalization: boolean flag to determine if batch normalization should be applied after convolutional layers
        add_inception_module: boolean flag to determine if inception module should be applied in the first and last enc/dec layers
        retrain_layers: if true, the layers in the list will be retrain to fine-tune the segmentation model for the concurrent task
        kernel_regularizer: kernel regularizer to be applied to the convolutional layers of the U-Net
        bias_regularizer: bias regularizer to be applied to the convolutional layers of the U-Net
        n_classes: number of classes in labels
    """

    def __init__(self, input_size=(None, None, None, 1),
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 deconv_kernel_size=(2, 2, 2),
                 deconv_strides=(2, 2, 2),
                 pool_size=(2, 2, 2),
                 pool_strides=(2, 2, 2),
                 depth=3,
                 activation='relu',
                 padding='same',
                 n_initial_filters=8,
                 add_batch_normalization=True,
                 add_inception_module=True,
                 retrain_layers=True,
                 kernel_regularizer=regularizers.l2(0.001),
                 bias_regularizer=regularizers.l2(0.001),
                 n_classes=3):

        self.input_size = input_size
        self.n_dim = len(input_size)  # Number of dimensions of the input data
        self.kernel_size = kernel_size
        self.strides = strides
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_strides = deconv_strides
        self.depth = depth
        self.activation = activation
        self.padding = padding
        self.n_initial_filters = n_initial_filters
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.add_batch_normalization = add_batch_normalization
        self.add_inception_module = add_inception_module
        self.retrain_layers = retrain_layers
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.n_classes = n_classes

    def create_model(self):
        '''
        This function creates a U-Net network based
        on the configuration.
        '''
        # Check if 2D or 3D convolution must be used
        if (self.n_dim == 3):
            conv_layer = layers.Conv2D
            max_pool_layer = layers.MaxPooling2D
            conv_transpose_layer = layers.Conv2DTranspose
            softmax_kernel_size = (1, 1)
            pee_input_size = 2
        elif (self.n_dim == 4):
            conv_layer = layers.Conv3D
            max_pool_layer = layers.MaxPooling3D
            conv_transpose_layer = layers.Conv3DTranspose
            softmax_kernel_size = (1, 1, 1)
            pee_input_size = 3
        else:
            print("Could not handle input dimensions.")
            return

        # Input layer
        temp_layer = layers.Input(shape=self.input_size, name="input_layer")
        input_tensor = temp_layer

        # Variables holding the layers so that they can be concatenated
        downsampling_layers = []
        upsampling_layers = []
        # Down sampling branch: First Layer
        for j in range(2):
            if self.add_inception_module == True:
                # Inception Module
                temp_layer = self.inception_module(
                    conv_layer=conv_layer,
                    input_layer=temp_layer,
                    n_filters=self.n_initial_filters,
                    first_kernel_size=self.kernel_size,
                    stride=self.strides,
                    pad=self.padding,
                    activation='linear',
                    large_kernels=True
                )
            if self.add_inception_module == False:
                # Standard Convolution
                temp_layer = conv_layer(self.n_initial_filters,
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        activation='linear',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer,
                                        name=f'conv3d_{j+1}_1')(temp_layer)
            # Batch Normalization
            if self.add_batch_normalization:
                temp_layer = layers.BatchNormalization(
                    axis=-1, fused=False, name=f'bn_{j+1}_1')(temp_layer)
            # Activation Layer
            temp_layer = layers.Activation(
                self.activation, name=f'act_{j+1}_1')(temp_layer)
        # Append for Skip Connection
        downsampling_layers.append(temp_layer)
        # Apply Max Pooling
        temp_layer = max_pool_layer(pool_size=self.pool_size,
                                    strides=self.pool_strides,
                                    padding=self.padding,
                                    name=f'max_pool3d_1')(temp_layer)

        # Down sampling branch: Remaining Layers
        for i in range(self.depth - 1):
            for j in range(2):
                # Standard Convolution
                temp_layer = conv_layer(self.n_initial_filters * pow(2, i + 1),
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        activation='linear',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer,
                                        name=f'conv3d_{j+1}_{i+2}')(temp_layer)
                # Batch Normalization
                if self.add_batch_normalization:
                    temp_layer = layers.BatchNormalization(
                        axis=-1, fused=False, name=f'bn_{j+1}_{i+2}')(temp_layer)
                # Activation Layer
                temp_layer = layers.Activation(
                    self.activation, name=f'act_{j+1}_{i+2}')(temp_layer)
            # Append for Skip Connection
            downsampling_layers.append(temp_layer)
            # Apply Max Pooling
            temp_layer = max_pool_layer(pool_size=self.pool_size,
                                        strides=self.pool_strides,
                                        padding=self.padding,
                                        name=f'max_pool3d_{i+2}')(temp_layer)

        for j in range(2):
            # Bottleneck
            temp_layer = conv_layer(self.n_initial_filters * pow(2, self.depth), kernel_size=self.kernel_size,
                                    strides=self.strides,
                                    padding=self.padding,
                                    activation='linear',
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    name=f'conv_bottleneck_{j + 1}')(temp_layer)
            if self.add_batch_normalization:
                temp_layer = layers.BatchNormalization(
                    axis=-1,
                    fused=False,
                    name=f'bn_bottleneck_{j + 1}')(temp_layer)
            # activation
            temp_layer = layers.Activation(
                self.activation, name=f'act_bottleneck_{j + 1}')(temp_layer)

        # Classification branch
        class_temp_layer = temp_layer

        # Up sampling branch
        temp_layer_edge = temp_layer
        temp_layer_merge = temp_layer

        for i in range(self.depth):
            # EDGE PATH
            temp_layer_edge = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                                   kernel_size=self.deconv_kernel_size,
                                                   strides=self.deconv_strides,
                                                   activation='linear',
                                                   padding=self.padding,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   bias_regularizer=self.bias_regularizer)(temp_layer_edge)
            temp_layer_edge = layers.Activation(
                self.activation)(temp_layer_edge)

            # MASK PATH
            temp_layer_mask = conv_transpose_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                                   kernel_size=self.deconv_kernel_size,
                                                   strides=self.deconv_strides,
                                                   activation='linear',
                                                   padding=self.padding,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   bias_regularizer=self.bias_regularizer)(temp_layer_merge)
            temp_layer_mask = layers.Activation(
                self.activation)(temp_layer_mask)

            # Concatenation
            temp_layer_edge = layers.Concatenate(axis=self.n_dim)(
                [downsampling_layers[(self.depth - 1) - i], temp_layer_edge])
            temp_layer_mask = layers.Concatenate(axis=self.n_dim)(
                [downsampling_layers[(self.depth - 1) - i], temp_layer_mask])

            for j in range(2):
                temp_layer_edge = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                             kernel_size=self.kernel_size,
                                             strides=self.strides,
                                             padding=self.padding,
                                             activation='linear',
                                             kernel_regularizer=self.kernel_regularizer,
                                             bias_regularizer=self.bias_regularizer)(temp_layer_edge)
                if self.add_batch_normalization:
                    temp_layer_edge = layers.BatchNormalization(
                        axis=-1, fused=False)(temp_layer_edge)
                temp_layer_edge = layers.Activation(
                    self.activation)(temp_layer_edge)

                temp_layer_mask = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                             kernel_size=self.kernel_size,
                                             strides=self.strides,
                                             padding=self.padding,
                                             activation='linear',
                                             kernel_regularizer=self.kernel_regularizer,
                                             bias_regularizer=self.bias_regularizer)(temp_layer_mask)
                if self.add_batch_normalization:
                    temp_layer_mask = layers.BatchNormalization(
                        axis=-1, fused=False)(temp_layer_mask)
                temp_layer_mask = layers.Activation(
                    self.activation)(temp_layer_mask)

            # if i % 2 != 1:
            temp_layer_edge = PEE(temp_layer_edge, self.n_initial_filters *
                                  pow(2, (self.depth - 1) - i), input_dims=pee_input_size)

            temp_layer_merge = Concatenate()(
                [temp_layer_edge, temp_layer_mask])
            temp_layer_merge = conv_layer(self.n_initial_filters * pow(2, (self.depth - 1) - i),
                                          kernel_size=self.kernel_size,
                                          strides=self.strides,
                                          padding=self.padding,
                                          activation='linear',
                                          kernel_regularizer=self.kernel_regularizer,
                                          bias_regularizer=self.bias_regularizer)(temp_layer_merge)

        # Convolution 1 filter sigmoidal (to make size converge to final one)
        temp_layer_mask = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                     strides=self.strides,
                                     padding='same',
                                     activation='linear',
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer)(temp_layer_merge)

        temp_layer_edge = conv_layer(self.n_classes, kernel_size=softmax_kernel_size,
                                     strides=self.strides,
                                     padding='same',
                                     activation='linear',
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer)(temp_layer_edge)

        # Classification output --> GAP to the bottleneck feature maps
        class_temp_layer = GlobalAveragePooling3D(name='gap')(class_temp_layer)

        # Osteophite size output
        temp_layer_osteo = Dense(
            self.n_initial_filters * pow(2, self.depth - 1),
            activation='relu',
            name='dense_0_osteo')(class_temp_layer)
        temp_layer_osteo = Dense(
            self.n_initial_filters * pow(2, self.depth - 3),
            activation='relu',
            name='dense_1_osteo')(temp_layer_osteo)
        out_osteo = Dense(
            3,
            activation='softmax',
            name='out_osteophyte')(temp_layer_osteo)

        # GH joint space following kellgren lawrence grading
        temp_layer_kl = Dense(
            self.n_initial_filters * pow(2, self.depth - 1),
            activation='relu',
            name='dense_0_kl')(class_temp_layer)
        temp_layer_kl = Dense(
            self.n_initial_filters * pow(2, self.depth - 3),
            activation='relu',
            name='dense_1_kl')(temp_layer_kl)
        out_kl = Dense(
            3,
            activation='softmax',
            name='out_impingement')(temp_layer_kl)

        # Humeroscapular alignment Output
        temp_layer_hsa = Dense(
            self.n_initial_filters * pow(2, self.depth - 1),
            activation='relu',
            name='dense_0_hsa')(class_temp_layer)
        temp_layer_hsa = Dense(
            self.n_initial_filters * pow(2, self.depth - 3),
            activation='relu',
            name='dense_1_hsa')(temp_layer_hsa)
        out_hsa = Dense(
            1,
            activation='sigmoid',
            name='out_hsa')(temp_layer_hsa)

        # Segmentation outputs
        output_tensor_edge = layers.Softmax(
            axis=-1, dtype='float32', name='out_edge')(temp_layer_edge)
        output_tensor_mask = layers.Softmax(
            axis=-1, dtype='float32', name='out_mask')(temp_layer_mask)

        # Define model
        self.model = Model(inputs=[input_tensor],
                           outputs=[
            output_tensor_edge,
            output_tensor_mask,
            out_osteo,
            out_kl,
            out_hsa
        ]
        )

        if self.retrain_layers == True:
            # Layers in this list will be retrained
            class_layers_names = [
                "conv_bottleneck_1",
                "conv_bottleneck_2",
                "dense_0_osteo",
                "dense_1_osteo",
                "dense_0_kl",
                "dense_1_kl",
                "dense_0_hsa",
                "dense_1_hsa",
                "out_osteophyte",
                "out_impingement",
                "out_hsa"
            ]

            for layer in self.model.layers:
                layer.trainable = False
                for class_layer in class_layers_names:
                    if layer.name == class_layer:
                        layer.trainable = True
                        break

    def inception_module(
            self, conv_layer, input_layer, n_filters, first_kernel_size, stride, pad, activation, large_kernels):
        '''
        Adds a 3 branch Inception module to the encoding path of a UNET like network
        Input: conv_layer         -> conv2D, conv2DTranspose, conv3D, conv3DTranspose
               input_layer        -> output of the previous network layer
               n_filters          -> number of feature maps
               first_kernel_size  -> kernel size of the first inception branch
               stride             -> stride for the conv filter
               pad                -> padding type for the conv filter
               activation         -> activation type for the conv filter
               large_kernels      -> if True  ks_1=f_k_s, ks_2=f_k_s+4, ks_3=f_k_s+8
                                     if False ks_1=f_k_s, ks_2=f_k_s+2, ks_3=f_k_s+4
        Output: output of the inception layer
        '''
        if large_kernels is False:
            second_kernel_size = tuple(
                [size + 2 for size in first_kernel_size])
            third_kernel_size = tuple([size + 4 for size in first_kernel_size])
        elif large_kernels is True:
            second_kernel_size = tuple(
                [size + 4 for size in first_kernel_size])
            third_kernel_size = tuple([size + 8 for size in first_kernel_size])
        else:
            raise ValueError("large_kernels can be set to True or False.")
        # First Inception Branch
        inception_1 = conv_layer(n_filters,
                                 kernel_size=first_kernel_size,
                                 strides=stride,
                                 padding=pad,
                                 activation=activation,
                                 kernel_regularizer=self.kernel_regularizer,
                                 bias_regularizer=self.bias_regularizer)(input_layer)
        # Second Inception Branch
        inception_2 = conv_layer(n_filters,
                                 kernel_size=second_kernel_size,
                                 strides=stride,
                                 padding=pad,
                                 activation=activation,
                                 kernel_regularizer=self.kernel_regularizer,
                                 bias_regularizer=self.bias_regularizer)(input_layer)
        # Third Inception Branch
        inception_3 = conv_layer(n_filters,
                                 kernel_size=third_kernel_size,
                                 strides=stride,
                                 padding=pad,
                                 activation=activation,
                                 kernel_regularizer=self.kernel_regularizer,
                                 bias_regularizer=self.bias_regularizer)(input_layer)
        # Layers Concatenation
        inception_output = layers.Concatenate(axis=self.n_dim)(
            [inception_1, inception_2, inception_3])
        # Return to Original Feature Maps
        inception_output = conv_layer(n_filters,
                                      kernel_size=first_kernel_size,
                                      strides=stride,
                                      padding=pad,
                                      activation=activation,
                                      kernel_regularizer=self.kernel_regularizer,
                                      bias_regularizer=self.bias_regularizer)(inception_output)

        return inception_output

    def set_initial_weights(self, weights):
        '''
        Set the initial weights of the U-Net, in case
        training was stopped and then resumed. An exception
        is raised in case the model currently configured
        has different properties than the one whose weights
        were stored.
        '''
        try:
            self.model.load_weights(weights)
        except:
            raise

    def get_n_parameters(self):
        '''
        Get the total number of parameters of the model
        '''
        return self.model.count_params()

    def plot_model(self):
        '''
        Plot Model
        '''
        tf.keras.utils.plot_model(
            self.model, to_file=os.path.join(os.getcwd(), 'model.png'))

    def summary(self):
        '''
        Print out summary of the model.
        '''
        print(self.model.summary())
