from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras import Model
import tensorflow as tf
import typing

tf.config.run_functions_eagerly(True)

@tf.function
def VGGNet(
    name: str,
    architecture: typing.List[ typing.Union[int, str] ],
    input_shape: typing.Tuple[int],
    classes: int = 1000
) -> Model:
  
    X_input = Input(input_shape)

 
    X = make_conv_layer(X_input, architecture)

    X = Flatten()(X)
    X = make_dense_layer(X, 4096)
    X = make_dense_layer(X, 4096)

    
    X = Dense(units = classes, activation = "softmax")(X)

    model = Model(inputs = X_input, outputs = X, name = name)
    return model

def make_conv_layer(
    X: tf.Tensor,
    architecture: typing.List[ typing.Union[int, str] ],
    activation: str = 'relu'
) -> tf.Tensor:
   
    for output in architecture:

      
        if type(output) == int:
            out_channels = output

            X = Conv2D(
                filters = out_channels,
                kernel_size = (3, 3),
                strides = (1, 1),
                padding = "same"
            )(X)
            X = BatchNormalization()(X)
            X = Activation(activation)(X)

          

       
        else:
            X = MaxPooling2D(
                pool_size = (2, 2),
                strides = (2, 2)
            )(X)

    return X

def make_dense_layer(X: tf.Tensor, output_units: int, dropout = 0.5, activation = 'relu') -> tf.Tensor:

    X = Dense(units = output_units)(X)
    X = BatchNormalization()(X)
    X = Activation(activation)(X)
    X = Dropout(dropout)(X)

    return X
