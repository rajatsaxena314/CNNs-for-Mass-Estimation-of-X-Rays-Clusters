from keras import layers, models, Input
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from loss import heteroscedastic_loss

def build_model():
    """
    Builds and compiles a CNN model that takes image and redshift inputs,
    and outputs a single regression value or outputs the mean and the log variance.

    Returns:
        A compiled Keras model.
    """
    alpha = 0.1

    #Start of the Convolution Layer
    image_input = Input(shape=(50, 50, 10), name='image_input') # Image input branch
    x = layers.Conv2D(16, (4, 4), activation=LeakyReLU(alpha=alpha))(image_input) # A layer with 16 Neurons and a filter of 4*4
    x = layers.Conv2D(16, (4, 4), activation=LeakyReLU(alpha=alpha))(x) # A layer with 16 Neurons and a filter of 4*4
    x = layers.Conv2D(32, (2, 2), activation=LeakyReLU(alpha=alpha))(x) # A layer with 32 Neurons and a filter of 2*2
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid")(x) # Applies 2x2 average pooling with stride 1 and no padding
    x = layers.Flatten()(x) #Data is Flattened 
    #End of the Convolution Layer

    # Redshift input branch
    redshift_input = Input(shape=(1,), name='redshift_input')

    # Concatenate and Dense Layers
    combined = layers.Concatenate()([x, redshift_input])
    combined = layers.Dense(32, activation=LeakyReLU(alpha=alpha))(combined) #A layer with 32 Neurons
    combined = layers.Dense(16, activation=LeakyReLU(alpha=alpha))(combined) #A layer with 16 Neurons
    combined = layers.Dense(8, activation=LeakyReLU(alpha=alpha))(combined)  #A layer with 08 Neurons
    output = layers.Dense(2, name='output')(combined)
    #End of Dense Layers

    # Build model
    model = models.Model(inputs=[image_input, redshift_input], outputs=output)

    #Compile Model
    model.compile(
        optimizer=Adam(1e-4),                   # Adam optimizer with low learning rate
        loss=heteroscedastic_loss,              # Custom loss for uncertainty-aware regression
        metrics=[heteroscedastic_loss]          # Also track the same loss as a metric
    )

    return model
