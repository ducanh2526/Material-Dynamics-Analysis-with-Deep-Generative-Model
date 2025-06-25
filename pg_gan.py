import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Add, Layer
from tensorflow.keras import backend

class PixelNormalization(Layer):
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        mean_square = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        l2 = tf.math.rsqrt(mean_square + 1.0e-8)
        normalized = inputs * l2
        return normalized

    def compute_output_shape(self, input_shape):
        return input_shape
    
class MinibatchStdev(Layer):
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)
    
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True) + 1e-8)
        average_stddev = tf.reduce_mean(stddev, keepdims=True)
        shape = tf.shape(inputs)
        tiled_shape = (shape[0],) + tuple([shape[i] for i in range(1, len(inputs.shape) - 1)]) + (1,)
        minibatch_stddev = tf.tile(average_stddev, tiled_shape)
        combined = tf.concat([inputs, minibatch_stddev], axis=-1)
        
        return combined
    
    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)
    

class WeightedSum(Add):
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')
    
    def _merge_function(self, inputs):
        assert (len(inputs) == 2)
        output = ((1.0 - self.alpha) * inputs[0] + (self.alpha * inputs[1]))
        return output

class WeightScaling(Layer):
    def __init__(self, shape, gain = np.sqrt(2), dtype=tf.float32, **kwargs):
        super(WeightScaling, self).__init__(**kwargs)
        shape = np.asarray(shape)
        shape = tf.constant(shape, dtype=tf.float32)
        fan_in = tf.math.reduce_prod(shape)
        self.wscale = gain*tf.math.rsqrt(fan_in+1e-8)
      
    def call(self, inputs, **kwargs):
        return inputs * self.wscale
    
    def compute_output_shape(self, input_shape):
        return input_shape

class Bias(Layer):
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value = b_init(shape=(input_shape[-1],), dtype='float32'), trainable=True)  

    def call(self, inputs, **kwargs):
        return inputs + self.bias
    
    def compute_output_shape(self, input_shape):
        return input_shape  

def WeightScalingDense(x, filters, gain, use_pixelnorm=False, activate=None):
    init = RandomNormal(mean=0., stddev=1.)
    in_filters = backend.int_shape(x)[-1]
    x = layers.Dense(filters, use_bias=False, kernel_initializer=init, dtype='float32')(x)
    x = WeightScaling(shape=(in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    if activate=='LeakyReLU':
        x = layers.LeakyReLU(0.2)(x)
    elif activate=='tanh':
        x = layers.Activation('tanh')(x)
    
    if use_pixelnorm:
        x = PixelNormalization()(x)
    return x

def WeightScalingConvND(x, filters, kernel_size, gain, use_pixelnorm=False, activate=None, strides=1, conv_type='2d'):
    Conv = layers.Conv2D if conv_type == '2d' else layers.Conv3D
    strides = strides if isinstance(strides, tuple) else (strides,) * (3 if conv_type == '3d' else 2)
    init = RandomNormal(0., 1.)
    in_filters = backend.int_shape(x)[-1]
    x = Conv(filters, kernel_size, strides=strides, use_bias=False, padding="same", kernel_initializer=init)(x)
    ks_shape = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * (3 if conv_type == '3d' else 2)
    x = WeightScaling(shape=ks_shape + (in_filters,), gain=gain)(x)
    x = Bias()(x)
    if activate == 'LeakyReLU':
        x = layers.LeakyReLU(0.2)(x)
    elif activate == 'tanh':
        x = layers.Activation('tanh')(x)
    if use_pixelnorm:
        x = PixelNormalization()(x)
    return x

class PGAN(Model):
    def __init__(self, latent_dim, filters, channels=1, d_steps=1, gp_weight=10.0, drift_weight=0.001, is_3d=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.filters = filters
        self.channels = channels
        self.d_steps = d_steps
        self.gp_weight = gp_weight
        self.drift_weight = drift_weight
        self.n_depth = 0
        self.is_3d = is_3d

        self.discriminator = self.init_discriminator()
        self.generator = self.init_generator()
        self.generator_stabilize = None
        self.discriminator_stabilize = None

    def sampler(self, args):
        z_mean, z_log_var = args
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, tf.shape(z_mean)[1]))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def init_generator(self):
        InputNoise = layers.Input(shape=(self.latent_dim,))
        conv_type = '3d' if self.is_3d else '2d'

        x = PixelNormalization()(InputNoise)
        shape_prod = 4*4 if conv_type=='2d' else 4*4*4
        x = WeightScalingDense(x, shape_prod * self.filters[0], gain=np.sqrt(2)/4, use_pixelnorm=True, activate='LeakyReLU')
        init_shape = (4, 4) if conv_type == '2d' else (4, 4, 4)
        x = layers.Reshape(init_shape + (self.filters[0],))(x)

        x = WeightScalingConvND(x, self.filters[0], kernel_size=4, gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True, conv_type=conv_type)
        x = WeightScalingConvND(x, self.filters[0], kernel_size=3, gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True, conv_type=conv_type)
        x = WeightScalingConvND(x, filters=self.channels, kernel_size=1, gain=1., activate='tanh', use_pixelnorm=False, conv_type=conv_type)

        return Model(InputNoise, x, name='generator')

    def init_discriminator(self):
        conv_type = '3d' if self.is_3d else '2d'
        init_shape = (4, 4) if conv_type == '2d' else (4, 4, 4)
        img_input = layers.Input(shape = init_shape+(self.channels, ),dtype=tf.float32)
        conv_type = '3d' if self.is_3d else '2d'

        x = WeightScalingConvND(img_input, self.filters[0], kernel_size=1, gain=np.sqrt(2), activate='LeakyReLU', conv_type=conv_type)
        x = MinibatchStdev()(x)
        x = WeightScalingConvND(x, self.filters[0], kernel_size=3, gain=np.sqrt(2), activate='LeakyReLU', conv_type=conv_type)
        x = WeightScalingConvND(x, self.filters[0], kernel_size=4, gain=np.sqrt(2), strides=(4,)*len(init_shape), activate='LeakyReLU', conv_type=conv_type)

        x = layers.Flatten()(x)
        x = WeightScalingDense(x, filters=1, gain=1.)
        return Model(img_input, x, name='discriminator')
    
    def fade_in_generator(self):
        conv_type = '3d' if self.is_3d else '2d'
        base = self.generator.layers[-5].output  # block before toRGB
        block_end = layers.UpSampling3D((2,2,2))(base) if self.is_3d else layers.UpSampling2D((2,2))(base)

        x1 = self.generator.layers[-4](block_end)
        x1 = self.generator.layers[-3](x1)
        x1 = self.generator.layers[-2](x1)
        x1 = self.generator.layers[-1](x1)

        x2 = WeightScalingConvND(block_end, filters=self.filters[self.n_depth], kernel_size=3, gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True, conv_type=conv_type)
        x2 = WeightScalingConvND(x2, filters=self.filters[self.n_depth], kernel_size=3, gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True, conv_type=conv_type)
        x2 = WeightScalingConvND(x2, filters=self.channels, kernel_size=1, gain=1., activate='tanh', use_pixelnorm=False, conv_type=conv_type)

        self.generator_stabilize = Model(self.generator.input, x2, name='generator')
        blended = WeightedSum()([x1, x2])
        self.generator = Model(self.generator.input, blended, name='generator')

    def fade_in_discriminator(self):
        conv_type = '3d' if self.is_3d else '2d'
        img_shape = list(self.discriminator.input.shape)
        img_shape = (img_shape[1]*2, img_shape[2]*2, img_shape[3]*2, img_shape[4]) if self.is_3d else (img_shape[1]*2, img_shape[2]*2, img_shape[3])
        img_input = layers.Input(shape = img_shape, dtype=tf.float32)
        
        x1 = layers.AveragePooling3D()(img_input) if self.is_3d else layers.AveragePooling2D()(img_input)
        x1 = self.discriminator.layers[1](x1) # Conv2D FromRGB
        x1 = self.discriminator.layers[2](x1) # WeightScalingLayer
        x1 = self.discriminator.layers[3](x1) # Bias
        x1 = self.discriminator.layers[4](x1) # LeakyReLU

        x2 = WeightScalingConvND(img_input, filters=self.filters[self.n_depth], kernel_size=1, gain=np.sqrt(2), activate='LeakyReLU', conv_type=conv_type)
        x2 = WeightScalingConvND(x2, filters=self.filters[self.n_depth], kernel_size=3, gain=np.sqrt(2), activate='LeakyReLU', conv_type=conv_type)
        x2 = WeightScalingConvND(x2, filters=self.filters[self.n_depth - 1], kernel_size=3, gain=np.sqrt(2), activate='LeakyReLU', conv_type=conv_type)
        x2 = layers.AveragePooling3D()(x2) if self.is_3d else layers.AveragePooling2D()(x2)

        # 4. Weighted Sum x1 and x2 to smoothly put the "fade in" block. 
        x = WeightedSum()([x1, x2])

        # 5. Add existing discriminator layers. 
        for i in range(5, len(self.discriminator.layers)):
            x2 = self.discriminator.layers[i](x2)
        self.discriminator_stabilize = Model(img_input, x2, name='discriminator')
        
        for i in range(5, len(self.discriminator.layers)):
            x = self.discriminator.layers[i](x)
        self.discriminator = Model(img_input, x, name='discriminator')

    def stabilize_generator(self):
        self.generator = self.generator_stabilize

    def stabilize_discriminator(self):
        self.discriminator = self.discriminator_stabilize

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha_shape = [batch_size] + [1] * len(real_images.shape[1:])
        alpha = tf.random.uniform(shape=alpha_shape, minval=0., maxval=1.)
        interpolated = real_images + alpha * (fake_images - real_images)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=list(range(1, len(grads.shape)))))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha_shape = [batch_size] + [1] * len(real_images.shape[1:])
        alpha = tf.random.uniform(shape=alpha_shape, minval=0., maxval=1.)
        interpolated = real_images + alpha * (fake_images - real_images)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=list(range(1, len(grads.shape)))))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        for _ in range(self.d_steps):
            z = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_images = self.generator(z, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)
                d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                drift = tf.reduce_mean(tf.square(real_logits))
                drift = tf.clip_by_value(drift, 0.0, 5000.0)
                d_loss = d_cost + self.gp_weight * gp + self.drift_weight * drift
            grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated = self.generator(z, training=True)
            gen_logits = self.discriminator(generated, training=True)
            g_loss = -tf.reduce_mean(gen_logits)
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {'d_loss': d_loss, 'g_loss': g_loss, 'gp': gp, 'drift': drift}
