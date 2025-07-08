import tensorflow as tf
import numpy as np

class GroupNormalization(tf.keras.layers.Layer):
    """MODIFIED FROM
        https://github.com/keras-team/keras/blob/v3.3.3/keras/src/layers/normalization/group_normalization.py#L10-L219
    """

    def __init__(
        self,
        groups=32,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError(
                f"Axis {self.axis} of input tensor should have a defined "
                "dimension but the layer received an input with shape "
                f"{input_shape}."
            )

        if self.groups == -1:
            self.groups = dim

        if dim < self.groups:
            raise ValueError(
                f"Number of groups ({self.groups}) cannot be more than the "
                f"number of channels ({dim})."
            )

        if dim % self.groups != 0:
            raise ValueError(
                f"Number of groups ({self.groups}) must be a multiple "
                f"of the number of channels ({dim})."
            )

        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

        if self.scale:
            self.gamma = self.add_weight(
                shape=(dim,),
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                shape=(dim,),
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

        super().build(input_shape)

    def call(self, inputs):
        reshaped_inputs = self._reshape_into_groups(inputs)
        normalized_inputs = self._apply_normalization(
            reshaped_inputs, inputs.shape
        )
        return tf.reshape(normalized_inputs, tf.shape(inputs))

    def _reshape_into_groups(self, inputs):
        input_shape = tf.shape(inputs)
        group_shape = list(inputs.shape)
        group_shape[0] = -1
        for i, e in enumerate(group_shape[1:]):
            if e is None:
                group_shape[i + 1] = input_shape[i + 1]

        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs

    def _apply_normalization(self, reshaped_inputs, input_shape):
        group_reduction_axes = list(range(1, len(reshaped_inputs.shape)))

        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        broadcast_shape = self._create_broadcast_shape(input_shape)
        mean, variance = tf.nn.moments(
            reshaped_inputs, axes=group_reduction_axes, keepdims=True
        )

        # Compute the batch normalization.
        inv = tf.math.rsqrt(variance + self.epsilon)
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)
            gamma = tf.cast(gamma, reshaped_inputs.dtype)
            inv = inv * gamma

        res = -mean * inv
        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
            beta = tf.cast(beta, reshaped_inputs.dtype)
            res = res + beta

        normalized_inputs = reshaped_inputs * inv + res
        return normalized_inputs

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

class SinusoidalTimeEmbedding(tf.keras.layers.Layer):
    """Transform an integer/float timestep t -> (B, embed_dim) sinusoidal vector"""
    def __init__(self, embed_dim=64, **kw):
        super().__init__(**kw)
        self.embed_dim = embed_dim

    def call(self, t):
        half = self.embed_dim // 2
        freq = tf.exp(
            tf.range(half, dtype=tf.float32) * -(tf.math.log(10000.0) / half)
        )
        args = tf.expand_dims(tf.cast(t, tf.float32), -1) * freq
        emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        return emb
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim
        })
        return config

class SigmaEmbedding(tf.keras.layers.Layer):
    """ map a scalar sigma -> (B, embed_dim) vector"""
    def __init__(self, embed_dim=256, **kw):
        super().__init__(**kw)
        self.up = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_dim, activation="gelu"),
            tf.keras.layers.Dense(embed_dim, activation="gelu"),
        ])

    def call(self, sigma):
        log_sigma = tf.math.log(sigma)
        #return self.up(tf.expand_dims(log_sigma, -1))
        return self.up(log_sigma)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim
        })
        return config

class FiLMResidual(tf.keras.layers.Layer):
    """Conv->GN->GELU->FiLM->Conv->GN + skip"""
    def __init__(self, n_filters, **kw):
        super().__init__(**kw)
        self.n = n_filters
        self.conv1 = tf.keras.layers.Conv2D(n_filters, 3, padding="same")
        self.gn1   = GroupNormalization(groups=8) # batch norm
        self.conv2 = tf.keras.layers.Conv2D(n_filters, 3, padding="same")
        self.gn2   = GroupNormalization(groups=8)
        self.act   = tf.keras.layers.Activation("gelu")
        
        self.skip_proj = None

        self.scale = tf.keras.layers.Dense(n_filters)
        self.shift = tf.keras.layers.Dense(n_filters)

    def build(self, input_shape):
        in_ch = input_shape[-1]
        if in_ch != self.n:
            self.skip_proj = tf.keras.layers.Conv2D(self.n, 1, padding="same", name="conv2d_skip_proj")

    def call(self, x, temb):
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h)

        # FiLM mod
        s = self.scale(temb)[:, None, None, :] # (B,1,1,C)
        b = self.shift(temb)[:, None, None, :] # (B,1,1,C)
        h = h * (1.0 + s) + b

        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act(h)

        if self.skip_proj is not None:
            x = self.skip_proj(x)

        return x + h
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_filters": self.n
        })
        return config

def downsample(n_filters):
    return tf.keras.layers.Conv2D(n_filters, 4, strides=2, padding="same")

def upsample(n_filters):
    return tf.keras.layers.Conv2DTranspose(n_filters, 4, strides=2, padding="same")

class BicubicUpSampling2D_dm(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super(BicubicUpSampling2D_dm, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs):
        return tf.image.resize(inputs, [int(inputs.shape[1] * self.size[0]), int(inputs.shape[2] * self.size[1])],
                               method=tf.image.ResizeMethod.BILINEAR)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size
        })
        return config

def upsample_v2(x, target_size):
    x_resized = BicubicUpSampling2D_dm((target_size, target_size))(x)
    return x_resized

def res_block_initial(x, num_filters, kernel_size, strides, name, sym_padding=False):
    if len(num_filters) == 1:
        num_filters = [num_filters[0], num_filters[0]]

    x1 = tf.keras.layers.Conv2D(filters=num_filters[0],
                                kernel_size=kernel_size,
                                strides=strides[0],
                                padding='same',
                                name=name + '_1', kernel_initializer ='he_normal')(x)

    x1 = tf.keras.layers.LeakyReLU(0.01)(x1)

    x1 = tf.keras.layers.Conv2D(filters=num_filters[1],
                                kernel_size=kernel_size,
                                strides=strides[1],
                                padding='same',
                                name=name + '_2', kernel_initializer ='he_normal')(x1)

    x = tf.keras.layers.Conv2D(filters=num_filters[-1],
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               name=name + '_shortcut', kernel_initializer ='he_normal')(x)
    x1 = tf.keras.layers.Add()([x, x1])
    x1 = tf.keras.layers.LeakyReLU(0.01)(x1)
    return x1

def down_block(x, filters, kernel_size, i =1, use_pool=True, method ='unet', sym_padding =False):

    x = res_block_initial(x, [filters], kernel_size, strides=[1, 1],
                          name='dm_decoder_layer_v2' + str(i),
                              sym_padding = sym_padding)
    if use_pool == True:
        return tf.keras.layers.AveragePooling2D(strides=(2, 2))(x), x
    else:
        return x


def up_block(x, y, filters, kernel_size, i =1, method ='unet', concat = True, sym_padding =False):
    x = upsample_v2(x, 2)
    if concat:
        x = tf.keras.layers.Concatenate(axis=-1)([x, y])
    x = res_block_initial(x, [filters], kernel_size, strides=[1, 1],
                          name='dm_encoder_layer_v2' + str(i),sym_padding = sym_padding)
    return x

def get_custom_dm_objects():
    return {
        'GroupNormalization': GroupNormalization,
        'SinusoidalTimeEmbedding': SinusoidalTimeEmbedding,
        'SigmaEmbedding': SigmaEmbedding,
        'FiLMResidual': FiLMResidual,
        'BicubicUpSampling2D_dm': BicubicUpSampling2D_dm
    }


def build_diffusion_unet(
        input_size,
        resize_output,
        num_filters,
        kernel_size,
        num_channels,
        num_classes,
        resize=True,
        final_activation=None,
        time_embed_dim=256,
        resize_output2=[172, 179]
    ) -> tf.keras.Model:

    # input
    inp_res_hr_noise   = tf.keras.Input(shape=[resize_output[0], resize_output[1], 1], name="y_res_hr_noise")
    inp_static_hr = tf.keras.Input(shape=[resize_output[0], resize_output[1], 1], name="static_hr")
    inp_lr     = tf.keras.Input(shape=[input_size[0], input_size[1], num_channels], name="x_lr")
    inp_t      = tf.keras.Input((), dtype=tf.int32, name="timestep")
    inp_mean_hr   = tf.keras.Input(shape=[resize_output[0], resize_output[1], 1], name="y_mean_hr")

    up_lr = tf.image.resize(inp_lr, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8], method="bilinear")
    #x0 = tf.concat([inp_res_hr_hoise, inp_static_hr, up_lr, inp_mean_hr], axis=-1)
    
    # TODO: shouldn't need to resize hr fields
    up_res_hr_noise = tf.image.resize(inp_res_hr_noise, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8], method="bilinear")
    up_static_hr = tf.image.resize(inp_static_hr, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8], method="bilinear")    
    up_mean_hr = tf.image.resize(inp_mean_hr, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8], method="bilinear")
    x0 = tf.concat([up_res_hr_noise, up_static_hr, up_lr, up_mean_hr], axis=-1)

    # timestep embedding
    temb = SinusoidalTimeEmbedding(embed_dim=64)(inp_t)
    temb = tf.keras.layers.Dense(time_embed_dim, activation="gelu")(temb)

    # encoder
    feats = []
    x = tf.keras.layers.Conv2D(num_filters[0], 3, padding="same")(x0)

    for mult in [1, 2, 4]:
        n = 32 * mult # num_filters
        x = FiLMResidual(n)(x, temb)
        feats.append(x)
        x = downsample(n)(x)

    # bottleneck
    x = FiLMResidual(32*4)(x, temb)

    # decoder
    for mult, skip in zip([4,2,1][::-1], feats[::-1]):
        n = 32 * mult # num_filters
        x = upsample(n)(x)
        x = tf.concat([x, skip], axis=-1)
        x = FiLMResidual(n)(x, temb)

    # output
    out = tf.image.resize(x, (resize_output[0], resize_output[1]),
                    method=tf.image.ResizeMethod.BILINEAR)
    out = tf.keras.layers.Conv2D(num_classes, 3, padding="same", name="y_residual")(out)

    return tf.keras.Model(
        inputs=[inp_res_hr_noise, inp_t, inp_lr, inp_static_hr, inp_mean_hr],
        outputs=out,
        name="diffusion_residual_unet",
    )

def build_diffusion_unet_v2(input_size, resize_output, num_filters, kernel_size, num_channels, num_classes, resize=True,
                          final_activation = tf.keras.layers.LeakyReLU(1),
                          time_embed_dim=256):
    inp_res_hr_noise = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1], name="y_res_hr_noise") # noise [+ highres residual] TODO: avoid resizing thi
    inp_static_hr = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1], name="static_hr") # orog_vector
    inp_mean_hr = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1], name="y_mean_hr") # init_prediction

    concat_image = tf.keras.layers.Concatenate(-1)([inp_res_hr_noise, inp_static_hr,  inp_mean_hr])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)
    
    inp_t = tf.keras.Input((), dtype=tf.float32, name="timestep")
    # inp_t = tf.keras.Input((), dtype=tf.int32, name="timestep")
    temb = SinusoidalTimeEmbedding(embed_dim=64, name="time_embed_1")(inp_t)
    temb = tf.keras.layers.Dense(time_embed_dim, activation="gelu", name="time_embed_2")(temb)

    inp_lr = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels], name="x_lr") # average
    # noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
    # inputs_abstract = tf.keras.layers.Concatenate(-1)([inp_lr, noise])
    inputs_abstract = inp_lr
    x = concatted_highres
    x = FiLMResidual(num_filters[0], name="FiLM_0")(x, temb)
    #x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=3, i =0, sym_padding=False)
    x, temp1 = down_block(x, num_filters[0], kernel_size=3, i =0, sym_padding=False)
    x = FiLMResidual(num_filters[1], name="FiLM_1")(x, temb)
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1, sym_padding=False)
    x = FiLMResidual(num_filters[2], name="FiLM_2")(x, temb)
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2, sym_padding=False)
    x1 = down_block(inputs_abstract, num_filters[0], kernel_size=3, i=4, use_pool=False, sym_padding=False)
    x1 = down_block(x1, num_filters[1], kernel_size=3, i=5, use_pool=False, sym_padding=False)
    #x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    x1 = res_block_initial(x1, [num_filters[2]], 3, [1, 1], f"noise_blockcccc", sym_padding=True)
    x1 = res_block_initial(x1, [num_filters[2]], 3, [1, 1], f"noise_blockcccec", sym_padding=True)
    x = tf.keras.layers.Concatenate(-1)([x1, x])
    x = FiLMResidual(num_filters[3], name="FiLM_3")(x, temb)
    x = res_block_initial(x, [num_filters[3]], 3, [1, 1], f"noise_blockceerere", sym_padding=True)
    x = res_block_initial(x, [num_filters[3]], 5, [1, 1], f"noise_blockderere", sym_padding=True)
    x = FiLMResidual(num_filters[3], name="FiLM_4")(x, temb)
    x = res_block_initial(x, [num_filters[3]*2], 3, [1, 1], f"noise_blockererere", sym_padding=True)
    
    # decode
    x = up_block(x, temp3, kernel_size=3, filters = num_filters[3], i =0, sym_padding=False)
    # noise2 = tf.keras.layers.Input(shape=[x.shape[1], x.shape[2], int(num_channels//2)])
    # x = tf.keras.layers.Concatenate(-1)([noise2, x])
    x = FiLMResidual(num_filters[3], name="FiLM_5")(x, temb)
    x = up_block(x, temp2, kernel_size=5, filters = num_filters[2], i =2, sym_padding=False)
    x = FiLMResidual(num_filters[2], name="FiLM_6")(x, temb)
    x = up_block(x, temp1, kernel_size=3, filters = num_filters[1], i =3, sym_padding=False)
    x = FiLMResidual(num_filters[1], name="FiLM_7")(x, temb)
    output = tf.image.resize(x, (resize_output[0], resize_output[1]),
                    method=tf.image.ResizeMethod.BILINEAR)
    output = res_block_initial(output, [num_filters[1]], 3, [1, 1], f"chur", sym_padding=True)
    output = res_block_initial(output, [num_filters[1]], 3, [1, 1], "output_convbbb123456", sym_padding=False)
    x = FiLMResidual(num_filters[1], name="FiLM_8")(x, temb)
    output = res_block_initial(output, [num_filters[0]], 5, [1, 1], "output_convbbb1234", sym_padding=False)
    output = tf.keras.layers.Conv2D(32, 3, activation=final_activation, padding ='same')(output)
    output = tf.keras.layers.Conv2D(16, 3, activation=final_activation, padding ='same')(output)
    output = tf.keras.layers.Conv2D(num_classes, 3, activation=final_activation, padding ='same')(output)
    input_layers = [inp_res_hr_noise, inp_t] + [inp_lr, inp_static_hr, inp_mean_hr]
    
    model = tf.keras.models.Model(input_layers, output, name='diffusion_model')
    model.summary()
    return model

def build_edm_unet( #TODO:
        input_size,
        resize_output,
        num_filters,
        kernel_size,
        num_channels,
        num_classes,
        resize=True,
        final_activation=None,
        embed_dim=256,
        resize_output2=[172, 179]
    ) -> tf.keras.Model:

    inp_res   = tf.keras.Input(shape=[resize_output[0], resize_output[1], 1], name="noise_mean_y_hr")
    inp_static = tf.keras.Input(shape=[resize_output[0], resize_output[1], 1], name="static_hr")
    inp_lr     = tf.keras.Input(shape=[input_size[0], input_size[1], num_channels], name="x_lr")
    inp_sigma      = tf.keras.Input(shape=(1,1,1), name="sigma")
    inp_mean2   = tf.keras.Input(shape=[resize_output[0], resize_output[1], 1], name="mean_y_hr")

    res_up = tf.image.resize(inp_res, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8], method="bilinear")
    static_up =tf.image.resize(inp_static, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8], method="bilinear")
    mean2_up = tf.image.resize(inp_mean2, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8], method="bilinear")
    lr_up   = tf.image.resize(inp_lr, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8], method="bilinear")         # (B,128,128,C)
    #x0      = tf.concat([inp_res, inp_static, lr_up, inp_mean2], axis=-1)
    x0      = tf.concat([res_up, static_up, lr_up, mean2_up], axis=-1)

    # EDM preconditioning
    sigma2 = tf.square(inp_sigma)
    sd2 = 0.5**2
    cin = tf.math.rsqrt(sigma2 + sd2)
    cskip = sd2 / (sigma2 + sd2)
    cout = inp_sigma * 0.5 * cin

    x0_s = x0 * cin

    print(f"SHAPE ==================")
    print(f"{x0_s.shape=}")
    print(f"SHAPE ==================")
    print(f"{inp_sigma.shape=}")

    temb = SigmaEmbedding(embed_dim)(inp_sigma)

    print(f"SHAPE ==================")
    print(f"{temb.shape=}")

    feats = []
    x = tf.keras.layers.Conv2D(num_filters[0], 3, padding="same")(x0_s)

    for mult in [1, 2, 4]:
        n = 32 * mult # num_filters
        x = FiLMResidual(n)(x, temb)
        feats.append(x)
        x = downsample(n)(x)

    x = FiLMResidual(32*4)(x, temb)

    for mult, skip in zip([4,2,1][::-1], feats[::-1]):
        n = 32 * mult # num_filters
        x = upsample(n)(x)
        x = tf.concat([x, skip], axis=-1)
        x = FiLMResidual(n)(x, temb)

    out = tf.image.resize(x, (resize_output[0], resize_output[1]),
                    method=tf.image.ResizeMethod.BILINEAR)
    out = tf.keras.layers.Conv2D(num_classes, 3, padding="same", name="y_residual")(out)

    res_up_fix = tf.image.resize(res_up, (resize_output[0], resize_output[1]),
                    method=tf.image.ResizeMethod.BILINEAR)

    out = out * cout + res_up_fix * cskip

    return tf.keras.Model(
        inputs=[inp_res, inp_sigma, inp_lr, inp_static, inp_mean2],
        outputs=out,
        name="edm_residual_unet",
    )
