import logging
import tensorflow as tf
import numpy as np
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

class PredictionCallbackDiffusion(tf.keras.callbacks.Callback):
    def __init__(self, unet, diffusion, ema_diffusion, scheduler, model_object, x_input, y_input, save_dir=None, batch_size =30, orog =None,
                                             output_mean =None, output_std = None, varname = "pr"):
        """
        Args:
            model: The trained model.
            sample_input: A sample input tensor to generate predictions.
            save_dir: Directory where prediction images will be saved.
        """
        super(PredictionCallbackDiffusion, self).__init__()
        self.unet = unet
        self.diffusion = diffusion
        self.ema_diffusion = ema_diffusion
        self.scheduler = scheduler
        self.model_object = model_object
        self.x_input = x_input
        self.y_input = y_input
        self.x_input_tensor = tf.convert_to_tensor(x_input[0].values[:batch_size])
        self.y_input_tensor = tf.convert_to_tensor(y_input)
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.orog = orog
        self.orog_tensor = tf.convert_to_tensor(orog)
        self.output_mean = output_mean
        self.output_std = output_std
        self.varname = varname

        
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """Runs at the end of each epoch to save a prediction image."""
        # Generate prediction
        if epoch % 3 == 0:
            tf.random.set_seed(16)
            orog_vector = self.model_object.expand_conditional_inputs(self.orog_tensor, self.batch_size)
            #average_combined, orog_vector,time_of_year_combined, spatial_means_combined,
            #                 spatial_stds_combined
            unet_prediction = self.unet.predict([
                                          self.x_input_tensor, orog_vector[:self.batch_size]], verbose=0)
            # TODO: list:
            #   FIXME: stop rescaling noise in diffusion model
            if isinstance(self.scheduler, DiffusionSchedule):
                tf.random.set_seed(16)
                residual_pred = tf.random.normal(shape=(self.batch_size, 172, 179, 1)) #FIXME: get correct shape
                for t in reversed(range(self.scheduler.timesteps)):
                    t_tensor = tf.fill((self.batch_size, 1), t)

                    eps_theta = self.ema_diffusion.predict([residual_pred, t_tensor, self.x_input_tensor, orog_vector, unet_prediction], verbose=0)
                    
                    beta_t = self.scheduler.beta[t]
                    alpha_t = self.scheduler.alpha[t]
                    alpha_bar_t = self.scheduler.alpha_bar[t]
   
                    residual_pred = 1.0 / tf.sqrt(alpha_t) * (residual_pred - (beta_t / tf.sqrt(1 - alpha_bar_t)) * eps_theta)

                    if t > 0:
                        eps = tf.random.normal(shape=(self.batch_size, 172, 179, 1)) # FIXME: get correct shape
                        residual_pred += tf.sqrt(beta_t) * eps

            elif isinstance(self.scheduler, EDMSchedule):
                """
                ode integration via 2nd order heun

                x0 ~ N(0, sigma_max^2 * I)

                dx/dsigma = (x - D(x; sigma)) / sigma
                """
                tf.random.set_seed(16)
                sigmas = self.scheduler.sigmas
                residual_pred = sigmas[0] * tf.random.normal(shape=(self.batch_size, 172, 179, 1)) #FIXME: get correct shape
                for i in range(self.scheduler.timesteps - 1):
                    sigma = sigmas[i]
                    sigma_next = sigmas[i + 1]
                    sigma_delta = sigma_next - sigma
                    sigma_tensor = tf.fill((self.batch_size, 1), sigma)

                    pred_noise = self.ema_diffusion.predict([residual_pred, sigma_tensor, self.x_input[0], orog_vector, unet_prediction], verbose=0)
                    d = (residual_pred - pred_noise) / sigma

                    residual_intermediate = residual_pred + sigma_delta * d
                    sigma_next_tensor = tf.fill((self.batch_size, 1), sigma_next)

                    pred_noise_next = self.ema_diffusion.predict([residual_intermediate, sigma_next_tensor, self.x_input[0], orog_vector, unet_prediction], verbose=0)
                    d_prime = (residual_intermediate - pred_noise_next) / sigma_next

                    residual_pred = residual_pred + 0.5 * sigma_delta * (d + d_prime)


            if self.varname == "pr":

                unet_final = tf.math.exp(unet_prediction) -1
                gan_final = tf.math.exp(unet_prediction + residual_pred)-1
            else:
                unet_final = tf.squeeze(unet_prediction) * self.output_std + self.output_mean
                gan_final = tf.squeeze(unet_prediction + residual_pred) * self.output_std + self.output_mean

            y_copy = self.y_input.copy()
            y_2 = self.y_input.copy()
            y_copy.values = tf.squeeze(unet_final)
            y_2.values = tf.squeeze(gan_final)
            y_2 = y_2.where(y_2>0.5, 0.0)
            y_copy = y_copy.where(y_2>0.5, 0.0)
            boundaries2 = [0, 5,12.5, 15, 20, 25,30, 35, 40, 50, 60, 70, 80, 100, 125, 150, 200, 250]
            colors2 = [[0.000, 0.000, 0.000, 0.000], [0.875, 0.875, 0.875, 0.784],\
                      [0.761, 0.761, 0.761, 1.000], [0.639, 0.886, 0.871, 1.000], [0.388, 0.773, 0.616, 1.000],\
                      [0.000, 0.392, 0.392, 0.588], [0.000, 0.576, 0.576, 0.667], [0.000, 0.792, 0.792, 0.745],\
                      [0.000, 0.855, 0.855, 0.863], [0.212, 1.000, 1.000, 1.000], [0.953, 0.855, 0.992, 1.000],\
                      [0.918, 0.765, 0.992, 1.000], [0.918, 0.612, 1.000, 1.000], [0.878, 0.431, 1.000, 1.000],\
                      [0.886, 0.349, 1.000, 1.000], [0.651, 0.004, 0.788, 1.000], [0.357, 0.008, 0.431, 1.000],\
                      [0.180, 0.000, 0.224, 1.000]]
            #reviated for clarity

            # Create the colormap using ListedColormap
            cmap = mcolors.ListedColormap(colors2)
            norm = mcolors.BoundaryNorm(boundaries2, cmap.N)

            for i in range(8):
                print(i)
                fig, ax = plt.subplots(1, 3, figsize = (16, 6), subplot_kw = dict(projection = ccrs.PlateCarree(central_longitude =171.77)))
                if self.varname =="pr":
                    y_copy.isel(time =i).plot.contourf(ax = ax[0], transform = ccrs.PlateCarree(), cmap = cmap, norm = norm)
                    y_2.isel(time =i).plot.contourf(ax = ax[1], transform = ccrs.PlateCarree(), cmap = cmap, norm = norm)
                    (np.exp(self.y_input.isel(time =i))-1) .plot.contourf(ax = ax[2], transform = ccrs.PlateCarree(), cmap = cmap, norm = norm)
                else:
                    true = self.y_input.isel(time =i) * self.output_std + self.output_mean
                    min_t = true.values.min()
                    max_t = true.values.max()
                    levels = np.arange(min_t, max_t, 0.5)
                    y_copy.isel(time =i).plot(ax = ax[0], transform = ccrs.PlateCarree(), cmap ='RdBu_r', levels = levels)
                    y_2.isel(time =i).plot(ax = ax[1], transform = ccrs.PlateCarree(), cmap ='RdBu_r', levels = levels)
                    true.plot.contourf(ax = ax[2], transform = ccrs.PlateCarree(), cmap ='RdBu_r', levels = levels)
                ax[0].coastlines('10m')
                ax[1].coastlines('10m')
                ax[2].coastlines('10m')
                ax[0].set_title('Unet')
                ax[1].set_title('DM')
                ax[2].set_title('GT')
                # Save the figure
                filename = os.path.join(self.save_dir, f"epoch_{epoch+1}_{i}.png")
                plt.savefig(filename, bbox_inches="tight", dpi =200)
                plt.close()

            print(f"Saved prediction image to {filename}")

class DiffusionSchedule: # Linear schedule
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        beta = np.linspace(beta_start, beta_end, timesteps)
        self.beta = tf.constant(beta, dtype=tf.float32)
        self.alpha = 1 - self.beta
        self.alpha_bar = tf.math.cumprod(self.alpha)
        self.sqrt_alpha_bar = tf.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = tf.sqrt(1 - self.alpha_bar)

# karras sigma rescaled:
# - for 100 steps: 0.01, 1.339
# - for 1000 steps: 0.01, 169.755
# EDM/CorrDiff found: timesteps=18, sigma_min=0.002, sigma_max=800.0, rho=7.0
class KarrasSigmaSchedule:
    def __init__(self, timesteps=18, sigma_min=0.002, sigma_max=800.0, rho=7.0):
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        # sigmas = []
        # for i in range(timesteps):
        #     fraction = i / (timesteps - 1)
        #     sigma_i = (sigma_max ** (1/rho) + fraction * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        #     sigmas.append(sigma_i)
        i = tf.range(timesteps, dtype=tf.float32)
        r = (sigma_max ** (1/rho) + i/(timesteps-1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        self.sigmas = tf.concat([r, tf.zeros_like(r[:1])], 0) # append sigma_N = 0

    def __len__(self):
        return self.sigmas.shape[0] - 1

class EDMSchedule:
    #def __init__(self, timesteps=100, sigma_min=0.02, sigma_max=50.0, rho=7.0):
    def __init__(self, timesteps=18, sigma_min=0.002, sigma_max=800.0, rho=7.0):
        """        
        For i = 0,..., timesteps - 1
            sigma_i = ( sigma_max^(1/rho) + (i/(timesteps-1)) * (sigma_min^(1/rho) - sigma_max^(1/rho)) )^rho
        """
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        sigmas = []
        for i in range(timesteps):
            fraction = i / (timesteps - 1)
            sigma_i = (sigma_max ** (1/rho) + fraction * (sigma_min ** (1/rho) - sigma_max **( 1/rho))) ** rho
            sigmas.append(sigma_i)
        self.sigmas = tf.constant(sigmas, dtype=tf.float32)


class ResidualDiffusion(tf.keras.Model):
    def __init__(self, diffusion=None,
                 ema_diffusion=None,
                 scheduler=None,
                 gp_weight=10.0,
                 orog=None, he=None,
                 vegt=None, unet=None, train_unet=True, intensity_weight=1.0,
                 average_intensity_weight=0.0, varname="pr",
                 ema_decay=None,
                 use_gan_loss_constraints=None):
        super(ResidualDiffusion, self).__init__()

        self.diffusion = diffusion
        self.ema_diffusion = ema_diffusion
        self.scheduler = scheduler
        self.gp_weight = gp_weight
        self.orog = orog
        self.he = he
        self.vegt = vegt
        self.unet = unet
        self.train_unet = train_unet
        self.intensity_weight = intensity_weight
        self.average_itensity_weight = average_intensity_weight
        self.varname = varname
        self.ema_decay = ema_decay
        self.use_gan_loss_constraints = use_gan_loss_constraints

        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.u_loss_tracker = tf.keras.metrics.Mean(name="unet_loss")

        if self.use_gan_loss_constraints:
            self.gan_mae_tracker = tf.keras.metrics.Mean(name="gan_mae")
            self.max_iten_pred_tracker = tf.keras.metrics.Mean(name="max_iten_pred")
            self.max_iten_true_tracker = tf.keras.metrics.Mean(name="max_iten_true")

    def compile(self, dm_optimizer, u_optimizer, u_loss_fn):
        super(ResidualDiffusion, self).compile()
        self.dm_optimizer = dm_optimizer
        self.u_optimizer = u_optimizer
        self.u_loss_fn = u_loss_fn
        

    # def gradient_penalty(self, batch_size, real_images, fake_images, average, orog_vector, he_vector, vegt_vector,
    #                      unet_preds):
    #     """
    #     need to modify
    #     """
    #     # Get the interpolated image
    #     alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    #     diff = fake_images - real_images
    #     interpolated = real_images + alpha * diff

    #     with tf.GradientTape() as gp_tape:
    #         gp_tape.watch(interpolated)
    #         pred = self.discriminator([interpolated, average, orog_vector, unet_preds],
    #                                   training=True)

    #     grads = gp_tape.gradient(pred, [interpolated])[0]
    #     norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    #     gp = tf.reduce_mean((norm - 1.0) ** 2)
    #     return gp

    @staticmethod
    def expand_conditional_inputs(X, batch_size):
        expanded_image = tf.expand_dims(X, axis=0)  # Shape: (1, 172, 179)

        # Repeat the image to match the desired batch size
        expanded_image = tf.repeat(expanded_image, repeats=batch_size, axis=0)  # Shape: (batch_size, 172, 179)

        # Create a new axis (1) on the last axis
        expanded_image = tf.expand_dims(expanded_image, axis=-1)
        return expanded_image

    @staticmethod
    def process_real_images(real_images_obj):
        output_vars, averages = real_images_obj  # Unpack the input

        # Extract relevant variables from the output_vars dictionary
        real_images = [
            output_vars['pr']
        ]

        real_images_future = [
            output_vars['pr_future'],
            
        ]

        # Extract average and average_future
        average = averages["X"]
        average_future = averages["X_future"]

        # Combine variables into single tensors
        real_images = tf.concat([tf.expand_dims(img, axis=-1) for img in real_images], axis=-1)
        real_images_future = tf.concat([tf.expand_dims(img, axis=-1) for img in real_images_future], axis=-1)

        # Combine all GCMs into one single batch timestep
        real_images = tf.concat([real_images[:, :, :, i, :] for i in range(real_images.shape[3])], axis=0)
        real_images_future = tf.concat([real_images_future[:, :, :, i, :] for i in range(real_images_future.shape[3])],
                                       axis=0)
        average = tf.concat([average[:, :, :, i, :] for i in range(average.shape[3])], axis=0)
        average_future = tf.concat([average_future[:, :, :, i, :] for i in range(average_future.shape[3])], axis=0)
        
        average_combined = tf.concat([average, average_future], axis =0)
        real_images_combined = tf.concat([real_images, real_images_future], axis =0)
        return real_images_combined, average_combined

    def train_step(self, real_images):
        real_images, average = self.process_real_images(real_images)

            # here the average represents the conditional input

        batch_size = tf.shape(real_images)[0]
        orog_vector = self.expand_conditional_inputs(self.orog, batch_size)
        he_vector = self.expand_conditional_inputs(self.he, batch_size)
        vegt_vector = self.expand_conditional_inputs(self.vegt, batch_size)
        # make sure the auxiliary inputs are the same shape as the training batch
        # if the U-Net is trained, apply gradients otherwise only use inference mode from the U-Net
        if self.train_unet:
            with tf.GradientTape() as tape:
                init_prediction = self.unet([average,
                                             orog_vector], training=True)
                orog_vec_mask = tf.cast(tf.squeeze(orog_vector) > 0.001, 'float32')
                loss_rain_ocean_unet = self.u_loss_fn(tf.squeeze((real_images[:, :, :, 0:1])) * (1 - orog_vec_mask), tf.squeeze(init_prediction) * (1 - orog_vec_mask))

                loss_rain_land_unet = self.u_loss_fn(tf.squeeze((real_images[:, :, :, 0:1])) * orog_vec_mask, tf.squeeze(init_prediction) * orog_vec_mask)

                loss_rain_unet = (5* loss_rain_land_unet + loss_rain_ocean_unet)/6.0
                mae_unet = loss_rain_unet#self.u_loss_fn(real_images[:, :, :], init_prediction)
            u_gradient = tape.gradient(mae_unet, self.unet.trainable_variables)

            self.u_optimizer.apply_gradients(zip(u_gradient, self.unet.trainable_variables))
        else:
            with tf.GradientTape() as tape:
                init_prediction = self.unet([ average,
                                             orog_vector], training=True)
                mae_unet = self.u_loss_fn(real_images[:, :, :], init_prediction)

        # Diffusion model
        noise = tf.random.normal(shape=tf.shape(real_images))

        if isinstance(self.scheduler, DiffusionSchedule):
            t = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=self.scheduler.timesteps, dtype=tf.int32)
            sqrt_alpha_bar_t = tf.gather(self.scheduler.sqrt_alpha_bar, t)
            sqrt_alpha_bar_t = tf.reshape(sqrt_alpha_bar_t, (batch_size, 1, 1, 1))
            sqrt_one_minus_alpha_bar_t = tf.gather(self.scheduler.sqrt_one_minus_alpha_bar, t)
            sqrt_one_minus_alpha_bar_t = tf.reshape(sqrt_one_minus_alpha_bar_t, (batch_size, 1, 1, 1))

        elif isinstance(self.scheduler, EDMSchedule):
            log_sigma_min = tf.math.log(self.scheduler.sigma_min)
            log_sigma_max = tf.math.log(self.scheduler.sigma_max)
            random_log_sigma = log_sigma_min + tf.random.uniform((batch_size, 1, 1, 1), minval=0.0, maxval=1.0) * (log_sigma_max - log_sigma_min)
            sigma = tf.exp(random_log_sigma)

        with tf.GradientTape() as tape:
            init_prediction_unet = self.unet([average,
                                              orog_vector], training=True)
            residual_gt = (real_images - init_prediction_unet)

            if isinstance(self.scheduler, DiffusionSchedule):
                residual_noisy = sqrt_alpha_bar_t * residual_gt + sqrt_one_minus_alpha_bar_t * noise
                noise_pred = self.diffusion([residual_noisy, t, average, orog_vector, init_prediction_unet], training=True)
                
                # reconstruct residual_gt (x0) using predicted noise (eps_theta(xt))
                # eq. 15 from Ho et al 2020
                residual_pred = (residual_noisy - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t # TODO: remove constraint

                # or try see if this works better: need to grab the other params from scheduler
                # residual_pred = 1.0 / tf.sqrt(alpha_t) * (residual_noisy - (beta_t / tf.sqrt(1 - alpha_bar_t)) * noise_pred)

            elif isinstance(self.scheduler, EDMSchedule):
                residual_noisy = residual_gt + sigma * noise
                noise_pred = self.diffusion([residual_noisy, sigma, average, orog_vector, init_prediction_unet], training=True)

            # print(f"{residual_noisy.shape=}")
            # print(f"{t.shape=}")
            # print(f"{average.shape=}")
            # print(f"{orog_vector.shape=}")
            # print(f"{init_prediction_unet.shape=}")
            
            if not self.use_gan_loss_constraints:
                dm_loss = self.u_loss_fn(noise, noise_pred)

            else:
                orog_vec_mask = tf.cast(tf.squeeze(orog_vector) > 0.001, 'float32')
                # loss_rain_ocean_dm = self.u_loss_fn(tf.squeeze((residual_gt)) * (1 - orog_vec_mask), tf.squeeze(residual_pred) * (1 - orog_vec_mask))
                # loss_rain_land_dm = self.u_loss_fn(tf.squeeze((residual_gt)) * orog_vec_mask, tf.squeeze(residual_pred) * orog_vec_mask)
                loss_rain_ocean_dm = self.u_loss_fn(tf.squeeze((noise)) * (1 - orog_vec_mask), tf.squeeze(noise_pred) * (1 - orog_vec_mask))
                loss_rain_land_dm = self.u_loss_fn(tf.squeeze((noise)) * orog_vec_mask, tf.squeeze(noise_pred) * orog_vec_mask)
            

                if isinstance(self.scheduler, EDMSchedule):
                    cout = sigma * 0.5 * tf.math.rsqrt(tf.square(sigma) + 0.5**2)
                    w = 1.0 / tf.square(cout)
                    w = w[:, None, None, None]
                    loss_rain_ocean_dm = w * loss_rain_ocean_dm
                    loss_rain_land_dm  = w * loss_rain_land_dm
                    # loss_rain_ocean_dm = loss_rain_ocean_dm / (sigma**2 + 1e-7)
                    # loss_rain_land_dm  = loss_rain_land_dm  / (sigma**2 + 1e-7)

                mae = (8* loss_rain_land_dm + loss_rain_ocean_dm)/9.0

                prediction = residual_pred + init_prediction_unet
                if self.varname == "pr":
                    maximum_intensity = tf.math.reduce_max(
                        real_images, axis=[-1, -2, -3])
                    maximum_intensity_predicted = tf.math.reduce_max(prediction,
                                                                    axis=[-1, -2, -3])
                    maximum_intensity_error = tf.reduce_mean(
                    tf.abs(maximum_intensity - maximum_intensity_predicted) ** 2)
                    average_intensity = tf.math.reduce_mean(
                    real_images, axis=[-1, -4])
                    average_intensity_predicted = tf.math.reduce_mean(prediction,
                                                                axis=[-1, -4])

                    average_intensity_error = tf.reduce_mean(
                tf.abs(average_intensity - average_intensity_predicted) ** 2)
                else:
                    maximum_intensity = tf.math.reduce_max(
                        real_images, axis=[-1, -2, -3])
                    maximum_intensity_predicted = tf.math.reduce_max(prediction,
                                                                    axis=[-1, -2, -3])
                    maximum_intensity_error = tf.reduce_mean(
                    tf.abs(maximum_intensity - maximum_intensity_predicted) ** 2)
                    minimum_intensity = tf.math.reduce_min(
                        real_images, axis=[-1, -2, -3])
                    minimum_intensity_predicted = tf.math.reduce_min(prediction,
                                                                    axis=[-1, -2, -3])
                    minimum_intensity_error = tf.reduce_mean(
                    tf.abs(minimum_intensity - minimum_intensity_predicted) ** 2)
                    maximum_intensity_error = 1/2 * (minimum_intensity_error + maximum_intensity_error)
                    average_intensity = tf.math.reduce_mean(
                    real_images, axis=[-1, -4])
                    average_intensity_predicted = tf.math.reduce_mean(prediction,
                                                                axis=[-1, -4])

                    average_intensity_error = tf.reduce_mean(
                tf.abs(average_intensity - average_intensity_predicted) ** 2)
                    
                dm_loss = mae + self.average_itensity_weight * average_intensity_error + self.intensity_weight * maximum_intensity_error

        
        dm_gradient = tape.gradient(dm_loss, self.diffusion.trainable_variables)
        self.dm_optimizer.apply_gradients(zip(dm_gradient, self.diffusion.trainable_variables))

        for weight, ema_weight in zip(self.diffusion.weights, self.ema_diffusion.weights):
            ema_weight.assign(self.ema_decay * ema_weight + (1 - self.ema_decay) * weight)


        self.g_loss_tracker.update_state(dm_loss)
        self.u_loss_tracker.update_state(mae_unet)

        if not self.use_gan_loss_constraints:
            return {
                "g_loss": self.g_loss_tracker.result(),
                "unet_loss": self.u_loss_tracker.result(),
            }

        self.gan_mae_tracker.update_state(mae)
        self.max_iten_pred_tracker.update_state(tf.math.exp(maximum_intensity_predicted))
        self.max_iten_true_tracker.update_state(tf.math.exp(maximum_intensity))

        return {
            "g_loss": self.g_loss_tracker.result(),
            "unet_loss": self.u_loss_tracker.result(),
            "gan_mae": self.gan_mae_tracker.result(),
            "max_iten_pred": self.max_iten_pred_tracker.result(),
            "max_iten_true": self.max_iten_true_tracker.result(),
        }


    @property
    def metrics(self):
        if not self.use_gan_loss_constraints:
            return [
            self.g_loss_tracker,
            self.u_loss_tracker,
        ]
        return [
            self.g_loss_tracker,
            self.u_loss_tracker,
            self.gan_mae_tracker,
            self.max_iten_pred_tracker,
            self.max_iten_true_tracker,
        ]
