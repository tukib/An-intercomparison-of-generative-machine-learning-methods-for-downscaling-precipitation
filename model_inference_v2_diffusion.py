import xarray as xr
import sys
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from tensorflow.keras.callbacks import Callback
import numpy as np
import tensorflow as tf
from functools import partial

AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar
import pandas as pd
import tensorflow.keras.layers as layers
import json
from tensorflow.keras.optimizers import Adam
from tensorflow._api.v2.distribute import MirroredStrategy
from tensorflow.keras import layers
import datetime
import tqdm
import time
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--dev", action="store_true", help="use default dev config instead of default hpc config")
argparser.add_argument("--shortperiod", action="store_true", help="use 2 year inference period")
argparser.add_argument("--config", type=str, help="target path (eval config file) to use", required=False)
argparser.add_argument("--gan", action="store_true", help="must be enabled if using a GAN model")
argparser.add_argument("target", type=str, help="target path (model/train config file) to use")

args = argparser.parse_args()

config_file = args.target
#config_file = sys.argv[-1]
with open(config_file, 'r') as f:
    config = json.load(f)

if args.config:
    config_file_for_test_data = args.config
elif args.dev:
    config_file_for_test_data = r'./model_inference/metadata_all_gcms_v3_diffusion_dev.json'
else:
    config_file_for_test_data = r'./model_inference/metadata_all_gcms_v3_diffusion.json'
# if sys.argv[-2] == '--dev':
#     config_file_for_test_data = r'./model_inference/metadata_all_gcms_v3_diffusion_dev.json'
with open(config_file_for_test_data) as f:
    config_test_data = json.load(f)

USING_GAN = True if args.gan else False

output_norm = 1

# the quantiles of which the climate change signal is computed over
quantiles = [ 0.5 , 0.7, 0.9, 0.925,
             0.95, 0.975, 0.98, 0.99,
             0.995, 0.998, 0.999, 0.9999]
# the periods of which the climate change signal / response is computed over
# historical_period = slice("1985","2014")
# future_period = slice("2070","2099")
# future_period2 = slice("2040","2069")
# historical_period = slice("2012","2014")
# future_period = slice("2097","2099")
# future_period2 = slice("2067","2069")

# 30 y period
historical_period = slice("1985","2014")
future_period = slice("2070","2099")

# 10 y period
# historical_period = slice("1995","2004")
# future_period = slice("2080","2089")

# 2 y period
if args.shortperiod:
# if sys.argv[-2] == '--dev':
    historical_period = slice("2003","2004")
    future_period = slice("2088","2089")

sys.path.append(os.getcwd())
from src.layers import *
from src.models import *
from src.gan import *
from src.process_input_training_data import *
from src.dm import ResidualDiffusion, DiffusionSchedule, KarrasSigmaSchedule, EDMSchedule, PredictionCallbackDiffusion
from src.models_dm import build_diffusion_unet, build_diffusion_unet_v2, build_edm_unet, get_custom_dm_objects

def load_model_dm(model_name, model_dir, epoch=None, gan=False):
    if epoch is None:
        epoch_suffix = 'final'
    else:
        epoch_suffix = f'epoch_{epoch}'

    dm_path = f'{model_dir}/{model_name}/ema_generator_{epoch_suffix}.h5'
    dm_path_alt = f'{model_dir}/{model_name}/generator_{epoch_suffix}.h5'
    unet_path = f'{model_dir}/{model_name}/unet_{epoch_suffix}.h5'

    custom_objects_unet = {"BicubicUpSampling2D": BicubicUpSampling2D,
                        "SymmetricPadding2D": SymmetricPadding2D}
    custom_objects_dm_or_gan = get_custom_dm_objects() if not gan else custom_objects_unet

    if os.path.exists(dm_path):
        dm = tf.keras.models.load_model(dm_path,
                                        custom_objects=custom_objects_dm_or_gan,
                                        compile=False)
    else:
        dm = tf.keras.models.load_model(dm_path_alt,
                                        custom_objects=custom_objects_dm_or_gan,
                                        compile=False)
    
    with open(f'{model_dir}/{model_name}/config_info.json') as f:
        config = json.load(f)

        unet = tf.keras.models.load_model(unet_path, custom_objects=custom_objects_unet
                                          , compile=False)

    return dm, unet, config["ad_loss_factor"]

def preprocess_input_data_eval(config: dict, match_index = True):
    vegt, orog, he = prepare_static_fields(config)
    means = xr.open_dataset(config["mean"])
    stds = xr.open_dataset(config["std"])

    X = xr.open_dataset(config["train_x"])  # .sel(time = slice("2016", None))
    X['time'] = pd.to_datetime(X.time.dt.strftime("%Y-%m-%d"))

    y = xr.open_dataset(config["train_y"])#, chunks={"time": 5000})
    y['time'] = pd.to_datetime(y.time.dt.strftime("%Y-%m-%d"))# .sel(time = slice("2016", None))

    # LOCAL CHANGE
    if config.get("gcms_for_eval") and config.get("output_varname"):
        gcms = config["gcms_for_eval"]
        y = y[[config["output_varname"]]]
        y = y.sel(GCM=gcms)
        X = X.sel(GCM=gcms)

    try:
        y = y.drop("lat_bnds")
        y = y.drop("lon_bnds")

    except:
        pass

    # preare the training data
    stacked_X, y = prepare_training_data(config, X, y, means, stds, match_index = match_index)

    return stacked_X, y, vegt, orog, he

def compute_quantiles(df,quantiles, period):
    df = df.sel(time = period)
    # this removes instances which have negative precipitation (os the minimum value is -0.0001) 
    # due to the leakyrelu activation function in GANs
    df = df.where(df>0.0, 0.0)
    seasonal_rainfall = df.groupby('time.season').mean()
    df = df.where(df>1, np.nan)
    quantiled_rain = df.quantile(q = quantiles, dim =["time"], skipna =True)
    return quantiled_rain, seasonal_rainfall


def compute_signal(df, quantiles, historical_period, future_period):

    historical_quantiles, seasonal_rainfall = compute_quantiles(df, quantiles, historical_period)
    future_quantiles, future_rainfall = compute_quantiles(df, quantiles, future_period)

    cc_signal = 100 * (future_rainfall - seasonal_rainfall)/seasonal_rainfall
    signal = 100 * (future_quantiles - historical_quantiles)/historical_quantiles
    historical_quantiles = historical_quantiles.rename({"pr":"hist_quantiles"})
    future_quantiles = future_quantiles.rename({"pr": "future_quantiles"})
    seasonal_rainfall = seasonal_rainfall.rename({"pr":"hist_clim_rainfall"})
    future_rainfall = future_rainfall.rename({"pr":"future_clim_rainfall"})
    signal = signal.rename({"pr":"cc_signal"})
    cc_signal = cc_signal.rename({"pr":"seas_cc_signal"})
    dset = xr.merge([historical_quantiles, future_quantiles,
                     signal, cc_signal, seasonal_rainfall, future_rainfall])
    return dset

def create_output(X, y):
    y = y.isel(time=0).drop("time")
    y = y.expand_dims({"time": X.time.size})
    y['time'] = (('time'), X.time.to_index())
    return y

def expand_conditional_inputs(X, batch_size):
    expanded_image = tf.expand_dims(X, axis=0)  # Shape: (1, 172, 179)

    # Repeat the image to match the desired batch size
    expanded_image = tf.repeat(expanded_image, repeats=batch_size, axis=0)  # Shape: (batch_size, 172, 179)

    # Create a new axis (1) on the last axis
    expanded_image = tf.expand_dims(expanded_image, axis=-1)
    return expanded_image

            # orog_vector = self.wgan.expand_conditional_inputs(self.orog, self.batch_size)
            # #average_combined, orog_vector,time_of_year_combined, spatial_means_combined,
            # #                 spatial_stds_combined
            # unet_prediction = self.unet.predict([
            #                               self.x_input[0].values[:self.batch_size], orog_vector[:self.batch_size]], verbose=0)

            # gan_prediction = self.generator.predict([random_latent_vectors,random_latent_vectors1,
            #                               self.x_input[0], orog_vector,unet_prediction])
            #
            # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

@tf.function
def predict_batch_residual(model, unet, latent_vectors, data_batch, orog, he, vegt, model_type, scheduler=None):
    if model_type == 'GAN':
        noise_dim = [tuple(model.inputs[i].shape[1:]) for i in range(len(model.inputs) - 1)]
        random_latent_vectors = tf.random.normal(
            shape=(data_batch.shape[0],) + noise_dim[0]
        )
        random_latent_vectors1 = tf.random.normal(
            shape=(data_batch.shape[0],) + noise_dim[1]
        )

        intermediate = unet([data_batch, orog], training=False)
        # intermediate = unet([latent_vectors[0], data_batch, orog, he, vegt], training=False)
        # intermediate = apply_gaussian_blur(intermediate, size=7, sigma=1.5)
        # max_value = tf.reduce_max(intermediate, axis=(1, 2, 3), keepdims=True)
        # min_value = tf.reduce_min(intermediate, axis=(1, 2, 3), keepdims=True)
        init_prediction = intermediate
        # print(intermediate)
        # intermediate = tf.cast(tf.math.sqrt(tf.clip_by_value(intermediate, clip_value_min=0, clip_value_max=2500)), 'float32')
        residual_pred = model([random_latent_vectors, random_latent_vectors1, data_batch, orog, init_prediction], training=False)
        # return model([latent_vectors[0], data_batch, orog, he, vegt, init_prediction],
        #              training=False) + intermediate  # +
        return intermediate + residual_pred
    if model_type == 'diffusion':
        intermediate = unet([data_batch, orog], training=False)

        if isinstance(scheduler, DiffusionSchedule):
            batch_size = data_batch.shape[0]
            residual_pred = tf.random.normal(shape=(batch_size, 172, 179, 1))
            for t in reversed(range(scheduler.timesteps)):
                t_tensor = tf.fill((batch_size, 1), t)

                eps_theta = model([residual_pred, t_tensor, data_batch, orog, intermediate], training=False)
                
                beta_t = scheduler.beta[t]
                alpha_t = scheduler.alpha[t]
                alpha_bar_t = scheduler.alpha_bar[t]

                residual_pred = 1.0 / tf.sqrt(alpha_t) * (residual_pred - (beta_t / tf.sqrt(1 - alpha_bar_t)) * eps_theta)

                if t > 0:
                    eps = tf.random.normal(shape=(batch_size, 172, 179, 1))
                    residual_pred += tf.sqrt(beta_t) * eps

            return intermediate + residual_pred
        elif isinstance(scheduler, KarrasSigmaSchedule):
            sigmas = scheduler.sigmas
            batch_size = data_batch.shape[0]
            residual_pred = tf.random.normal(shape=(batch_size, 172, 179, 1)) * sigmas[0]
            for i in range(len(scheduler)-1):
                sigma_i, sigma_j = sigmas[i], sigmas[i+1]
                sigma_delta = sigma_j - sigma_i

                # predictor (Euler)
                t_tensor_i = tf.fill((batch_size, 1), sigma_i / sigmas[0])
                eps_i = model([residual_pred, t_tensor_i, data_batch, orog, intermediate], training=False)
                # drift_i = (residual_pred - sigma_i * eps_i) / sigma_i
                # residual_pred_euler = residual_pred + sigma_delta * drift_i # (-eps_i)
                residual_pred_euler = residual_pred + sigma_delta * (-eps_i)
                
                # corrector (Heun)
                t_tensor_j = tf.fill((batch_size, 1), sigma_j / sigmas[0])
                eps_j = model([residual_pred_euler, t_tensor_j, data_batch, orog, intermediate], training=False)
                # drift_j = (residual_pred_euler - sigma_j * eps_j) / sigma_j if sigma_j > 0.0 else drift_i
                # residual_pred = residual_pred + sigma_delta * 0.5 * (drift_i + drift_j) # (-eps_i + -eps_j)
                residual_pred = residual_pred + sigma_delta * 0.5 * (-eps_i + -eps_j)

            return intermediate + residual_pred

        else:
            raise Exception("scheduler not defined")
    else:
        # return unet([latent_vectors[0], data_batch, orog, he, vegt], training=False)
        return unet([data_batch, orog], training=False)

def predict_parallel_resid(model, unet, inputs, output_shape, batch_size, orog_vector, he_vector, vegt_vector,
                           model_type='GAN', output_add_factor =1, scheduler=None):
    n_iterations = inputs.shape[0] // batch_size
    remainder = inputs.shape[0] - n_iterations * batch_size

    dset = []

    with tqdm.tqdm(total=n_iterations, desc="Predicting", unit="batch") as pbar:
        for i in range(n_iterations):
            data_batch = inputs[i * batch_size: (i + 1) * batch_size]
            random_latent_vectors1 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[0].shape[1:]))
            #random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
            orog = expand_conditional_inputs(orog_vector, batch_size)
            he = expand_conditional_inputs(he_vector, batch_size)
            vegt = expand_conditional_inputs(vegt_vector, batch_size)

            output = predict_batch_residual(model, unet, [random_latent_vectors1], data_batch, orog, he, vegt,
                                            model_type, scheduler=scheduler)

            dset += (np.exp(output.numpy()[:, :, :, 0]) - output_add_factor).tolist()
            pbar.update(1)  # Update the progress bar

    if remainder != 0:
        random_latent_vectors1 = tf.random.normal(shape=(batch_size,) + tuple(model.inputs[0].shape[1:]))
        #random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
        orog = expand_conditional_inputs(orog_vector, remainder)
        he = expand_conditional_inputs(he_vector, remainder)
        vegt = expand_conditional_inputs(vegt_vector, remainder)

        output = predict_batch_residual(model, unet, [random_latent_vectors1[:remainder]],
                                        inputs[inputs.shape[0] - remainder:], orog, he, vegt, model_type, scheduler=scheduler)

        dset += (np.exp(output.numpy()[:, :, :, 0]) - output_add_factor).tolist()
    output_shape['pr'].values = dset

    return output_shape

stacked_X, y, vegt, orog, he = preprocess_input_data_eval(config_test_data, match_index=False)

dm, unet, adv_factor = load_model_dm(
    config["model_name"],
    config_test_data["output_folder"],
    epoch=config_test_data.get("eval_epoch", None),
    gan=USING_GAN)

if USING_GAN:
    generator_model_type = 'GAN'
    scheduler = DiffusionSchedule()
else:
    generator_model_type = 'diffusion'
    if config.get("diffusion_type") == "EDM":
        scheduler = EDMSchedule()
    elif config.get("diffusion_type") == "karras_sigma":
        scheduler = KarrasSigmaSchedule(timesteps=config["dm_timesteps"], sigma_min=config["dm_sigma_min"], sigma_max=config["dm_sigma_max"], rho=config["dm_rho"])
    else:
        scheduler = DiffusionSchedule(timesteps=config["dm_timesteps"], beta_start=config["dm_beta_start"], beta_end=config["dm_beta_end"])

try:
    y = y.isel(GCM=0)[['pr']]
except:
    y =y[['pr']]

for gcm in stacked_X.GCM.values:
    print(f"preparing data for a GCM {gcm}")

    if not os.path.exists(f'./outputs/{config["model_name"]}'):
        os.makedirs(f'./outputs/{config["model_name"]}')

    with open(f'./outputs/{config["model_name"]}/config_info.json', 'w') as f:
        json.dump(config, f)

    output_shape = create_output(stacked_X, y)
    output_shape.pr.values = output_shape.pr.values * 0.0
    
    inputs_hist = stacked_X.sel( GCM =gcm).transpose('time','lat','lon','channel').sel(time = historical_period).values
    inputs_hist_len = len(inputs_hist)

    output_hist_path = f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_res_hist_bwar780.nc'
    if not os.path.exists(output_hist_path):
        timer_start = time.perf_counter()
        output_hist = xr.concat([predict_parallel_resid(dm, unet,
                                    inputs_hist,
                                    output_shape.sel(time = historical_period), 64, orog.values, he.values, vegt.values,
                                    model_type=generator_model_type, output_add_factor = output_norm, scheduler=scheduler) for i in range(1)],
                                dim ="member")
        timer_stop = time.perf_counter()
        timer_elapsed = timer_stop - timer_start
        print(f"Historical period - {generator_model_type} - elapsed: {timer_elapsed:.6f} s ({timer_elapsed/inputs_hist_len:.6f} s/it, {inputs_hist_len} it)")

        output_hist.to_netcdf(output_hist_path)
        del output_hist
    else:
        print(f"Historical period - {generator_model_type} - skipped (netcdf file already exists)")


    output_hist_unet_path = f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_unet_hist_bwar780.nc'
    if not os.path.exists(output_hist_unet_path):
        timer_start = time.perf_counter()
        output_hist_unet = xr.concat([predict_parallel_resid(dm, unet,
                                    inputs_hist,
                                    output_shape.sel(time = historical_period), 64, orog.values, he.values, vegt.values,
                                    model_type='UNET', output_add_factor = output_norm) for i in range(1)],
                                dim ="member")
        timer_stop = time.perf_counter()
        timer_elapsed = timer_stop - timer_start
        print(f"Historical period - unet only - elapsed: {timer_elapsed:.6f} s ({timer_elapsed/inputs_hist_len:.6f} s/it, {inputs_hist_len} it)")

        output_hist_unet.to_netcdf(output_hist_unet_path)
        del output_hist_unet
    else:
        print("Historical period - unet only - skipped (netcdf file already exists)")

    del inputs_hist

    inputs_future = stacked_X.sel( GCM =gcm).transpose('time','lat','lon','channel').sel(time = future_period).values
    inputs_future_len = len(inputs_future)


    output_future_path = f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_res_future_bwar780.nc'
    if not os.path.exists(output_future_path):
        timer_start = time.perf_counter()
        output_future = xr.concat([predict_parallel_resid(dm, unet,
                                    inputs_future,
                                    output_shape.sel(time = future_period), 64, orog.values, he.values, vegt.values,
                                    model_type=generator_model_type, output_add_factor = output_norm, scheduler=scheduler) for i in range(1)],
                                dim ="member")
        timer_stop = time.perf_counter()
        timer_elapsed = timer_stop - timer_start
        print(f"Future period - {generator_model_type} - elapsed: {timer_elapsed:.6f} s ({timer_elapsed/inputs_future_len:.6f} s/it, {inputs_future_len} it)")

        output_future.to_netcdf(output_future_path)
        del output_future
    else:
        print(f"Future period - {generator_model_type} - skipped (netcdf file already exists)")


    output_future_unet_path = f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_unet_future_bwar780.nc'
    if not os.path.exists(output_future_unet_path):
        timer_start = time.perf_counter()
        output_future_unet = xr.concat([predict_parallel_resid(dm, unet,
                                    inputs_future,
                                    output_shape.sel(time = future_period), 64, orog.values, he.values, vegt.values,
                                    model_type='UNET', output_add_factor = output_norm) for i in range(1)],
                                dim ="member")
        timer_stop = time.perf_counter()
        timer_elapsed = timer_stop - timer_start
        print(f"Future period - unet only - elapsed: {timer_elapsed:.6f} s ({timer_elapsed/inputs_future_len:.6f} s/it, {inputs_future_len} it)")

        output_future_unet.to_netcdf(output_future_unet_path)
        del output_future_unet
    else:
        print("Future period - unet only - skipped (netcdf file already exists)")

    del inputs_future

    # # normalization is now with a 1
    # output_hist = xr.concat([predict_parallel_resid(dm, unet,
    #                                stacked_X.sel( GCM =gcm).transpose('time','lat','lon','channel').sel(time = historical_period).values,
    #                                output_shape.sel(time = historical_period), 64, orog.values, he.values, vegt.values,
    #                                model_type='diffusion', output_add_factor = output_norm, scheduler=scheduler) for i in range(5)],
    #                         dim ="member")
    # output_hist_reg = xr.concat([predict_parallel_resid(dm, unet,
    #                                stacked_X.sel( GCM =gcm).transpose('time','lat','lon','channel').sel(time = historical_period).values,
    #                                output_shape.sel(time = historical_period), 64, orog.values, he.values, vegt.values,
    #                                model_type='UNET', output_add_factor = output_norm) for i in range(1)],
    #                         dim ="member")

    # output_future = xr.concat([predict_parallel_resid(dm, unet,
    #                                      stacked_X.sel(GCM=gcm).transpose('time', 'lat', 'lon', 'channel').sel(
    #                                          time=future_period).values,
    #                                      output_shape.sel(time=future_period), 64, orog.values, he.values,
    #                                      vegt.values, model_type='diffusion', output_add_factor = output_norm, scheduler=scheduler) for i in range(5)], dim ="member")
    # output_future_reg = xr.concat([predict_parallel_resid(dm, unet,
    #                                      stacked_X.sel(GCM=gcm).transpose('time', 'lat', 'lon', 'channel').sel(
    #                                          time=future_period).values,
    #                                      output_shape.sel(time=future_period), 64, orog.values, he.values,
    #                                      vegt.values, model_type='UNET', output_add_factor = output_norm) for i in range(1)], dim ="member")
    
    # output_future2 = xr.concat([predict_parallel_resid(dm, unet,
    #                                      stacked_X.sel(GCM=gcm).transpose('time', 'lat', 'lon', 'channel').sel(
    #                                          time=future_period2).values,
    #                                      output_shape.sel(time=future_period2), 64, orog.values, he.values,
    #                                      vegt.values, model_type='diffusion', output_add_factor = output_norm, scheduler=scheduler) for i in range(5)], dim ="member")
    # output_future_reg2 = xr.concat([predict_parallel_resid(dm, unet,
    #                                      stacked_X.sel(GCM=gcm).transpose('time', 'lat', 'lon', 'channel').sel(
    #                                          time=future_period2).values,
    #                                      output_shape.sel(time=future_period2), 64, orog.values, he.values,
    #                                      vegt.values, model_type='UNET', output_add_factor = output_norm) for i in range(1)], dim ="member")
    # outputs = xr.concat([output_hist, output_future], dim ="time")
    # outputs_reg = xr.concat([output_hist_reg, output_future_reg], dim="time")
    # outputs_test = outputs.sel(time = slice("2098","2099"))
    # outputs_reg_test = outputs_reg.sel(time=slice("2098", "2099"))
    # outputs = compute_signal(outputs[['pr']], quantiles, historical_period, future_period)
    # outputs_reg = compute_signal(outputs_reg[['pr']], quantiles, historical_period, future_period)
    
    # outputs2 = xr.concat([output_hist, output_future2], dim ="time")
    # outputs_reg2 = xr.concat([output_hist_reg, output_future_reg2], dim="time")
    # outputs2 = compute_signal(outputs2[['pr']], quantiles, historical_period, future_period2)
    # outputs_reg2 = compute_signal(outputs_reg2[['pr']], quantiles, historical_period, future_period2)
    
    # #outputs.attrs['title'] = outputs.attrs['title'] + f'   /n ML Emulated NIWA-REMS GAN v1 GCM: {gcm}'
    # if not os.path.exists(f'./outputs/{config["model_name"]}'):
    #     os.makedirs(f'./outputs/{config["model_name"]}')
    # outputs.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_ens_v2.nc')
    # outputs_reg.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_unet_v2.nc')
    # outputs2.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_ens_mid_century_v2.nc')
    # outputs_reg2.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_unet_mid_century_v2.nc')
    # outputs_test.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_ens_test_sample_v2.nc')
    # outputs_reg_test.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_unet_test_sample_v2.nc')
    # output_hist.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_res_hist_bwar780.nc')
    # output_hist_unet.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_unet_hist_bwar780.nc')
    # output_future.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_res_future_bwar780.nc')
    # output_future_unet.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_unet_future_bwar780.nc')

    # with open(f'./outputs/{config["model_name"]}/config_info.json', 'w') as f:
    #     json.dump(config, f)
