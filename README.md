# On-the-Extrapolation-of-Generative-Adversarial-Networks-for-downscaling-precipitation-extremes
This repository contains the code for the Manuscript titled: "On the Extrapolation of Generative Adversarial Networks for downscaling precipitation extremes in warmer climates".
The data required for this repository can be found at: https://doi.org/10.5281/zenodo.13755687

Please contact me if you have any further questions about this work (Neelesh Rampal). 

# Copying the Data from Zenodo / Cloning
First, clone this repository ```bash git clone https://github.com/nram812/On-the-Extrapolation-of-Generative-Adversarial-Networks-for-downscaling-precipitation-extremes.git```.

Then copy the files so the "inputs" and "outputs" folders so they appear like the following:
* inputs/
    * predictor/
        * target_ACCESS-CM2_hist_ssp370_pr.nc
        * Other_GCMs_hist_SSP370_target_fields_pr.nc

    * target/
        * predictor_ACCESS-CM2_hist_ssp370.nc
        * Other_GCMs_hist_SSP370_predictor_fields.nc
        
* outputs/
    * gt_cc_signal_final_v2_geq1mm.nc

These files are collectively about 40GB in size. The "static" predictor files, "normalization" files should be part of this repository. 
Please note that the "Other_GCMs...." are only used for evaluation. The training and evaluation dataset configurations are provided in a configuration file. 

## Setting Up A Python Environment
These experiments have been performed on an Nvidia A100 GPU, and thus the code has been developed in a
slurm (job scheduling) and unix enivironment.

These experiments can be run without slurm, but will require special configurations.

To create the python environment for training this algorithm use the following commands:

```bash
conda env create -f environment.yaml
conda activate ml_env_v2
```


## Training the Model
This step assumes that one has access to a HPC/GPU cluster for training. All experimental configurations are provided in the following directory. 
* experiment_configs/
    *future_full.json
    *future_only.json
    *historical.json

You may need to modify these configuration files for your specific directory/configuration. But these configuration files use relative paths, so hopefully little modification is required 


An example Json file is shown below.
Please modify "mean" (data to normalize the predictors), "std" (data to normalize the predictors),
 "train_x" (predictor variables),"train_y" (target variables), "src_path" (the path where you have cloned the repo),
 "output_folder" (where you would like to store the model files), "static_predictors" (the location where the topography files are stored).
Once the configuration files have been modified for each of the experiments (please note all need to be manually modified), we can run these experiments by slurm.

### Running the Experiments
The experiments have been run using Python 3.8.5, but the experiments are generally compatible with >Python 3.6 and Tensorflow >2.5.

There are three important scripts when running the experiments
* run_bash_config.sh (submits slurm jobs for every config file in parallel)
* run_config_experiments.sl (the slurm configuration for each job)
* train_model_rain_future.py (the script that train the model)

The paths in the code have been configured with relative paths. 

#### Option 1: Run_Bash_Config.sh for to run all experiments

While only 128GB of memory is typically required for all these experiments (slurm jobs), we run all experiments, with a default of 200GB of memory on slurm.
To run all the experiments (with or without the intensity constraint), please execute the "run_bash_config.sh" file.
You will need to modify the following, for your specific operation.

Please see the comments below, on how to modify the file (run_bash_config.sh). Please note you may need to modify the slurm files to run all experiments for your cluster. 

```bash
#!/bin/bash -l

module purge # optional
module load NeSI
module load cuDNN/8.1.1.33-CUDA-11.2.0
#conda activate ml_env
nvidia-smi

#cd "/nesi/project/niwa00018/ML_downscaling_CCAM/On-the-Extrapolation-of-Generative-Adversarial-Networks-for-downscaling-precipitation-extremes"
# change directory to the working directory of interest

directory="experiment_configs"

if [ ! -d "$directory" ]; then
    echo "Directory does not exist: $directory"
    exit 1
fi


for file in "$directory"/*.json; do
    if [ -f "$file" ]; then
        # Process each .json file as needed
        echo "Processing file: $file"
        sbatch ops/run_config_experiments.sl $file

        # Add your custom logic here to work with each file
        # For example, you could read the content of the file:
        # content=$(cat "$file")
        # echo "File content: $content"
    fi
done

```
#### Option 2: Run the Python Script
If you are running this on a personal machine you can run an individual experiment by the following code:
```bash
/path/to/python/interpret train_model_rain_future.py /path/to/config/file/config.json
```
## Inference of the Model and Climate Change Signal
The model weights will automatically be saved to the *models/your_model_name" folder every 10 epochs

To run the model in inference you will need to re-name the weights of your final model of choice. Please do this for every experiment. 
i.e. 
```bash 
    cp ./models/Future_Hist_trained_GAN/unet_epoch_200.h5 ./models/Future_Hist_trained_GAN/unet_final.h5
    cp ./models/Future_Hist_trained_GAN/generator_epoch_200.h5 ./models/Future_Hist_trained_GAN/generator_final.h5
```

Once you have trained a model, you can run the model in inference, and generate it's climate change signal. 
This can be done for all three models with the following jobs:

```bash 
    sbatch model_inference/apply_cascaded_model_ancil_pr.sl ./models/Future_Hist_trained_Gan/config_info.json
    sbatch model_inference/apply_cascaded_model_ancil_pr.sl ./models/Future_only_trained_Gan/config_info.json
    sbatch model_inference/apply_cascaded_model_ancil_pr.sl ./models/historically_trained_Gan/config_info.json
```

By running the model in inference, the outputs will be saved in the *outputs* folder. Note only the climatologies are saved as a pose to all predictions. 
The model inference uses a different configuration file for the data, and will load the evaluation data here. 






