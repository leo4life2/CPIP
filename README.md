# CPIP: Contrastive Position-Image Pertaining for VPR

CPIP is a project adapted from the implementation of CLIP (Contrastive Language–Image Pre-training) aimed at enhancing Visual Place Recognition (VPR). By leveraging contrastive learning between position and image data, CPIP aims to improve the accuracy and robustness of VPR systems.

## Background

Visual Place Recognition (VPR) is a crucial component in various applications, including autonomous navigation and augmented reality. The CPIP project introduces a novel approach by incorporating contrastive learning to improve feature extraction and matching in VPR tasks.

## Setup and Training Guide

### Prerequisites

- Access to a Singularity container environment.
- An overlay file for creating a writable file system within Singularity.
- SLURM for job scheduling on HPC resources.

### Environment Setup

1. **Singularity and Overlay**: Ensure Singularity is installed and you have an overlay file ready for use.

2. **Create a Conda Environment**:
   Navigate to your Singularity overlay directory and execute the following command to create a Conda environment using the `environment.yml` file. This file includes all necessary dependencies for CPIP.

   ```bash
   singularity exec --overlay your_overlay_file.ext3 /path/to/singularity_image.sif /bin/bash -c "
       conda env create -f /path/to/environment.yml
   "
   ```
   Replace the placeholders with the appropriate paths for your setup.

### Running CPIP Training

Utilize the provided SLURM script template to submit training jobs. Customize the script with your details before submission.

#### SLURM Script Customization

Below is the SLURM script adapted for CPIP. Replace placeholders with your information.

```bash
#!/bin/bash

# SLURM settings
#SBATCH --partition=a100_1,a100_2,v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=CPIP
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<YOUR_EMAIL>
#SBATCH --output=CPIP_%j.out
#SBATCH --error=CPIP_%j.err

# Paths
ext3_path=/scratch/$USER/<OVERLAY_FILE>
sif_path=/scratch/work/public/singularity/<SINGULARITY_IMAGE>
TORCH_HOME=/scratch/$USER/pytorch

cd /scratch/$USER/<CPIP_DIRECTORY>

# Start training
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
conda activate cpip
python main.py"
```

- `<YOUR_EMAIL>`, `<OVERLAY_FILE>`, `<SINGULARITY_IMAGE>`, and `<CPIP_DIRECTORY>` should be updated to reflect your project's specifics.

#### Submitting the Job

Submit the customized SLURM script using:

```bash
sbatch your_customized_script.slurm
```

### Output and Model Saving

- The script's standard output (stdout) and standard error (stderr) will be saved to the files specified in the SLURM script's `--output` and `--error` options, respectively. These files will contain important information about the job's execution and any errors that may have occurred.
- The best model achieved during training will be saved as `best.pt` in the current working directory. Ensure you have sufficient permissions and space in this directory to save the model file.

### Configuration Adjustments

Modify `config.py` as needed to tune training parameters, select models, and adjust epochs for your specific VPR objectives.

## Project Goal

The ultimate goal of CPIP is to advance VPR capabilities by integrating contrastive learning techniques, thereby improving the performance of systems relying on visual place recognition. This project is a stepping stone towards achieving higher accuracy and reliability in VPR applications.