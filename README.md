# Integrating Model-based Reconstruction and Deep Learning for Accelerating Mass Spectrometry Imaging


Official implementation of "Integrating Model-based Reconstruction and Deep Learning for Accelerating Mass Spectrometry Imaging".

## Citation
If you find our work or this repository useful, please consider giving a star ⭐ and a citation.

```
@article{PnPMSI2025,
  title={Integrating Model-based Reconstruction and Deep Learning for Accelerating Mass Spectrometry Imaging},
  author={Mithunjha Anandakumar and Timothy Trinklein and Jonathan Sweedler and Fan Lam},
  journal={},
  volume={},
  issue={},
  pages={},
  year={},
  doi = {}
}
```

This repository contains the implementation of the UNet-based denoiser model as a regularizer and its integration with model-based reconstruction under the plug-and-play framework.

![figure](https://github.com/user-attachments/assets/f18fd553-f206-4a02-a791-a53f40e19a03)


## Dataset
Find the dataset used in our work at : [https://databank.illinois.edu](https://databank.illinois.edu/datasets/IDB-9740536)


## Getting Started
### Installation
The algorithms were developed in the Pytorch Environment : [https://pytorch.org/](https://pytorch.org/).

```python
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Run the code below to install all other dependencies.

```python
pip install -r requirements.txt
```

### Training of the Denoiser

Use the following code to train the model for a particular dataset case and loss function.

```python
python3 training.py --experiment_name <EXPERIMENT_NAME> --epochs <#EPOCHS> --data_directory <PATH_TO_DATA> --save_path <PATH_TO_A_FOLDER> --noise_level_range <[LOWER, UPPER]> 
```

### Plug and Play for sparse sampled ion images

Use the following code to evaluate the performance of the pre-trained model for any dataset case.

```python
python3 run.py --sampling_percentage <SAMPLING_PERCENTAGE> --max_iteration <MAX_ITER> --model_path <MODEL_PATH> --save_path <OUTPUT_PATH>
```

### Sample output
The quantitative and qualitative output for the **rat brain FT-ICR data at sampling percentage = 50%** dataset case follows.

*Evaluation metric : mean value for the entire test dataset (standard deviation)* 

**SSIM** : 

**PSNR** :




