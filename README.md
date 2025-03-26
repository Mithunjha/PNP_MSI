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

This repository contains the implementation of the model-based forward model, which generates simulated DEEP image stacks, and the DEEP-squared inverse model, which reconstructs de-scattered images from 32 patterned DEEP-TFM measurements.


## Dataset
Find the dataset used in our work at : [https://databank.illinois.edu](https://databank.illinois.edu/datasets/IDB-9740536)


## Getting Started
### Installation
The deep learning algorithms were developed in Pytorch Environment : [https://pytorch.org/](https://pytorch.org/) and the forward model was implemented in MatLab.

```python
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Run the below code to install all other dependencies.

```python
pip install -r requirements.txt
```

### Training of the Denoiser

Use the following code to train the model for a particular dataset case and loss function.

```python
python3 run.py --save_model_path <PATH_TO_A_FOLDER> --lossfunc <LOSS_FUNCTION> --experiment_name <EXPERIMENT_NAME> --epochs <#EPOCHS>
```

### Plug and Play for sparse sampled ion images

Use the following code to evaluate the performance of the pre-trained model for any dataset case.

```python
python3 run.py --case <CASE> --model_path <MODEL_PATH> --output_path <OUTPUT_PATH>
```

### Sample output
The quantitative and qualitative output for the **rat brain FT-ICR data at sampling percentage = 50%** dataset case follows.

*Evaluation metric : mean value for the entire test dataset (standard deviation)* 

**SSIM** : 

**PSNR** :




