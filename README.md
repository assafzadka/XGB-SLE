# XGB-SLE
This repository contains the code and resources for the paper [A wearable sensor and machine learning estimate step length in older adults and patients with neurological disorders](https://www.nature.com/articles/s41746-024-01136-2), published in npj Digital Medicine. 

![Fig 6 - Copy](https://github.com/user-attachments/assets/5fc90b09-e4f7-4de7-a133-3e79f8febbbf)
## Overview 
This project develops machine learning models to estimate step length from wearable sensor data. The models are trained and tested on a diverse dataset including healthy adults and patients with neurological conditions such as Parkinson's Disease (PD) and Multiple Sclerosis (MS).


## Installation 
To get started, clone the repository and install the required packages: ```bash git clone https://github.com/assafzadka/XGB-SLE.git cd XGB-SLE pip install -r requirements.txt

## Usage
### Training Models
To train the XGBoost models:
python src/train_xgb.py --data_path data/training_data.csv

### Testing Models
To test the trained models:
python src/test_data.py --model_path models/xgb_model.json --data_path data/test_data.csv

### Model Selection
To compare different models:
python src/model_selection.py --data_path data/complete_dataset.csv

## Results
The results of our models are evaluated based on RMSE and ICC metrics. Below are some of the visualizations from our experiments:
![Fig 3 - Copy](https://github.com/user-attachments/assets/82ec2dad-afec-4d1a-8a2b-48cdf62001c9)

## Citation
If you use this code in your research, please cite:
@article{Zadka2024,
  title={A wearable sensor and machine learning estimate step length in older adults and patients with neurological disorders},
  author={Assaf Zadka, Neta Rabin, Eran Gazit, Anat Mirelman, et al.},
  journal={npj Digital Medicine},
  year={2024},
  doi={10.1038/s41746-024-01136-2}
}
## Contributions
We welcome contributions! Please feel free to contact.

## License
The code in this repository is licensed under the MIT License. Feel free to use, modify, and share it under those terms.
