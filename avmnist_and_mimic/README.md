# AV-MNIST and MIMIC-III 

This guide provides instructions on accessing and using the AV-MNIST dataset and the MIMIC-III dataset for multimodal learning.

## AV-MNIST Dataset

### Data Access
Download the `avmnist.tar.gz` file from [this link](https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view?usp=sharing).

### Usage 🛠️
The available command-line options for AV-MNIST include:

- `--fusion_type=`: Choose the fusion method with options `lf` (late fusion) or `lrtf` (low-rank tensor fusion).
- `--model_type=`: Select the modeling approach from `unimodal`, `inter_modality`, `intra_modality`, and `inter_and_intra_modality`.
- `--test`: Evaluate the model performance.

#### Example Command 💻
```bash
python avmnist_main.py --model_type="inter_and_intra_modality" --fusion_type='lf'
```

## MIMIC-III Dataset

### Data Preparation
Access to MIMIC-III requires authorization. Follow the instructions [here](https://mimic.mit.edu/iv/access/) to obtain necessary credentials.

### Usage 🛠️
The command-line options available for MIMIC-III include:

- `--task=`: Specifies the task to be performed. The available options are `mortality`, `mimic_1`, `mimic_7`.
- `--model_type=`: Defines the model's type. Options include `unimodal`, `inter_modality`, `intra_modality`, `inter_and_intra_modality`.
- `--test`: Activates evaluation mode for the trained model.

#### Example Command 💻

```bash
python your_program.py --task="mortality" --model_type="inter_and_intra_modality"
```

## 🎗️ Acknowledgment

The code is build upon [pliang279/MultiBench](https://github.com/pliang279/MultiBench).

We thank the authors for their amazing work and releasing the code base.
