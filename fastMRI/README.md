# FastMRI

## Data Preparation

### Downloading the Dataset

The FastMRI dataset can be obtained from the following link: [FastMRI Dataset](https://fastmri.med.nyu.edu). Additionally, annotations are available at [FastMRI-plus Annotations](https://github.com/microsoft/fastmri-plus/tree/main/Annotations).

### Generating Slice-level Data

To obtain slice-level data, update the necessary paths in the provided script and execute the following commands:
```bash
cd data_processing/knee/
python knee_singlecoil.py
```

### Creating Data Splits

Generate the training, validation, and testing splits by running:

```bash
cd data_processing/knee/
python generate_knee_metadata.py
```


## Model Training

Execute the command below to train the inter+intra-modality model. The script allows for the customization of input parameters through various flags to suit different model requirements.

```bash
python classifier.py --data_space='ktoi_w_magphase' --lr=1e-6 --weight_decay=1e-2 --model_type_class='multimodal'
```

### Flag Descriptions

- `--data_space`: Specifies the input space for the model. The available options are `ktoi_w_mag`, `ktoi_w_phase`, `ktoi_w_magphase` specifying the unimodal and inter-modality model.
- `--lr`: Sets the learning rate for model training.
- `--weight_decay`: Defines the weight decay.
- `--model_type_class`: Use `multimodal` to run `inter_and_inter_modality` model.
- `--mode`: Specifies the mode of operation, such as training, validation, or testing.



