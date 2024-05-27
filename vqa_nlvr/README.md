# VQA-VS and NLVR2

## Dataset Preparation

The VQA-VS data can be downloaded from [**GoogleDrive**](https://drive.google.com/drive/folders/1i6xqke5X5GoQ8YGoNcs3rtMsDtgs4OLG?usp=sharing). To create the arrow files, we follow [ViLT](https://github.com/dandelin/ViLT) and [METER](https://github.com/zdou0830/METER). See [this link](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.

## Fine-tuning on Downstream Tasks

- Download the FIBER pre-trained model from [**here**](https://datarelease.blob.core.windows.net/fiber/coarse_grained/fiber_pretrain.ckpt).
### VQA-VS

```bash
python run.py with \
  seed=3 \
  task_inter_and_intra_multimodal_vqa \
  data_root=./datasets/vqa/arrow_files/ \
  log_dir=./inter_and_intra_modality/ \
  num_gpus=4 \
  per_gpu_batchsize=16 \
  max_epoch=100 \
  learning_rate=1e-4 \
  load_path=fiber_pretrain.ckpt
```

### NLVR2

```bash
python run.py with \
  seed=3 \
  task_inter_and_intra_multimodal_nlvr2 \
  data_root=./datasets/vqa/arrow_files/ \
  log_dir=./inter_and_intra_modality/ \
  num_gpus=4 \
  per_gpu_batchsize=16 \
  max_epoch=100 \
  learning_rate=1e-4 \
  load_path=fiber_pretrain.ckpt
```

## Acknowledgements

The code is based on [FIBER](https://github.com/microsoft/FIBER/tree/main/coarse_grained).

We thank the authors for their amazing work and releasing the code base.
