<div align="center">

# A Framework for Multi-modal Learning:<br> Jointly Modeling Inter- & Intra-Modality Dependencies
</div>

**TL;DR: We distinguish between different modeling paradigms for multi-modal learning from the perspective of generative models and offer a general recipe for designing models that efficiently leverage multi-modal data, leading to more accurate predictions.**


## Abstract
Supervised multi-modal learning involves mapping multiple modalities to a target label.
Previous studies in this field have concentrated on capturing in isolation either the inter-modality dependencies (the relationships between different modalities and the label) or the intra-modality dependencies (the relationships within a single modality and the label). 
We argue that these conventional approaches that rely solely on either inter- or intra-modality dependencies may not be optimal in general.
We view the multi-modal learning problem from the lens of generative models where we consider the target as a source of multiple modalities and the interaction between them. Towards that end, we propose inter- \& intra-modality modeling (I2M2) framework, which captures and integrates both the inter- and intra-modality dependencies, leading to more accurate predictions. 
We evaluate our approach using real-world healthcare and vision-and-language datasets with state-of-the-art models, demonstrating superior performance over traditional methods focusing only on one type of modality dependency. 

## Prerequisites

```
$ pip install -r requirements.txt
```

## 📚 Datasets Overview and Instructions

Our project utilizes several datasets, each organized within specific folders. Below is an overview of the datasets and links to detailed instructions in their respective folders:

### AVMNIST
- **Description:** Audio-Vision MNIST (AV-MNIST) combines audio and visual modalities for MNIST digit (0-9) recognition task. 
- **Instructions:** For detailed instructions on how to use this datasets, refer to the [README in the AVMNIST_and_MIMIC folder](/avmnist_and_mimic/README.md).

### fastMRI
- **Description:** The fastMRI dataset is a large-scale dataset that consists of raw k-space knee data alongside anonymized clinical magnetic resonance (MR) images and pathology labels.
- **Instructions:** Detailed steps for using the fastMRI dataset can be found in the [README in the fastMRI folder](/fastMRI/README.md).

### MIMIC-III
- **Description:** he MIMIC-III dataset encompasses ten years of intensive care unit (ICU) patient data
from Beth Israel Deaconess Medical Center. The dataset is divided into two modalities: 1) time-series modality, which
includes hourly medical measurements over 24 hours, and 2) static modality, capturing a patient’s medical information.
We consider three tasks: a) predicting the mortality of a patient within 1 day, 2 days, 3 days, 1 week, 1 year and beyond, and b) two binary classification tasks for ICD-9 codes, one to assess if a patient falls under group 1 (codes 140-239; neoplasms) and another for group 7 (codes 460-519; diseases of respiratory system).
- **Instructions:** For detailed instructions on how to use this datasets, refer to the [README in the AVMNIST_and_MIMIC folder](/avmnist_and_mimic/README.md).

### VQA
- **Description:** The objective of VQA is to answer
questions about images. The eval-
uation encompasses the IID and nine out-of-distribution
(OOD) test-sets released by VQA-VS dataset.
- **Instructions:** Comprehensive guidelines on these datasets are available in the [README in the VQA_and_NLVR2 folder](/vqa_nlvr/README.md).

### NLVR
- **Description:** NLVR2 represents a binary classification task in which the goal is to determine whether the
text description correctly describes a pair of two images.
- **Instructions:** Comprehensive guidelines on these datasets are available in the [README in the VQA_and_NLVR2 folder](/vqa_nlvr/README.md).

