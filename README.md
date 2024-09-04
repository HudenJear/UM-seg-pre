

<br><br><br>

# UMsepre (PyTorch)

This is UMsepre Segmentation and basal diameter prediction model deplementation. You are welcome to use model codes and the pretrained model in your work.

For better perfromance , this model is still under development...


## Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

Use train command to locate the missing package.

### Dataset

Please put the data in the UM dataset dictionary, there are some sample files for the 


### train


<!-- - Train a model:

```bash
CUDA_VISIBLE_DEVICES=0 python ./UMsep/train.py -opt ./UMsep/options/UM-assist.yml 
``` -->

- Test the model:
```bash
 python ./UMsep/test.py -opt ./UMsep/options/test_UMseg.yml
```
