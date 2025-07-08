## Installation

- Python 3.9
- PyTorch 1.9.0
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
# Clone the github repo and go to the default directory 'RecuDiff-UT'.
git clone https://github.com/lishenqu/RecuDiff-UT.git
cd RecuDiff-UT
conda create -n RecuDiff-UT python=3.9
conda activate RecuDiff-UT
pip install -r requirements.txt
```

## Training

  ```shell
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train/Blur_S1.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train/Blur_S2.yml --launcher pytorch

- The training experiment is in `experiments/`.

## Testing

  ```python
  # generate images
  python test.py -opt options/test/Blur.yml
