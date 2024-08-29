# MPM

This README provides instructions for using An Iterative Approach for High-Quality Mask Generation in Image Matting (MPM).


<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9-ff69b4.svg" /></a>
<a href= "https://pytorch.org/"> <img src="https://img.shields.io/badge/PyTorch-2.1-2BAF2B.svg" /></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
## Prerequisites
- PyTorch 2.0
- Python 3.9
- You can use requirements to install the environment, but I wouldn't recommend it as I haven't tested it.
## Usage

1. Generate Accurate Binary Mask
    - Open `MPM_Mask_Acquire`
    - python scripts/PaintSeg.py --outdir $outdir$ --iters $iter_num$ --steps $diffusion step$ --dataset $dataset$ 
    - Use iterative methods to generate accurate binary masks
2. Use MPM_MTM_Modules
    - Pass the generated binary mask and original image as inputs to `MPM_MTM_Modules.py`

That's it! You're ready to use MPM with PyTorch and Python 3.9.
