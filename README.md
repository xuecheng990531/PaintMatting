# MPM

This README provides instructions for using An Iterative Approach for High-Quality Mask Generation in Image Matting (MPM).

<p align="center"><img src="imgs/mpms.png" width="700"/></p>
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
    - Run the following command for iterative optimization from roughprompt to accurate mask.
    ```angular2html
    python scripts/PaintSeg.py --outdir $outdir$ --iters $iter_num$ --steps $diffusion step$ --dataset $dataset$ 
    ```
    - Put the obtained results in the `MPM_MTM_Modules` folder together with the datasets.
2. Use MPM_MTM_Modules
    - Run the `main.py` file for training.
   ```angular2html
    python main.py  --datasets {your datesets location} --fe {frozen encoder?} --norm {is norm your datasets image?}
    ```

That's it! You're ready to use MPM with PyTorch and Python 3.9.
