#!/usr/bin/env bash
set -e

conda install pandas=1.2.0
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install git+https://github.com/RobustBench/robustbench.git@v1.0
pip install robustness
pip install argparse
pip install pyyaml