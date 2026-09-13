# Introduction

Personal Project driven by interest in math and Machine Learning. Began with building a complete digit recognition MLP in numpy, eventually scaled it to 
a CNN to recognize both digits and the english alphabet. Currently working on training a model to recognize image, the basis of which is in this repo, 
along with implementing versions of the alphabet CNN to develop a note taking app. 

# Requirements

Requires Python 3.12 to run the torch CUDA version required for my laptop GPU. All other requirements are stored in requirements.txt
For CUDA to work on my laptop, I needed to use torch from https://download.pytorch.org/whl/cu126. Running 

'''pwsh
python -m pip install -r requirements.txt
'''

fails to install this version. 

Must separately run: 

'''pwsh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
'''