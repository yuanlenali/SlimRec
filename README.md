# SlimRec
SlimRec provides two different ways to compress NN based recommendation models, i.e., popularity hashing and quantized aware training. The NN based recommendation models 's size mostly comes from embedding table of sparse features. An embedding table 's size equals to the hash size (vocab size) * embedding dimmension * parameter sizes (32 bits if a float). 
  - Popularity hashing compresses the hash size by mapping the most frequent ids to a consectuve space, and collide less frequent ids together. Popularity hashing utilize the long tail distribution of data, i.e., both users and movies. An 0.3 compression ratio can be achieved even with slight performance gain. The gains is due to the mitigation of cold start problem by colliding different less frequent ids together.
  - Quantized awared training achieved the best performance among all quantization techniques, i.e., post training quantization.

## Installation

    source install.sh

## Data
### Dataset
The dataset used is movie Tweeting Dataset, a recsys challenge 2014 dataset. The raw dataset can be found in raw_data folder
### Data Distribution
The dataset distribution follows long tail distribution. The analysis of the data can be found in jupyter notebooks in analysis folder.
### Data Preprocessing
Please run the jupyter notebook in data_preprocessing folder, to generate processed dataset for training and testing. The processed dataset is in data folder.
### Movie Images
Movie Images will be used during inference. To accelerate the web application, please run utils/crawl_image.py to download all movie images in advance. Otherwise, it is still ok to download the necessary movie images when during inference.

## Web Application
The inference is shown with streamlit frontend. To run the demo, 

	streamlit run app.py

