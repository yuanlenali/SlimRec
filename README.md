# SlimRec
SlimRec provides two different ways to compress NN based recommendation models, i.e., popularity hashing and quantized aware training. The NN based recommendation models 's size mostly comes from embedding table of sparse features. An embedding table 's size equals to the hash size (vocab size) * embedding dimmension * parameter sizes (32 bits if a float). 
  - Popularity hashing compresses the hash size by converting the most frequent ids tot a consectuve space, and collide less frequent ids together. An 0.3 compression ratio can be achieved even with slight performance gain. The gains is due to the mitigation of cold start problem by colliding different less frequent ids together.
  - Quantized awared training achieved the best performance among all quantization techniques, i.e., post training quantization.

## Installation

    source install.sh

## Data
### Dataset
The dataset used is movie Tweeting Dataset, which is also used in recsys 2014. The raw dataset can be found in raw_data folder
### Data Preprocessing
Please run the jupyter notebook in data_preprocessing folder, to generate processed dataset for training and testing. The processed dataset is in data folder.
### Movie Images
Pls run utils/crawl_image.py to download all movie images. They will be used for the web application.

## Web Application
The inference is shown with streamlit frontend. To run the demo, 

	streamlit run app.py

