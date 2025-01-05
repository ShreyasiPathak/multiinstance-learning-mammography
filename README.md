# Weakly Supervised Learning for Breast Cancer Prediction on Mammograms in Realistic Settings

## Introduction
This repository contains the source code of case-level breast cancer prediction using mammography. The model takes a set of images per mammography case (exam) as input and predicts the class label benign or malignant. The model generates a saliency map for each image and 6 candidate ROIs per image. <br/>

A overview of our model framework can be seen below. 

<img src="mil-breast-cancer-model-overview.png" alt="model-overview" style="height: 300px; width:800px;"/>

A visualization of our model output is shown below. It shows 4 images in a case and 6 ROI candidates extracted by our model for each image, along with the importance (attention) score associated with the images and the ROIs. This is a malignant case which has been classified as malignant by our model and the relevent ROIs containing mass abnormality have been correctly extracted.

<img src="visualization_case_patches.PNG" alt="model-output-visualization" style="height: 400px; width:800px;"/>

### A realistic setting dataset (MGM) 
Examples of different mammogram images that can be present in a case in realistic clinical scenario (e.g., in our private dataset MGM) can be found [here](/MGM-image-samples).
In a realistic setting, cases can contain a non-fixed number of images. We have shown the different image/view combinations in the MGM cases [here](/MGM-view-combination/MGM-view-combination.md) to get a clearer perspective of types of cases in a realistic clinical setting. Due to privacy regulations, access to this dataset is not yet possible, but we are working on finding a solution for this. 

### Results
The F1 and AUC score of our models can be found in our paper, however, we have also included the precision and recall scores of our models in this repository (can be found [here](Detailed-Result-Table.md)). The number of models parameters can be found [here](Detailed-Result-Table.md).

## Running the code
Below you can find the instructions for training our models - both single-instance and multi-instance models, in this repository. 

### Prerequisites
- Python 3.9.15
- Pytorch 1.11.0+cu113
- Cuda 11.3
- matplotlib 3.5.1
- opencv-python 4.6
- pandas 1.4.2
- openpyxl 3.0.9
- scikit-learn 1.0.2
- seaborn 0.11.2

### Access to Datasets
We used 3 datasets in our work - CBIS (public dataset), VinDr (public dataset) and MGM (private dataset). We provide instructions on how to train and test our model on the 2 public datasets. <br/> 
- CBIS can be downloaded from [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629). <br/>
- VinDr can be downloaded from [here](https://vindr.ai/datasets/mammo). <br/>

### Creation of preprocessed datasets
1. Convert the dicom images to png with this [script](/src/data_processing/dicom_to_png.py). <br/>
2. Convert the original png images to preprocessed png images (to remove irrelevant information and remove extra black background) according to our [image cleaning script](/src/data_processing/image_cleaning.py). Example of the results of our image preprocessing algorithm can be found [here](/image-preprocessing). We used these preprocessed images as input to our model.

### Preparation of input csv file 
1. Create the input csv file which contains the list of input instances and their corresponding groundtruth, for multi-instance and single-instance model training using the [script](/src/data_processing/input_csv_file_creation_cbis.py).
2. We have provided a small snippet from our input csv file for CBIS and VinDr [here](/input-csv-files).

### Configuration files
For training models using our script, you need to create configuration files containing all settings for the training. You can refer to our configuration files [here](all-config-files).

For creating your own configuration file for training a model, do as follows:
   > cd src <br/>
   > python setup/config_file_creation.py  <br/>

Please add your absolute input data path to the field "preprocessed_imagepath" and the path to the input csv file in the fields "SIL_csvfilepath" and "MIL_csvfilepath" in the script. <br/>

For reproducibility, we release the configuration files used for training all the models reported in our paper [here](all-config-files). 

### Model training
1. Our model training script can be found [here](src). 
2. Add the following command in your terminal or sbatch file (this is the path to wherever you have downloaded the src folder), otherwise the main script will not be able to find different modules: 
   > export PYTHONPATH=/home/src 
3. Run the code as follows: 
   > python train_eval/train.py --config_file_path sample-config-files/cbis/es-att-img/ --num_config_start 0 --num_config_end 1 --mode train <br/>
   
   Explanation of the above command: <br/>
   --config_file_path, I have added the location of my config file. You can add yours. Trained models and results will get stored in this location. <br/>
   --num_config_start and --num_config_end are the start and end id of the config files if there are >1 config files that you want to execute. If you only want to train the model on 1 config file, then put the num_config_start and num_config_end as 0 and 1. This argument is useful during hyperparameter tuning, where each config file contains a different hypeparameter combination <br/>

## Evaluation of trained models
1. We have provided our pretrained models [here](https://www.dropbox.com/scl/fo/jgmh6f9t0po0d6rofi9mu/h?rlkey=znua1rnytc60uzz103a7yre9r&st=06vt139f&dl=0) for users who want to only evaluate our trained model on CBIS/VinDr, or if someone wants to use our pretrained model for training on other datasets.
2. For testing one of our pretrained models, run:
  > python train_eval/train.py --config_file_path MIL-breastcancer-pretrained-models/cbis/es-att-img --num_config_start 0 --num_config_end 1 --mode test

## State-of-the-art (SOTA) reproducibility
We have described in detail how we reproduced 4 SOTA models and have also added some extra details for training our model [here](Reproducing-SOTA-and-training-details-MIL-models.md).<br/>
Further, you can train our implementation of the SOTA models using our source code by using the config files [here](all-config-files/Table10). 

## ROI IoU and DSC score calculation
IoU and DSC scores can be calculated using the script [here](src/analysis/mask_roi_match.py).<br/>
Sample csv file used in this script for vindr can be found [here](input-csv-files/vindr/vindr_singleinstance_imgpreprocessing_size_withroiloc.csv). The fields are image name, min_x, min_y, max_x, max_y of the processed image, original image height and width, processed image height and width, study_id, series_id, laterality, view_position, breast_birads, breast_density, category of the finding, birads of the finding, xmin, ymin, xmax, ymax of the ROI, split (training or test). <br/>
Sample csv file used in this script for cbis can be found [here](input-csv-files/cbis/cbis-ddsm_singleinstance_imgpreprocessing_size_mod.csv). The fields are image name, min_x, min_y, max_x, max_y of the processed image, original image height and width, processed image height and width. The script needs the path location of the ROI masks. <br/>

## Citation to our paper
Pathak, S., Schlötterer, J., Geerdink, J., Veltman, J., van Keulen, M., Strisciuglio, N. and Seifert, C., 2024. [Weakly Supervised Learning for Breast Cancer Prediction on Mammograms in Realistic Settings](https://arxiv.org/pdf/2310.12677). arXiv preprint arXiv:2310.12677.
```
@misc{pathak2024caselevelbreastcancerprediction,
      title={Case-level Breast Cancer Prediction for Real Hospital Settings}, 
      author={Shreyasi Pathak and Jörg Schlötterer and Jeroen Geerdink and Jeroen Veltman and Maurice van Keulen and Nicola Strisciuglio and Christin Seifert},
      year={2024},
      eprint={2310.12677},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2310.12677}, 
}
```
