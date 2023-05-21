# IMAD
This repo contains code and published data for the AINL2023 paper [IMAD: IMage Augmented multi-modal Dialogue](https://arxiv.org/abs/2305.10512).

Our dataset serves for task of interpreting image in the context of dialogue. Published code could help with 
1. Classifying if utterance is replaceable with image
2. Finding the best image for utterance
3. Generation of utterance that was replaced with an image

# Data
IMAD Dataset created from mutlitple dialogue dataset sources and Unsplash images. 
Every sample from dataset is 
1. Context of dialogue
2. Image
3. Replaced utterance

![Example](examples.png)

Dataset is availible with images_id at [HuggingFace](https://huggingface.co/datasets/VityaVitalich/IMAD) or could be requested with an images directly via email.

# Code

## Replace Text with Image

This tool performs classification if utterance could be potentially replaced with an image. For this purposes data should contain list of features:
1. Image Score
2. Sentence Similarity
3. BLEU
4. Maximum Entity Score
5. Thresholding

Classification is performed with model from [models directory](../main/models/).
Generation of all the features is performed with models from [features scripts directory](../main/scorers/).
Example of usage is shown at [Text Replacing Tutorial](main/TextReplacingTest.ipynb). Note that scripts are using [Paths](#Paths), which is essential to this script

## Find better image with VQA

This tool is capable of finding better image with the use of BLIP VQA. Long story short it finds top-N (N is specified) images that are closest to utterance and then scores them with VQA model. 
This is performed with models from [features scripts directory](../main/scorers/).
Example of usage is shown at [Image Replacing Tutorial](main/VQATest.ipynb)

## Paths

This is a special dataclass, that contains all the paths that would be used in scripts

* __dialog_features_path__ is the path to the directory where utterances embedding vectors are stored. Initially it could be empty and vectors will be generated during the script run. The example is shown in tutorial and default value is ``` './feature_vectors/test_vectors/' ```. Make sure you create new directory or clean it before running your examples
* __image_vectors_path__ is the path to the .pt file that contains images embedding vectors. Default value is ``` './images/vectors.pt' ```
* __output_path__ is the path to the output .json file. Script will save all the output to that path and also read from it sometimes. Default values is ``` './outputs/test_output.json' ``` .
* __temporary_path__ is the path to the temporary .json file. It is used to store some outputs, that are not valuable at the end. Default values is ``` './outputs/temporary_path.json' ```
* __entity_vectors_path__ is the path to the directory where entities embedding vectors are stored. Initially it could be empty and vectors will be generated during the script run. Default value is ``` './feature_vectors/entity_vectors/' ```.
* __images_dataset_path__ : is the path to the dataset containing information about images. It should contain image ids, url, description and ai_desription. You can leave them blank except the id. Default value is ``` './images/dataset.json' ```. 
* __path2images__ is the path to the directory, that contains raw images. Images should be named with id, that has been used in __images_dataset_path__. Default value is './images/full_images/'.
* __path2images_features__ is the path to the directory, that contains images embedding vectors, that are named the same as id in __images_dataset_path__. Default value is ``` './images/vectors' ```
* __path2trained_model__ is the path to the trained model for Text Replacing. You can use the default value ``` './models/random_forest.joblib' ```


# License

TODO

# References

TODO
