# IMAD
This repo contains code and published data for the AINL2023 paper [IMAD: IMage Augmented multi-modal Dialogue](link.com).

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

Classification is performed with model from [models directory](../main/models/). Example of usage is shown at [Text Replacing Tutorial](main/TextReplacingTest.ipynb)
