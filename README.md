# SF2935_music_classification
Music Classification Spotify API - Project SF2935 HT22
This repository contains the code for the project work in the class SF2935: Modern Methods of Statitical Learning at KTH.

## Goal

The group needed to train different machine learning models to classify songs accoding to whether our professor would like them or not.

## Data

The training dataset contained high-level features of the songs (e.g danceability, loudness, valence, tempo, etc.), available via Spotify's API.
The testing dataset was given at a later time.

## Models

The group choose 3 models from different parts of the ML-landscape:
- LDA/QDA
- Neural Networks
- Ensemble methods (LightGBM)

The implementation of the models can be found in the respective folders.

## Evaluation

Standard cross-validation was used for every model, only evaluating the accuracy score, because the dataset was balanced, and there was no importance on which class to predict more accurately (which could have prompted the use of different metrics).

The accuracy scores of the final models on the test data are 65.3, 65.3 and 66.7 percent, respectively.
