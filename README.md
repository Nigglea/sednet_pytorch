# Polyphonic Sound event detection network implemented in Pytorch
Repository to host a degree project for Acoustic Civil Engineering & MsC in Informatics.

## Database
pending.

## Feature Extraction

In order to extract information we will use for the first instance, the logarithmic Mel spectrogram.

## Neural network

- The neural network used in ```12_mbe_train.py``` and ```92_tut_train.py``` was Polyphonic-SEDnet designed by Adavanne et al.(2017). 

- The neural network used in ```22_stai_train.py```was a Custom SEDnet, modified from the SEDnet by Adavanne et al. (2017)

- The neural network used in ```32_staimbe_train.py```was a STAIMBEnet, a modified architecture from the SEDnet by Adavanne et al. (2017)

## Virtual environment

|Library|Version|
|-------|-------|
|python|3.9.5|
|pytorch|1.9.0|
|torchaudio|0.9.0|
|numpy|1.20.3|
|scipy|1.7.0|
|matplotlib|3.3.4|
|scikit-learn|0.24.2|
|tensorboard|0.6.0|

## Bibliography

1. Adavanne, S.; Virtanen, T. A report on sound event detection with different binaural features. In Proceedings of the Sound Event Detection in the DCASE 2017 challenge, IEEE/ACM Transactions on Audio, Speech, and Language Processing, Munich, Germany, 16 November 2017; pp. 1–4.
2. Poblete, V.; Espejo, D.; Vargas, V.; Otondo, F.; Huijse, P. Characterization of Sonic Events Present in Natural-Urban Hybrid Habitats Using UMAP and SEDnet: The Case of the Urban Wetlands. Appl. Sci. 2021, 11, 8175. https://doi.org/10.3390/app11178175
3. Espejo, D.; Vargas, V.; Viveros-Muñoz, R.; Labra, F. A.; Huijse, P., & Poblete, V. (2024). Short-time acoustic indices for monitoring urban-natural environments using artificial neural networks. Ecological Indicators, 160. https://doi.org/10.1016/j.ecolind.2024.111775

## Author

- diego dot espejo at alumnos dot uach dot cl // diego dot espejo at aumilab dot cl
- vpoblete at uach dot cl
