# Trabajo Fin de Grado - Ingeniería de Software

Facial expressions detection using Keras model and Haar Cascade classifier. Trained with FERPlus and AffectNet with followinf results:

| Number of classes | Accuracy         |
| ----------------- |:----------------:|
| 3 classes         | 86%              |
| 5 classes         | 74%              |

# Requirements
You can install the required packages directly from the [requirements.txt](./requirements.txt) file using the following command:

	pip install -r requirements.txt

You also need the following:
- Python 3.10 (will not work with older versions)
- Webcam for Camera mode
- Linux, macOS or Windows operative system


# How to run

Use this command for help:

	python3 run.py --help

It will output the next options: 

- **-h, --help:** show this help message and exit
- **--class-mode {basic,extra}:** Number of classes to detect. Basic mode classifies 3 diferent emotions with 87% of accuracy while Extra mode classifies 5 different classes with 74% of accuracy.
- **--input-mode {webcam,screen}:** Input source to detect faces.
- **--background, --no-background:** Execute in background without window.

