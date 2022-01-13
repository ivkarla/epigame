## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Data preparation](#data-preparation)
* [Analysis guide](#analysis-guide)

## General info
Epigame is a project for epileptogenic network definition. 

The project is created with:
* Python version: 3.7.1
	
## Setup
To run this project, use conda package manager.
To install conda, refer to: https://docs.conda.io/en/latest/miniconda.html

Clone the epigame repository to your local machine.
Open the Anaconda Prompt. Navigate to the epigame folder and set up the virtual environment:
```
$ cd epigame
$ conda create --name epigame --file environment.yml
```

## Data preparation
1. Extract the SEEG data of a single seizure as an EDF+ file. The recording should be 10 minutes long, ending at seizure end.
2. Place the EDF+ files in the _data_ folder.

## Analysis guide

To analyze the data, run the following command:

````
$ python main_pipeline.py
````

You will be asked to input the time window, connectivity method, and the frequency band of interest. 

_Example: Analysis of transition to seizure by spectral coherence (imaginary part) in low gamma band (30-70 Hz)._

````
Time window:
1. Non-seizure (baseline)
2. Pre-seizure
3. Transition to seizure
4. Seizure
Indicate a number: 3

Connectivity method:
1. PEC
2. Spectral Coherence
3. Phase Lock Value
4. Phase-Lag Index
5. Cross-correlation
6. Phase-amplitude coupling
Indicate a number: 2

Imaginary part (Y/N): Y

Filter the signal (Y/N): Y

Set band range min: 30
Set band range max: 70
````

The preprocessed epochs are saved in _"../results/preprocessed"_.
The identified epileptogenic networks are saved in _"../results/EN"_.