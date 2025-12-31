# NOTE: THIS IS WORK IN PROGRESS, MANY THINGS ARE UNSUPPORTED OR SLIGHTLY BUGGY.

This repository contains an extension for training parametric models (models that, in addition to input audio data, can take a set of parameter values, emulating knobs on the analog gear).
The setup is fairly simple;

First, your data.json file needs a couple of modifications.
You need to put ```"type" : "parametric"``` in your file to denote that this is a parametric dataset.
Next, instead of having a single ```"y_path"```, this is instead put into a ```"data"``` array where each entry contains the ```"y_path"``` for that capture, as well as the parameter values used during that capture.
The assumption here is that the analog device is captured multiple times with the knobs at different positions.
All of this data is then fed into the model during training.
You can see an example data file at ```extensions/examples/example_data.json```

Second, your model.json needs a couple of modifications.
You need to set the type to ```"ParametricX"``` where ```X``` is the name of the original architecture. Currently only ```"ParametricLSTM"``` is supported.
Next, you need to set the ```"input_size"``` on the model - This is 1 + the number of parameters you want to model. So, if you want to model an overdrive model with the "Drive" and "Tone" knobs, the input size would be 3 - 1 (for the audio) + 2 (for the two knobs).
You can see an example model file at ```extensions/examples/example_model_lstm.json```

The learning.json file you can use exactly as before. (:

The code for loading these models at runtime are currently under construction and unavailable. At some point in the future I would like to make a plugin that can load parametric models and automatically add controls for the different parameters, but this might be a lot of time in the future, but this repository handles the hard part - Training the models.

# NAM: Neural Amp Modeler

[![Build](https://github.com/sdatkinson/neural-amp-modeler/actions/workflows/python-package.yml/badge.svg)](https://github.com/sdatkinson/neural-amp-modeler/actions/workflows/python-package.yml)

This repository handles training, reamping, and exporting the weights of a model.
For playing trained models in real time in a standalone application or plugin, see the partner repo,
[NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

For more information about the NAM ecosystem please check out https://www.neuralampmodeler.com/.

## Documentation
Online documentation can be found here: 
https://neural-amp-modeler.readthedocs.io

To build the documentation locally on a Linux system:
```bash
cd docs
make html
```

Or on Windows,
```
cd docs
make.bat html
```
