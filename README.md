# KANs in HEP

This repository contains the code used for the following paper:

```bibtex
@article{Erdmann:2024unt,
    author = {Erdmann, Johannes and Mausolf, Florian and Sp{\"a}h, Jan Lukas},
    title = "{KAN We Improve on HEP Classification Tasks? Kolmogorov{\textendash}Arnold Networks Applied to an LHC Physics Example}",
    eprint = "2408.02743",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1007/s41781-025-00138-3",
    journal = "Comput. Softw. Big Sci.",
    volume = "9",
    number = "1",
    pages = "9",
    year = "2025"
}
```

The samples were produced with MadGraph, Pythia8 and Delphes using [MCGenMGPyDelphes](https://github.com/FloMau/MCGenMGPyDelphes).

The Python environment used for this work is listed in [requirements.txt](requirements.txt).

The main scripts are:

- [DelphesAnalyser.py](DelphesAnalyser.py) defines the event selection for the samples.
- [prepareML_Delphes.py](prepareML_Delphes.py) computes the physics observables and saves .csv files for training, validation and testing in `ML_datasets/`.
- [trainKAN.py](trainKAN.py) and [trainMLP.py](trainMLP.py) train Kolmogorov–Arnold networks (KANs) and multilayer perceptrons (MLPs), respectively, using JSON configuration files in `/configs`.
- [evaluateModel.py](evaluateModel.py) evaluates the trained models on the test samples and compares ROC curves, AUC values and background rejection.
- [modules/](modules/) contains shared utilities for training and plotting.

Adjust the sample and model paths in the scripts to your local setup.
