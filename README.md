# [Noise2Atom: Unsupervised Denoising for Scanning Transmission Electron Microscopy Images](http://fengwang.github.io/noise2atom/#/)

----

## [Noise2Atom](http://fengwang.github.io/noise2atom)

## Requirements:

- Python 3.8.5
- Tensorflow 1.14 __important__
- opencv 4.4.0
- python-imageio 2.8.0
- python-numpy 1.19.1
- python-tifffile 2020.7.24
- python-pathos 0.2.3


We use [a telegram bot](https://core.telegram.org/bots) to monitor the real time training process.
The private key and private chat id in file `code/message.py` should be updated before training.

## Denoising on your own dataset

- Simulating Gaussian-like atomic images by using routine implemented in file `code/simulate_physical_model.py`.
- Config then execute the training routine implemented in file `code/train.py`


## Cite us

```
@article{wang_noise2atom_2020,
	title = {{Noise2Atom}: unsupervised denoising for scanning transmission electron microscopy images},
	volume = {50},
	copyright = {All rights reserved},
	issn = {2287-4445},
	shorttitle = {{Noise2Atom}},
	url = {https://doi.org/10.1186/s42649-020-00041-8},
	doi = {10.1186/s42649-020-00041-8},
	language = {en},
	number = {1},
	urldate = {2020-10-23},
	journal = {Applied Microscopy},
	author = {Wang, Feng and Henninen, Trond R. and Keller, Debora and Erni, Rolf},
	month = oct,
	year = {2020},
	pages = {23}
}
```


## License

GNU AGPLv3

