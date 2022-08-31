# DSTA Brainhack 2022 TIL AI Hackathon [Team KEIWUAI20OFF]

Champions (University Category)



## Directory

* ``src/``: SDK and simulator packages.
* ``config/``: SDK and simulator sample configuration files.
* ``data/``: Simulator sample data.
* ``docs/``: Documentation source.
* ``stubs/``: Code stubs for participants.
* ``notebooks/``: Training notebooks for qualifying round


## Install

```sh
pip install .
```

Robomaster SDK will need to be installed separately.

## Build the documentation

Most of the information is available in the Sphinx docs.

```sh
# install dependencies
pip install sphinx sphinx-autoapi sphinx-rtd-theme

# build docs
sphinx-build -b html docs/source docs/build
```

Access the docs at `docs/build/index.html`.

## Get model weights

Download model weights from [gdrive](https://drive.google.com/drive/folders/1q60Qoj_65Og3Nk3u_pDXQ70flAj7vEXx?usp=sharing) and place into ``data/models/``


## Run simulator
```
til-simulator -c config/sim_config.yml
```

## Run robot script
```
python ./stubs/autonomy.py
```
