# Lab 1 Oliver Dantzer

## Create virtual environment

```bash
python -m venv venv
```

## Enter the virtual environment

Linux

```bash
source ./venv/bin/activate
```

Windows

```bash
source ./venv/Scripts/activate
```

## Install requirements

```bash
pip install -r requirements.txt
```

## Download dataset

```bash
python src/download_dataset.py
```

## visualize training data

```bash
python src/interactive_visualizer.py  # without augmentations
python src/interactive_visualizer.py -a  # with augmentations
```
