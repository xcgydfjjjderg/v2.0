
## Installation

```python setup.py install```

## Requirements

  * Python 3.7
  * TensorFlow (2.0)
  * pandas


## Usage

To reproduce the experiments mentioned in the paper you can run the following commands:


**Douban**
```bash
python train.py -d douban --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing 
```

**Flixster**
```bash
python train.py -d flixster --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing
```

**Yahoo Music**
```bash
python train.py -d yahoo_music --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing
```
##Description
```
graph autoencoders on Ml-100k folder based on ML-100k dataset,you run the train.py file directly.graph autoencoder on three datesets folder based on Douban,Flixster,and Yahoo Music datasets,you run above mentioned commands.All datasets are already in the corresponding folder.
```


