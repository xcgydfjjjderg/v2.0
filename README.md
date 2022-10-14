
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
##Dataset Description
```
Because uploading to github has a file size limit, all four datasets were deleted. Readers can use the code to download the relevant datasets by themselves.
```


