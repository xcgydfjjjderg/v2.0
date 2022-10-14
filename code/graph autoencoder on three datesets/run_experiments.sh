#!/bin/bash

# Douban
python train.py -d douban --accum stack -do 0.7 -nleft -nb 2 -e 200 --features  --feat_hidden 64 --testing > douban_testing.txt  2>&1

# Flixster
python train.py -d flixster --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing > flixster_testing.txt  2>&1

# Yahoo Music
python train.py -d yahoo_music --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing > yahoo_music_testing.txt  2>&1



