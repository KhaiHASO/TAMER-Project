#!/bin/bash
version=$1
# 
python eval/test.py data/HME100k $version testb 320000 True

