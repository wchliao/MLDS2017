# Homework 2: Video Caption Generation

## Core Technique

Seq2seq & Attention

## Introduction

Given a short video, please generate a caption that is appropriate with the video.

For more details, please refer to the [spec](https://docs.google.com/presentation/d/1OtD_BD6_Ljvr3aqLjHnnNX_h55BirD3cxhExq9wySmI/edit?usp=sharing).

## Models

There are 3 kinds of models.

They are seq2seq model, seq2seq + attention model, and s2vt model respectively.

## Usage

``` bash
$ python ./src/[model file] [testing id file] [testing feature path] [--train] [--test]
```

* `[model file]`: `s2s.py` for seq2seq model, `attention.py` for seq2seq + attention model, and `s2vt.py` for s2vt mdoel
* `[testing id file]`: the path of `testing_id.csv`
* `[testing feature path]`: the path of `testing_data/feat/`
* `[--train]` & `[--test]`: specifies the program to run training/testing

