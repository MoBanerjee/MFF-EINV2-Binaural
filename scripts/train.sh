#!/bin/bash

GPU_ID=2

CUDA_VISIBLE_DEVICES=$GPU_ID python code/main.py train --port=12360