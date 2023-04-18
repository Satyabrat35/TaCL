import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random

import time
import logging

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="bert-base-uncased")
    parser.add_argument("--sim", type=str, default="cosine")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--use_contrastive_loss", type=bool, default=True)
    pass