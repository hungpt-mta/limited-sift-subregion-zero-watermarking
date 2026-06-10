import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

IMG_SIZE = 512
EPS = 1e-12

ARNOLD_A = 1
ARNOLD_B = 1
ARNOLD_ITERS = 10

np.random.seed(0)

TIMING_REPEATS = 1  # keep as in your script
