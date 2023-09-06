import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np 
from codes.utils import create_set

ANNOTATIONS_FILE = "datos/1608.txt"
COLUMNS = ["filenames","start","end","class"]
GROUNDTRUTH = pd.read_csv(ANNOTATIONS_FILE, sep=",",names=COLUMNS)
FILENAMES = np.unique(GROUNDTRUTH.filenames)

files_train,files_test = train_test_split(FILENAMES, test_size=0.2)
train = create_set(files_train, GROUNDTRUTH)
test = create_set(files_test, GROUNDTRUTH)
train.to_csv("datos/train_set.txt",sep =" ",index=False,header=False)
test.to_csv("datos/test_set.txt",sep =" ",index=False,header=False)