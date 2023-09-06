import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np 
from codes.utils import create_set

ANNOTATIONS_FILE = "../TUT-sound-events-2017-development/meta.txt"
COLUMNS = ["filenames","ambient","start","end","class","mixture","files"]
GROUNDTRUTH = pd.read_csv(ANNOTATIONS_FILE, sep="	",names=COLUMNS)
FILENAMES = np.unique(GROUNDTRUTH.filenames)

GROUNDTRUTH.drop(['ambient','mixture','files'],axis=1,inplace=True)

files_train,files_test = train_test_split(FILENAMES, test_size=0.2)
train = create_set(files_train, GROUNDTRUTH)
test = create_set(files_test, GROUNDTRUTH)
train.to_csv("datos/tut__train_set.txt",sep =" ",index=False,header=False)
test.to_csv("datos/tut__test_set.txt",sep =" ",index=False,header=False)