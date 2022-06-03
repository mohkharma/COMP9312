# from datasets import load_dataset
#
# dataset = load_dataset("banking77")
#
# print (dataset.shape)

import pandas as pd
from sklearn.model_selection import train_test_split



# load the iris dataset and get X and Y data

data = pd.read_csv("/data/banking/train-all.csv")
df = pd.DataFrame(data)

train,test = train_test_split(df, test_size=0.20, random_state=0)

train,val = train_test_split(train, test_size=0.10, random_state=0)


train.to_csv('train-all.csv',index=False)
test.to_csv('test-original.csv',index=False)
val.to_csv('val.csv',index=False)