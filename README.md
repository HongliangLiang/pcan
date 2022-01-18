## Introduction

Code for paper:

Path Context Augmented Statement and Network for Learning Programs

Da Xiao, Dengji Hang, Lu Ai, Shengping Li, Hongliang Liang

Empirical Software Engineering, 2022


## Environment

    dgl==0.5.3  
    python==3.8  
    torch==1.7.1+cu110  

## BCB Data

- Training Data：data/bcb/pairs.pkl （When start to training, it will be automatically pro-rated randomly into training sets and development sets）

    select about 100 million randomly from Raw BCB Data Set

- Testing Data：data/bcb/pairs.pkl 

- Pcan tree structure data (include training set, development set and test set) after data pre-processing：data/bcb/processed.pkl

- Testing Data construction method： 

Because more data in the positive example can be found in the Raw BCB data, the type is constructed based on the positive example data and the condition that the positive example data accounts for 14% of the total data. 

1. For example, if the positive sample of type = 1 is 15 and the negative sample of type = 1 is 2, then the total number of negative sample that need to be constructed is 15/0.14 = 107, then the number of negative sample that need to be constructed is 107- 15 - 2 = 95, we mark the type of 95 random samples labeled as negative sample that do not have a type found in the negative sample as 1. 

2. From 1 to 4 classes are all constructed as method 1.

3. Positive examples of Type = 4(16,889 can be found) < 124750 * 0.14, as described in this article, remain unchanged. Then, based on the data of 16889, we complete the negative examples,

4. Because have no enough positive data, so the total number is 122283 instead of 124750.

Fanal pre-processing BCB Data distribute follow as：

| Type |  T1   |  T2   | ST3 | MT3| T4 | total 
|  ----  | ----  | ---- | ---- | ---- | ----|----
|radio|  0.0026 | 0.00054  | 0.0020 | 0.0091 | 0.9858
|  all + | 442 | 93 | 342 | 1563 |169450 | 171890 
|  all - | 0 | 2 | 22 | 391 |262050 | 262465 
|  test + | 15 | 4 | 32 | 180 |16889 | 17120 
|  all - | 92 | 24 | 196 |1105 | 103746 | 105163


## Running

- Training

    python bcb_run.py --dataset_name bcb --epochs 1500 --nhead 4 --batch_size 64 --dropout_rate 0.2  --mode train

    model file：outputs/models/.*pt

- evaluate

    python bcb_run.py --dataset_name bcb --type 1 --batch_size 1 --mode eval --model_path ./outputs/models/model.pt


    evaluate result: outpus/eval_result.txt


Among,

dataset_name：name of data set

type：positive example type (from 1 to 5,  1 is the easiest and 5 is the most difficult) that you want to evaluate.

model_path: model file path







