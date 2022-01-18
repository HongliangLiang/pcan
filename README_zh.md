


## 运行环境

    dgl==0.5.3  
    python==3.8  
    torch==1.7.1+cu110  

## 数据

- 训练数据：data/bcb/pairs.pkl （训练时会自动按比例进行随机切分为训练集和开发集）

- 测试数据：data/bcb/pairs.pkl 

- 通过数据预处理之后的pcan树结构数据（训练集，开发集和测试集）：data/bcb/processed.pkl

- 训练数据： 从原始bcb数据集中随机挑选了约100万

- 测试数据构造逻辑：

因为正例中有较多的数据可以从源码数据中找到type，因此，基于正例数据以及正例数据占总数据的14%这个条件进行构造。

1、比如type=1的正例为15个，type=1的负例为2个，那么需要总的type=1的数据为15/0.14=107,   则，需要构造的负例数量=107 - 15 - 2 = 95；那么此时，我们把负例中没有找到类型的随机抽取95条标注为负例的type标记为1。

2、1-4类均如1方法构建

3、type4的正例（能找到16889条）<文章所述124750 * 0.14,，维持原样。然后基于16889的数据对负例进行补齐，

4、因为没有补齐正例，因此目前总数据为122283条

最终的数据比例如下：

| 类型 |  T1   |  T2   | ST3 | MT3| T4 | total 
|  ----  | ----  | ---- | ---- | ---- | ----|----
|占比|  0.0026 | 0.00054  | 0.0020 | 0.0091 | 0.9858
|  all + | 442 | 93 | 342 | 1563 |169450 | 171890 
|  all - | 0 | 2 | 22 | 391 |262050 | 262465 
|  test + | 15 | 4 | 32 | 180 |16889 | 17120 
|  all - | 92 | 24 | 196 |1105 | 103746 | 105163


## 运行

- 训练

    python bcb_run.py --dataset_name bcb --epochs 1500 --nhead 4 --batch_size 64 --dropout_rate 0.2  --mode train

    模型结果：outputs/models/.*pt

- 评测

    python bcb_run.py --dataset_name bcb --type 1 --batch_size 1 --mode eval --model_path ./outputs/models/model.pt


    评测是结果: outpus/eval_result.txt


其中,

dataset_name：数据集名称

type：评测正例中的数据类型，按难易程度有1,2,3,4,5种类型，1最简单

model_path: 模型路径







