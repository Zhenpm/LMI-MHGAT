# LMI-MHGAT

LMI-MHGAT is a model used to predicted lncRNA-miRNA interactions. It not only enables cross species prediction, but also exhibits higher performance in imbalanced networks which are closer to reality.

![overview.png](overview.png)

# Dependencies

- TensorFlow 1.x


# Datasets

The example datasets can be obtained in folder data

- "data" folder includes all the data used for human LMIs.
  - main.txt : the experimental validated lncRNA-miRNA interactions.
  - extra.txt : the other six layers data, including lncRNA-lncRNA sequence similarity data, lncRNA-lncRNA co-expression data, miRNA-miRNA sequence similarity data, miRNA-miRNA co-expression data, miRNA-mRNA interaction data and lncRNA-mRNA interaction data.
  - valid_false_data_by_edge.txt : validation set negative samples.
  - testing_false_data_by_edge.txt : test set negative samples.

# Usage

#### **Applying existing datasets to LMI-MHGAT**

**to use human datasets**

```
python ./src/main.py --input ./data --eval-type 1 --kfold 5
```

main parameters introduction:

```powershell
--input ./data/human	# dataset
--epoch 100          	#the number of epochs for training, default is 100
--eval-type	1			#the edge types for evaluation, default is 1
--edge-dim 10			#the number of edge embedding dimensions, default is 10
--att-dim 20			#the number of attention dimensions, default is 20
--walk-length 10		#the length of walk per source, default is 10
--negative-samples 5 	#negative samples for optimization, default is 5
--neighbor-samples 10	#neighbor samples for aggregation, default is 10
--kfold	5				#k-fold cross validation, default is 5
--att-head 1			#the number of attention head
```

#### **Applying your own datasets to LMI-MHGAT**

```
python ./src/main.py --input ./datapath --eval-type 1 --kfold 5
```

to build your own datasets, we advise you to get all data needed and process them by R packages.

to get better effect, you can adjust the above parameters.
