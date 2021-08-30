# BDE-FGCN-DFT
  Prediction of experimental bond dissociation energy and NMR chemical shift with GCN and QM descriptors

![image](https://github.com/jeah-z/BDE-FGCN-DFT/blob/main/images/FlowChart.png)
Fig. 1 Architecture of BDE-FGCN model

# BDE data format of input
Example:

SMILES,id1,id2,qm,bde,type   
CCC(C)CC(C)C,2,4,-0.030523501145157478,-0.02985865631274702,C-C

This code adopts SMILES as input with the index of two bonded atoms in the rdkit. Above atomic index usually is the same as the index in the SMILES string. 

Example:

SMILES,id1,id2,qm,bde,type      
CCC(C)C,2,2,0.3732913481927531,0.5532308091542087,C-H

If the the bond involve implicit hydrogen, users could input the heavy atom's index twice, this script will detect implicit hydrogen index automatically.

# NMR data format of input

Example:

SMILES,id,qm,nmr,type    
[C@H]1([C@H]([C@H]2CCCN2[C@@H]1CO)O)O,1,0.3783899996060892,-0.183985470513843,C

This code adopts SMILES as input with the index of atom in the rdkit. Above atomic index usually is the same as the index in the SMILES string. 

Example:

SMILES,id,qm,nmr,type    
c1ccc(cc1)N(=O)=O,1,7.605982509759489,1.304699548882766,H

If the NMR involves a implicit hydrogen bonded with a heavy atom, users could input the heavy atom's index, this script will detect implicit hydrogen index automatically.


# to train the model 
For the prediction of bond dissociation energy.
## Training of the low-dimensional dft model
```
python -u model/train.py --model bde --epochs 10000 --saveFreq 20 --train_file ./data/bde_CH_train.csv --valid_file ./data/bde_CH_valid.csv --test_file ./data/bde_CH_test.csv --device cuda:0 --save bde_CH_June7_dft
```
## Training of the high-dimensional gcn model that is based on the trained dft model
```
python -u model/train.py --model bde --epochs 10000 --saveFreq 20 --train_file ./data/bde_CH_train.csv --valid_file ./data/bde_CH_valid.csv --test_file ./data/bde_CH_test.csv --device cuda:0 --dft_model bde_CH_June7_dft/dft_model_5000 --save bde_CH_June7_gcn
```
For the prediction of NMR chemical shift.
## Training of the low-dimensional dft model
```
# nohup python -u model/train.py --model nmr --epochs 30000 --saveFreq 50 --train_file ./data/nmr_C_train.csv --valid_file ./data/nmr_C_valid.csv --test_file ./data/nmr_C_test.csv --device cuda:0 --save nmr_C_June7_dft  > nmr_C_June7_dft.txt 2>&1 &

```
## Training of the high-dimensional gcn model that is based on the trained dft model
```
python -u model/train.py --model nmr --epochs 10000 --saveFreq 20 --train_file ./data/nmr_C_train.csv --valid_file ./data/nmr_C_valid.csv --test_file ./data/nmr_C_test.csv --device cuda:0 --save nmr_C_June7_gcn  --dft_model nmr_C_June7_dft/dft_model_5000 
```
# dependency

- rdkit
- dgl
- pytorch
- python==3.6
- numpy 
- pandas
- zipfile
- os
- pathlib
- tqdm
- pathos
- argparse

# related repository

This code was based on https://github.com/tencent-alchemy/Alchemy. If this script is of any help to you, please cite them.

- K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.
SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) [link](http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions)  
```
- @article{chen2019alchemy,
  title={Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models},
  author={Chen, Guangyong and Chen, Pengfei and Hsieh, Chang-Yu and Lee, Chee-Kong and Liao, Benben and Liao, Renjie and Liu, Weiwen and Qiu, Jiezhong and Sun, Qiming and Tang, Jie and Zemel, Richard and Zhang, Shengyu},
  journal={arXiv preprint arXiv:1906.09427},
  year={2019}
}
```
