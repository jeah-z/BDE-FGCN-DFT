import pandas as pd
import numpy as np
import argparse
import pickle
import os
import rdkit
import rdkit
from rdkit import Chem
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset from literature", default='')
parser.add_argument("--normalization", help="if normalize the value", type=int, default=1)
parser.add_argument("--loadLiteratureData", help="if load literature data", type=int, default=1)
args = parser.parse_args()
dataset = args.dataset
datasetPath=Path(dataset)
normalization = args.normalization
loadLiteratureData=args.loadLiteratureData

validation_split = .2
shuffle_dataset = True
random_seed = 42
modelList = [200, 1000, 5000, 10000, 50000, 200000]



def load_obj(name):
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f)
def save_obj(obj, name):
    # os.system('mkdir obj')
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
if loadLiteratureData:
    # obj=args.dataset
    dataDf=load_obj(dataset)
    # print(list(df.keys()))
    # print(list(df['train_df'].keys()))
    # print(list(df['train_df'].loc[0]))
    # print(len(df['train_df'])
    trainDf = dataDf['train_df']
    print(trainDf.loc[0])
    smi_id_list = []
    for index in trainDf.index:
        # if index > 100:
        #     break
        # smi = self.target.loc[index]['SMILES']
        print(f'Processing mol NO: {index}')
        mol = trainDf.loc[index]['rdmol']
        atoms = mol.GetAtoms()
        nmrDic = trainDf.loc[index]['value'][0]
        print(f'nmrDic={nmrDic}')
        for atomIdx in nmrDic.keys():
            atomId = atomIdx
            atomType = atoms[atomIdx].GetSymbol()
            assert atomType in ['C', 'H']
            print(f'atomType={atomType}, atomId={atomIdx}')
            # atomType = self.target.loc[index]['type']
            exp_val = nmrDic[atomIdx]
            smi_id_list.append([mol, atomId, exp_val, atomType])
    nmrDf_full = pd.DataFrame(smi_id_list, columns=['mol', 'atomId', 'exp_val', 'atomType'])
    save_obj(nmrDf_full,  dataset+ "_full.pkl") 
nmrDf_full = load_obj(dataset + "_full.pkl")
print(f'lennmrDfFull={len(nmrDf_full)}')
for model in modelList:
    
    os.system(f'mkdir ./model_{model}')
    savePath=Path(f'model_{model}') 
    savePath=savePath.joinpath(datasetPath.stem)
    # savePath=datasetPath.parent.joinpath(f'model_{model}')
    nmrDf = nmrDf_full.sample(model)
    nmrDf = pd.DataFrame(nmrDf)
    print(nmrDf)

    dataset_size=len(nmrDf)
    invalid_id = []
    small_id = []
    for i in nmrDf.index:        
        mol = nmrDf.loc[i]['mol']
        try:
            # mol = Chem.MolFromSmiles(smi)
            AllChem.Compute2DCoords(mol)
            atoms = mol.GetAtoms()
            natom = len(atoms)
            if natom <= 5:
                small_id.append(i)

        except:
            print(smi + "was not valid SMILES\n")
            invalid_id.append(i)

    tmp_csv = nmrDf.copy(deep=True)
    input_csv=nmrDf
    train_csv = tmp_csv.loc[small_id]
    input_csv.drop(labels=invalid_id+small_id, axis=0, inplace=True)
    tmp_index = list(range(len(input_csv)))
    input_csv.index = tmp_index

    # Creating data indices for training and validation splits:
    dataset_size = len(input_csv)
    print('dataset_size= %s' % (dataset_size))
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size*0.5))
    # split2 = int(np.floor(validation_split * dataset_size*0.5))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    test_indices, train_indices, val_indices,  = indices[:
                                                        split], indices[split:-split], indices[-split:]
        # print(train_indices)

    # Creating PT data samplers and loaders:
    train_sampler = input_csv.loc[train_indices]
    valid_sampler = input_csv.loc[val_indices]
    test_sampler = input_csv.loc[test_indices]
    train_sampler = pd.concat([train_csv, train_sampler])
    train_target_mean = train_sampler['exp_val'].mean()
    train_target_std = train_sampler['exp_val'].std()

    if normalization > 0:
        train_sampler['exp_val'] = (
            train_sampler['exp_val'] - train_target_mean) / train_target_std

        valid_sampler['exp_val'] = (
            valid_sampler['exp_val'] - train_target_mean) / train_target_std
            
        test_sampler['exp_val'] = (
            test_sampler['exp_val']-train_target_mean)/train_target_std

        # train_target_mean = train_sampler['NMR'].mean()
        # train_target_std = train_sampler['NMR'].std()
        # train_sampler['NMR'] = (
        #     train_sampler['NMR'] - train_target_mean) / train_target_std

        # valid_sampler['NMR'] = (valid_sampler['NMR']-train_target_mean)/train_target_std

        mean_file = open(str(savePath) + '_mean_std.txt', 'w')
        # mean_file.writelines('train_parm_mean= %s\n' % (train_parm_mean))
        # mean_file.writelines('train_parm_std= %s\n' % (train_parm_std))
        mean_file.writelines('train_target_mean= %s\n' % (train_target_mean))
        mean_file.writelines('train_target_std= %s' % (train_target_std))

    print(f'The total train dataset size: {len(train_sampler)}')
    print(f'The total validation dataset size: {len(valid_sampler)}')
    print(f'The total test dataset size: {len(test_sampler)}')
    save_obj(train_sampler, str(savePath) + "_train")
    save_obj(valid_sampler, str(savePath) + "_valid")
    save_obj(test_sampler, str(savePath) + "_test")
        

        # train_sampler.to_csv(dataset+"_train.csv", index=False)
        # valid_sampler.to_csv(dataset + "_valid.csv", index=False)
        # train_sampler.to_csv(dataset+"_train.csv", index=False)