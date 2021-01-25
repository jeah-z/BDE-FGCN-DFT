import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem

normalization = 1
validation_split = .1
shuffle_dataset = True
random_seed = 42
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset to train")
parser.add_argument("--type", help="dataset type: bde or nmr")
parser.add_argument("--atomType", help="atomic type: C or H,all")
args = parser.parse_args()
dataset = args.dataset
datasetType = args.type

if datasetType == 'bde':
    input_csv = pd.read_csv(dataset + ".csv", skiprows=1,
                            names=['SMILES', 'id1', 'id2', 'qm', 'bde', 'type'])
    input_csv = input_csv[['SMILES', 'id1', 'id2', 'qm', 'bde', 'type']]
    dataset_size = len(input_csv)
    print(f'The total dataset size: {dataset_size}')
    invalid_id = []
    small_id = []
    for i in range(dataset_size):
        smi = input_csv.loc[i]['SMILES']
        try:
            mol = Chem.MolFromSmiles(smi)
            AllChem.Compute2DCoords(mol)
            atoms = mol.GetAtoms()
            natom = len(atoms)
            if natom <= 5:
                small_id.append(i)

        except:
            print(smi + "was not valid SMILES\n")
            invalid_id.append(i)

    tmp_csv = input_csv.copy(deep=True)
    train_csv = tmp_csv.iloc[small_id]
    input_csv.drop(labels=invalid_id+small_id, axis=0, inplace=True)
    tmp_index = list(range(len(input_csv)))
    input_csv.index = tmp_index
    # Creating data indices for training and validation splits:
    dataset_size = len(input_csv)
    print('dataset_size= %s' % (dataset_size))
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size*0.5))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[split:-
                                                       split], indices[:split], indices[-split:]
    # print(train_indices)

    # Creating PT data samplers and loaders:
    train_sampler = input_csv.loc[train_indices]
    train_sampler = pd.concat([train_csv, train_sampler])
    train_target_mean = train_sampler['bde'].mean()
    train_target_std = train_sampler['bde'].std()

    train_qm_mean = train_sampler['qm'].mean()
    train_qm_std = train_sampler['qm'].std()

    valid_sampler = input_csv.loc[val_indices]
    test_sampler = input_csv.loc[test_indices]
    if normalization > 0:
        train_sampler['bde'] = (
            train_sampler['bde'] - train_target_mean) / train_target_std
        valid_sampler['bde'] = (
            valid_sampler['bde'] - train_target_mean) / train_target_std
        test_sampler['bde'] = (
            test_sampler['bde'] - train_target_mean) / train_target_std

        train_sampler['qm'] = (
            train_sampler['qm'] - train_target_mean) / train_target_std
        valid_sampler['qm'] = (
            valid_sampler['qm'] - train_target_mean) / train_target_std
        test_sampler['qm'] = (
            test_sampler['qm'] - train_target_mean) / train_target_std

        mean_file = open(dataset + '_mean_std.txt', 'w')
        # mean_file.writelines('train_parm_mean= %s\n' % (train_parm_mean))
        # mean_file.writelines('train_parm_std= %s\n' % (train_parm_std))
        mean_file.writelines('train_bde_mean= %s\n' % (train_target_mean))
        mean_file.writelines('train_bde_std= %s\n' % (train_target_std))
        mean_file.writelines('train_qm_mean= %s\n' % (train_qm_mean))
        mean_file.writelines('train_qm_std= %s\n' % (train_qm_std))

    print(f'The total train dataset size: {len(train_sampler)}')
    print(f'The total validation dataset size: {len(valid_sampler)}')
    train_sampler.to_csv(dataset+"_train.csv", index=False)
    valid_sampler.to_csv(dataset+"_valid.csv", index=False)
    test_sampler.to_csv(dataset+"_test.csv", index=False)

if datasetType == 'nmr':
    input_csv = pd.read_csv(dataset + ".csv", skiprows=1,
                            names=['SMILES', 'id', 'qm', 'nmr', 'type'])
    input_csv = input_csv[['SMILES', 'id', 'qm', 'nmr', 'type']]
    print(f'Dataset size before dropping duplicates: {len(input_csv)}')
    input_csv = input_csv.drop_duplicates()
    print(f'Dataset size after dropping duplicates: {len(input_csv)}')
    dataset_size = len(input_csv)
    print(f'The total dataset size: {dataset_size}')
    tmp_index = list(range(len(input_csv)))
    input_csv.index = tmp_index
    invalid_id = []
    small_id = []
    for i in range(dataset_size):
        smi = input_csv.loc[i]['SMILES']
        nmrType = input_csv.loc[i]['type']
        try:
            mol = Chem.MolFromSmiles(smi)
            heavyAtoms = mol.GetAtoms()
            heavyLen = len(heavyAtoms)
            mol = Chem.rdmolops.AddHs(mol)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            AllChem.Compute2DCoords(mol)
            atoms = mol.GetAtoms()
            natom = len(atoms)
            atomId = int(input_csv.loc[i]['id'])
            atomSymbol = atoms[int(atomId)].GetSymbol()

            if nmrType == 'H':
                print('Implicit H model!')
                print(f'smi={smi}')
                H_id = ''
                atom = atoms[atomId]
                bonds = atom.GetBonds()
                neighb = []
                for bond in bonds:
                    idxBegin = bond.GetBeginAtomIdx()
                    idxEnd = bond.GetEndAtomIdx()
                    neighb.append(idxBegin)
                    neighb.append(idxEnd)

                neighb = list(set(neighb))
                print(f'neighb={neighb}')
                for idx in neighb:
                    atom = atoms[idx]
                    if atom.GetAtomicNum() == 1:
                        # print(idx)
                        H_id = idx
                        # print("H_id= %s" % (H_id))
                        break
                if H_id == '':
                    print(smi + "No bonded H was found, please check the input!")
                    print(f'atomID: {atomId}, Symbol:{atomSymbol}')
                    invalid_id.append(i)
                    continue
                    # sys.exit(0)
                print(f'{idx} -> {H_id}')

            else:
                assert atomSymbol == nmrType
            if heavyLen <= 5:
                small_id.append(i)

        except Exception as e:
            print(smi + "was not valid SMILES\n")
            print(f'atomID: {atomId}, Symbol:{atomSymbol}')
            invalid_id.append(i)

    tmp_csv = input_csv.copy(deep=True)
    train_csv = tmp_csv.iloc[small_id]
    input_csv.drop(labels=invalid_id+small_id, axis=0, inplace=True)
    tmp_index = list(range(len(input_csv)))
    input_csv.index = tmp_index
    # Creating data indices for training and validation splits:
    dataset_size = len(input_csv)
    print('dataset_size= %s' % (dataset_size))
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size*0.5))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[split:-
                                                       split], indices[:split], indices[-split:]
    # print(train_indices)

    # Creating PT data samplers and loaders:
    train_sampler = input_csv.loc[train_indices]
    train_sampler = pd.concat([train_csv, train_sampler])
    train_target_mean = train_sampler['nmr'].mean()
    train_target_std = train_sampler['nmr'].std()

    train_qm_mean = train_sampler['qm'].mean()
    train_qm_std = train_sampler['qm'].std()

    valid_sampler = input_csv.loc[val_indices]
    test_sampler = input_csv.loc[test_indices]
    if normalization > 0:
        train_sampler['nmr'] = (
            train_sampler['nmr'] - train_target_mean) / train_target_std
        valid_sampler['nmr'] = (
            valid_sampler['nmr'] - train_target_mean) / train_target_std
        test_sampler['nmr'] = (
            test_sampler['nmr'] - train_target_mean) / train_target_std

        train_sampler['qm'] = (
            train_sampler['qm'] - train_target_mean) / train_target_std
        valid_sampler['qm'] = (
            valid_sampler['qm'] - train_target_mean) / train_target_std
        test_sampler['qm'] = (
            test_sampler['qm'] - train_target_mean) / train_target_std

        mean_file = open(dataset + '_mean_std.txt', 'w')
        # mean_file.writelines('train_parm_mean= %s\n' % (train_parm_mean))
        # mean_file.writelines('train_parm_std= %s\n' % (train_parm_std))
        mean_file.writelines('train_nmr_mean= %s\n' % (train_target_mean))
        mean_file.writelines('train_nmr_std= %s\n' % (train_target_std))
        mean_file.writelines('train_qm_mean= %s\n' % (train_qm_mean))
        mean_file.writelines('train_qm_std= %s\n' % (train_qm_std))

    print(f'The total train dataset size: {len(train_sampler)}')
    print(f'The total validation dataset size: {len(valid_sampler)}')
    train_sampler.to_csv(dataset+"_train.csv", index=False)
    valid_sampler.to_csv(dataset+"_valid.csv", index=False)
    test_sampler.to_csv(dataset+"_test.csv", index=False)
