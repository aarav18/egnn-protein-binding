import os
import shutil
import time

import numpy as np
import pandas as pd

import tempfile

import deepchem as dc
from deepchem.utils import download_url, load_from_disk
from deepchem.utils.docking_utils import prepare_inputs

from rdkit import Chem
from rdkit.Chem import AllChem

from egnn import EGNN

def load_raw_dataset(data_dir):
    dataset_file = os.path.join(data_dir, "pdbbind_core_df.csv.gz")
    
    # check if data folder exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # check if dataset has already been downloaded
    if not os.path.exists(dataset_file):
        download_url("https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/pdbbind_core_df.csv.gz", data_dir)
    
    return load_from_disk(dataset_file)

def process_raw_data(pdbids, ligand_smiles):
    for (pdbid, ligand) in zip(pdbids, ligand_smiles):
        p, m = None, None
        try:
            p, m = prepare_inputs(protein=pdbid, ligand=ligand, replace_nonstandard_residues=False, remove_heterogens=False, remove_water=False, add_hydrogens=False, pdb_name=f"{pdbid}")
        except:
            print(f"{pdbid} failed sanitization.")
    
    if p and m:
        Chem.rdmolfiles.MolToPDBFile(p, f"{pdbid}.pdb")
        Chem.rdmolfiles.MolToPDBFile(m, f"ligand_{pdbid}.pdb")
    
    # collect .pdb files into directory
    if not os.path.exists("pdbs"):
        os.makedirs("pdbs")
    
    for file in os.listdir("."):
        if file.endswith(".pdb"):
            shutil.move(file, os.path.join("pdbs", file))
    
    # ensure valid protein-ligand pairs
    proteins = [p for p in os.listdir("pdbs") if len(p) == 8]
    ligands = [l for l in os.listdir("pdbs") if l.startswith("ligand")]
    failures = set([p[:-4] for p in proteins]) - set([l[7:-4] for l in ligands])
    for pdbid in failures:
        proteins.remove(pdbid + ".pdb")
    
    return proteins, ligands

def featurize_data(proteins, ligands):
    pdbids = [p[:-4] for p in proteins]
    processed_dataset = raw_dataset[raw_dataset["pdb_id"].isin(pdbids)]
    labels = processed_dataset.label
    
    # fingerprint featurizer
    fp_featurizer = dc.feat.CircularFingerprint(size=2048)
    features = fp_featurizer.featurize([Chem.MolFromPDBFile(os.path.join("pdbs", l)) for l in ligands])
    
    dataset = dc.data.NumpyDataset(X=features, y=labels, ids=pdbids)
    
    return dataset
    
    
if __name__ == '__main__':
    # start timing for benchmarking
    start_time = time.time()
    
    # load the raw dataset
    raw_dataset: pd.DataFrame = load_raw_dataset("data")
    print(f"Loaded dataset: {time.time() - start_time}s")
    
    # filter to include only pdb_id, smiles, and label
    raw_dataset = raw_dataset[["pdb_id", "smiles", "label"]]
    
    # extract ids and smiles
    pdbids = raw_dataset["pdb_id"].values
    ligand_smiles = raw_dataset["smiles"].values
    
    # process the raw dataset
    proteins, ligands = process_raw_data(pdbids, ligand_smiles)
    print(f"Processed dataset: {time.time() - start_time}s")
    
    print(len(proteins))
    print(len(ligands))
    
    # featurize the processed dataset
    dataset = featurize_data(proteins, ligands)
    # print(f"Featurized dataset: {time.time() - start_time}s")
    
    # split dataset
    train_dataset, valid_dataset, test_dataset = dc.splits.RandomSplitter().split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42)
    # print(f"Split dataset: {time.time() - start_time}s")
    
    # ACTUAL TRAINING, VALIDATION, AND EVALUATION DONE ELSEWHERE