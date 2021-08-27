from os import listdir
from pandas_plink import read_plink
from pandas import read_csv
from re import search
from sklearn.model_selection import train_test_split
from .thresholding import causality_table

from numpy import save, load
from tensorflow.keras.utils import to_categorical

def sorted_listdir(path):
    key = lambda s: int(search("\d+", s).group())
    list = listdir(path)
    list.sort(key=key)
    return list

def get_causals_file(path):
    for d in listdir(path):
        if d.startswith(r"SNP_NrSNP"):
            return d

def get_Ysim_file(path):
    for d in listdir(path):
        if d.startswith(r"Ysim_") and d.endswith(".csv"):
            return d

def get_dset(path, generate_ct=True):
    has_ct = False
    has_split = False
    
    dset = {}

    for file in listdir(path):
        if file.startswith("ct"):
            has_ct = True
        if file == "X_train.npy":
            has_split = True

    if not has_split:
        generate_split(path)

    X_train = load(f"{path}/X_train.npy", allow_pickle=True)
    y_train = to_categorical(load(f"{path}/y_train.npy", allow_pickle=True))

    X_test = load(f"{path}/X_test.npy", allow_pickle=True)
    y_test = to_categorical(load(f"{path}/y_test.npy", allow_pickle=True))

    if not has_ct:
        bim, fam, bed = read_plink(path + "/Genotypes.bed", verbose=False)
        #check if SNPs are named 
        if not bim["snp"][0].startswith("SNP_"):
            dset["ct"] = causality_table(
                bed.astype("int8").T, 
                read_csv(path + "/" + get_Ysim_file(path))["Trait_1"].values > 0, 
                list(read_csv(path + "/" + get_causals_file(path)).columns[1:].to_numpy()),
                names = bim["snp"]
                )   
        else:
            dset["ct"] = causality_table(
                bed.astype("int8").T, 
                read_csv(path + "/" + get_Ysim_file(path))["Trait_1"].values > 0, 
                list(read_csv(path + "/" + get_causals_file(path)).columns[1:].to_numpy())
                )   
        dset["ct"].to_csv(f"{path}/ct.csv")
    
    ct = read_csv(f"{path}/ct.csv")

    dset = ({
    "X_train": X_train,
    "y_train": y_train,
    "X_test": X_test,
    "y_test": y_test,
    "ct": ct,
    "true_causals": list(read_csv(path + "/" + get_causals_file(path)).columns[1:].to_numpy())
    })


    return dset
    
"""
def get_dsets(path):
    bim, fam, bed = read_plink(path + "/Genotypes.bed", verbose=False)


    has_ct = False
    for file in listdir(path):
        if file.startswith("ct"):
            has_ct = True

    if not has_ct:
        dset["ct"] = causality_table(dset["X"], dset["y"], dset["true_causals"])
        dset["ct"].to_csv(f"{path}/ct.csv")

    dset["ct"] = read_csv(path + "/ct.csv")
    return dset
"""



def generate_split(path):
    
    bim, fam, bed = read_plink(path + "/Genotypes.bed", verbose=False)

    dset = ({
    "X": bed.astype("int8").T,
    "y": read_csv(path + "/" + get_Ysim_file(path))["Trait_1"].values > 0,
    "true_causals": list(read_csv(path + "/" + get_causals_file(path)).columns[1:].to_numpy())
    })


    #dset["ct"] = read_csv(path + "/ct.csv")
    
    X_train, X_test, y_train, y_test = train_test_split(dset["X"].compute(), dset["y"], shuffle=True, test_size=0.2)
    
    save(f"{path}/X_train.npy", X_train)
    save(f"{path}/y_train.npy", y_train)
    
    save(f"{path}/X_test.npy", X_test)
    save(f"{path}/y_test.npy", y_test)
    

