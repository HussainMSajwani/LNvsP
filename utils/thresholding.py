from tqdm.auto import tqdm, trange
from statsmodels.api import Logit, tools
from pandas import read_csv, crosstab, DataFrame
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score
from numpy import arange, log10, sum, abs, max, min
from multiprocessing import Pool
from functools import partial
from os import cpu_count

def logit_pvals_(X, y):
    out = []
    for snp in trange(X.shape[1], desc = "Generating logistic p values"):
        lr = Logit(y, tools.add_constant(X[:, snp])).fit(disp=0, method="bfgs")
        p = lr.pvalues[1]
        out.append(p if not p == 0 else 1e-100)
    return out

def single_pval(i, X, y):
    lr = Logit(y, tools.add_constant(X[:, i])).fit(disp=0, method="bfgs") 
    pvals = lr.pvalues
    if len(pvals) != 2:
        p = 1
    else:
        p = pvals[1]
    return p if not p == 0 else 1e-100

def logit_pvals(X, y):
    out = []
    #P = Pool(int(cpu_count()/2 - 1))
    pvals_iterable = map(partial(single_pval, X=X, y=y), range(X.shape[1]))

    with tqdm(total = X.shape[1], desc="generating p values") as pbar:
        for p_val in pvals_iterable:
            out.append(p_val)
            pbar.update()

    #P.close()
    #P.join()
    return out


def chi2_pvals(X, y):
    out = []
    for snp in trange(X.shape[1], desc = "Generating chi2 p values"):
        p = chi2_contingency(crosstab(X[:, snp], y))[1]
        out.append(p if not p == 0 else 1e-100)
    return out


def causality_table(X, y, causal_snps, names = []):
    if len(names) == 0:
        table = DataFrame({
            "SNP": ["SNP_" + str(snp+1) for snp in range(X.shape[1])],
            "causal": [1 if "SNP_" + str(snp+1) in causal_snps else 0 for snp in range(X.shape[1])],
            "logistic_p": logit_pvals(X, y),
    #        "chi2_p": chi2_pvals(X, y)
        })
    else:
        table = DataFrame({
            "SNP": names,
            "causal": [1 if snp in causal_snps else 0 for snp in names],
            "logistic_p": logit_pvals(X, y),
    #        "chi2_p": chi2_pvals(X, y)
        })  
    return table


def thresholding_recall(thresh, ct, logistic=1):
    method = "logistic" if logistic else "chi2"    
    return recall_score(ct["causal"], ct[f"{method}_p"] < thresh, zero_division=0)


def thresholding_precision(thresh, ct, logistic=1):
    method = "logistic" if logistic else "chi2" 
    return precision_score(ct["causal"], ct[f"{method}_p"] < thresh, zero_division=1)


def plot_thresholding_metrics(ct, logistic=1, title="", thresh_range=(0, 50), step_size=None, legend=True):
    if step_size == None:
        step_size = (thresh_range[1] - thresh_range[0])/10
    rrange = arange(thresh_range[0], thresh_range[1], step=step_size, dtype=float)

    plt.plot(rrange, [thresholding_precision(10**(-i), ct) for i in rrange], label="precision", marker="o")
    plt.plot(rrange, [thresholding_recall(10**(-i), ct) for i in rrange], label="recall", marker="o")

    plt.grid()
    plt.xticks(arange(thresh_range[0], thresh_range[1], step=step_size))
    plt.xlabel("$-\log_{10}$threshold")
    plt.ylim((0, 1.1))
    plt.title(title)
    if legend:
        plt.legend()


def manhattan_plot(ct, logistic=1, legend=True):
    method = "logistic" if logistic else "chi2"
    groups = ct.groupby("causal")
    for is_causal, group in groups:
        plt.scatter(group.index, -log10(group[f"{method}_p"]), label="causal" if is_causal else "not causal")
    plt.xlabel("SNP id")
    plt.ylabel("$-\log_{10}$p")
    plt.title("p_manhattan")
    if legend:
        plt.legend(title="causality")

def get_SNPs(ct, p, logistic=True):
    method = "logistic_p" if logistic else "chi2_p"
    return ct.query(f"{method} < {p}").index.values

def get_n_SNPs(ct, n, logistic=True):
    method = "logistic_p" if logistic else "chi2_p"
    ct = ct.sort_values(by=method)
    return ct[:n].index.values


#AE stuff
def ae_ct(ct, ae):
    weights = ae.layers[1].get_weights()[0]
    sum_weights = sum(abs(weights), axis=1)
    ct["weights"] = sum_weights
    return ct

def ae_thresh_plot(ct):
    groups = ct.groupby("causal")
    for is_causal, group in groups:
        plt.scatter(group.index, group["weights"], label="causal" if is_causal else "not causal")
    
    plt.legend(title="causality")
    plt.title("ae_manhattan")
    plt.xlabel("SNP id")
    plt.ylabel("Sum of Weights")

def ae_thresholding_recall(thresh, ct):
    return recall_score(ct["causal"], ct["weights"] > thresh, zero_division=0)


def ae_thresholding_precision(thresh, ct):
    return precision_score(ct["causal"], ct["weights"] > thresh, zero_division=1)


def ae_plot_thresholding_metrics(ct, title="", thresh_range=None, step_size=None, legend=True):
    if thresh_range == None:
        thresh_range = (min(ct["weights"]), max(ct["weights"]))
    if step_size == None:
        step_size = (thresh_range[1] - thresh_range[0])/10
    

    rrange = arange(thresh_range[0], thresh_range[1], step=step_size, dtype=float)

    plt.plot(rrange, [ae_thresholding_precision(i, ct) for i in rrange], label="precision", marker="o")
    plt.plot(rrange, [ae_thresholding_recall(i, ct) for i in rrange], label="recall", marker="o")

    plt.grid()
    plt.xticks(arange(thresh_range[0], thresh_range[1], step=step_size))
    plt.xlabel("Weights")
    plt.ylim((0, 1.1))
    plt.title(title)
    if legend:
        plt.legend()

def ae_get_SNPs(ct, thresh):
    return ct.query(f"weights > {thresh}").index.values


def ae_get_n_SNPs(ct, n):
    ct = ct.sort_values(by="weights", ascending=False)
    return ct[:n].index.values
