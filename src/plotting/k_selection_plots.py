import os
import math
import mudata
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.regression.mixed_linear_model import MixedLM
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import cnmf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import scipy.sparse as sp
import anndata as ad


# collect all NMF runs 
# important: i must use the same package inference (torch-nmf or sk-nmf) to call combine here
def load_stablity_error_data(output_directory,run_name, components = [30, 50, 60, 80, 100, 200, 250, 300], 
density_threshold = 0.5, local_neighborhood_size = 0.3):

    cnmf_obj = cnmf.cNMF(output_dir=output_directory, name=run_name)

    stats = []
    norm_counts = sc.read(cnmf_obj.paths['normalized_counts'])
    for k in components:
        stats.append(cnmf_obj.consensus(k, skip_density_and_return_after_stats=True, show_clustering=False, 
        close_clustergram_fig=True, norm_counts=norm_counts, density_threshold = density_threshold,local_neighborhood_size = local_neighborhood_size).stats)

    stats = pd.DataFrame(stats)

    print("min stablity is", stats['silhouette'].min())
    print("max stablity is", stats['silhouette'].max())

    print("min error is",stats['prediction_error'].min())
    print("max error is",stats['prediction_error'].max())

    return stats

# plot NMF stability and error
def plot_stablity_error(stats, folder_name = None, file_name = None):
    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Top subplot - Stability (using silhouette as stability metric)
    ax1.plot(stats['k'], stats['silhouette'], 'k-', linewidth=2)
    ax1.set_ylabel('Stability', fontsize=12)
    ax1.set_xlim(0, 310)
    #ax1.set_ylim(0.1, 0.4)  # Adjust based on your data range
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # Bottom subplot - Error (using prediction_error)
    ax2.plot(stats['k'], stats['prediction_error'], 'k-', linewidth=2)
    ax2.set_xlabel('k', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_xlim(0, 310)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    if folder_name and file_name:
        fig.savefig(f"{folder_name}/{file_name}.png", dpi=300, bbox_inches="tight" )#  transparent=True)


# Load data for differeent enrichment test
def load_enrichment_data(folder, components = [30, 50, 60, 80, 100, 200, 250, 300]):

    # loading function 
    def load(k, term, folder):
        # read evaluation results
        df = pd.read_csv(folder, sep='\t')
        df = df.loc[df['Adjusted P-value']<=0.05]
        df['num_programs'] = k
        df['test_term'] = term

        return df

    # collect all the results for each k
    term_df = []

    for k in components:

        term_df.append(load(k, 'go_terms', '{}/{}/{}_GO_term_enrichment.txt'.format(folder,k,k)))
        term_df.append(load(k, 'genesets', '{}/{}/{}_geneset_enrichment.txt'.format(folder,k,k)))
        term_df.append(load(k, 'traits', '{}/{}/{}_trait_enrichment.txt'.format(folder,k,k)))
        #term_df.append(load(k, 'motifs', '{}/{}/{}_motif_enrichment.txt'.format(folder,k,k)))

    term_df = pd.concat(term_df, ignore_index=True)

    # Count unique terms per k
    count_df = pd.DataFrame(index=components, columns=term_df['test_term'].unique()) 

    for k in components:
        for col in count_df.columns:
            count_df.loc[k, col] = term_df.loc[(term_df['num_programs']==k) & (term_df['test_term']==col), 'Term'].unique().shape[0]


    #print out some stats
    print("min go_terms is", count_df['go_terms'].min())
    print("max go_terms is", count_df['go_terms'].max())

    print("min genesets is", count_df['genesets'].min())
    print("max genesets is", count_df['genesets'].max())

    print("min traits is", count_df['traits'].min())
    print("max traits is", count_df['traits'].max())

    #print("min motif is", count_df['motifs'].min())
    #print("max motif is", count_df['motifs'].max())


    return count_df


# plot loaded df
def plot_enrichment(count_df, folder_name = None, file_name = None):

    # Create the plot with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

    # Top subplot - Stability (using silhouette as stability metric)
    ax1.plot(count_df.index, count_df['go_terms'], 'k-', linewidth=2)
    ax1.set_ylabel('GO Terms', fontsize=12)
    ax1.set_xlim(0, 310)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # Bottom subplot - Error (using prediction_error)
    ax2.plot(count_df.index, count_df['genesets'], 'k-', linewidth=2)
    ax2.set_xlabel('k', fontsize=12)
    ax2.set_ylabel('Genesets', fontsize=12)
    ax2.set_xlim(0, 310)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    # Bottom subplot - Error (using prediction_error)
    ax3.plot(count_df.index, count_df['traits'], 'k-', linewidth=2)
    ax3.set_xlabel('k', fontsize=12)
    ax3.set_ylabel('Traits', fontsize=12)
    ax3.set_xlim(0, 310)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=10)

    if folder_name and file_name:
        fig.savefig(f"{folder_name}/{file_name}.png", dpi=300, bbox_inches="tight")# transparent=True)


# load perturbation data 
def load_perturbation_data(folder, pval = 0.000335, components = [30, 50, 60, 80, 100, 200, 250, 300]):
    
    # Compute no. of unique regulators
    test_stats_df = []

    for k in components:  
        # Run perturbation assocation
        for samp in ['D0', 'sample_D1', 'sample_D2', 'sample_D3']:
            test_stats_df_ = pd.read_csv('{}/{}/{}_perturbation_association_results_{}.txt'.format(folder,k,k,samp), sep='\t')
            test_stats_df_['sample'] = samp
            test_stats_df_['K'] = k
            test_stats_df_['fdr'] = fdrcorrection(test_stats_df_['pval'])[1]
            test_stats_df.append(test_stats_df_)
            
    test_stats_df = pd.concat(test_stats_df, ignore_index=True)

    # pring some stats
    plotting_df = test_stats_df.loc[test_stats_df.pval< pval, ['K','target_name']].drop_duplicates().groupby(['K']).count().reset_index()

    print("min regulators is", plotting_df["target_name"].min())
    print("max regulators is", plotting_df["target_name"].max())

    return test_stats_df


# plot perturbation data
def plot_perturbation(test_stats_df, pval =0.000335, folder_name = None, file_name = None):

    fig, axs = plt.subplots(ncols=1, nrows=2,figsize=(5, 5))

    axs[0].set_title(f'Unique regulators of progams per sample (pval <= {str(pval)})', fontsize=10)
    plotting_df = test_stats_df.loc[test_stats_df.pval<=pval, ['K', 'sample','target_name']].drop_duplicates().groupby(['K', 'sample']).count().reset_index()
    sns.lineplot(x='K', y='target_name', hue='sample', data=plotting_df, ax=axs[0])
    axs[0].set_ylabel('# Regulators', fontsize=10)
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')


    axs[1].set_title(f'Unique regulators of progams (pval <= {str(pval)})', fontsize=10)
    plotting_df = test_stats_df.loc[test_stats_df.pval<=pval, ['K','target_name']].drop_duplicates().groupby(['K']).count().reset_index()
    sns.lineplot(x='K', y='target_name', data=plotting_df, color='black', ax=axs[1])
    axs[1].set_ylabel('# Regulators', fontsize=10)

    if folder_name and file_name:
        fig.savefig(f"{folder_name}/{file_name}.png", dpi=300, bbox_inches="tight")#  transparent=True)


# load total explained variance
def load_explained_variance_data(folder, components = [30, 50, 60, 80, 100, 200, 250, 300]):

    stats = {}
    for k in components:
    
        input_path = f"{folder}/{k}/{k}_Explained_Variance_Summary.txt"
        df = pd.read_csv(input_path, sep = '\t', index_col = 0)

        stats[k] = df['Total'].values[0]

    
    print("min Explained_variance is", min(stats.values()))
    print("max Explained_variance is", max(stats.values()))

    return stats


# plot NMF explained variance
def plot_explained_variance(stats, folder_name=None, file_name=None):
    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 2))

    # Plot the data
    ax.plot(list(stats.keys()), list(stats.values()), 'k-', linewidth=2)
    ax.set_xlabel('Components')
    ax.set_ylabel('TotalExplained Variance')
    ax.set_xlim(0, 310)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Save the figure if folder and file names are provided
    if folder_name and file_name:
        fig.savefig(f"{folder_name}/{file_name}.png", dpi=300, bbox_inches="tight") # transparent=True)
    
    plt.show()



''' Faster stability + error calculation & error calculation is a bit off 
because the refit_usages is not the same as the saved one, it needs to refit but it takes time 
def load_stablity_error_data_(k, cnmf_obj, norm_counts, rf_usages_path, density_threshold, local_neighborhood_size):

    # load files 
    merged_spectra = cnmf.load_df_from_npz(cnmf_obj.paths['merged_spectra']%k)
    rf_usages = pd.read_csv(rf_usages_path, sep="\t")
    
    # calculate stability 
    n_neighbors = int(local_neighborhood_size * merged_spectra.shape[0]/k)
    l2_spectra = (merged_spectra.T/np.sqrt((merged_spectra**2).sum(axis=1))).T

    kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
    kmeans_model.fit(l2_spectra)
    kmeans_cluster_labels = pd.Series(kmeans_model.labels_+1, index=l2_spectra.index)
    silhouette = silhouette_score(l2_spectra.values, kmeans_cluster_labels, metric='euclidean')

    # Find median usage for each gene across cluster
    median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median()

    # Normalize median spectra to probability distributions.
    median_spectra = (median_spectra.T/median_spectra.sum(1)).T

    # reset index and col names 
    median_spectra.columns = range(len(median_spectra.columns))
    median_spectra = median_spectra.reset_index(drop=True)

    

    rf_usages = rf_usages.drop("bc_wells", axis = 1)
    rf_usages.columns = range(len(rf_usages.columns))
    rf_usages = rf_usages.reset_index(drop=True)


    # Compute prediction error as a frobenius norm
    rf_pred_norm_counts = rf_usages.dot(median_spectra)        
    if sp.issparse(norm_counts.X):
        prediction_error = ((norm_counts.X.todense() - rf_pred_norm_counts)**2).sum().sum()
    else:
        prediction_error = ((norm_counts.X - rf_pred_norm_counts)**2).sum().sum()    
        
    return pd.DataFrame([k, density_threshold, silhouette,  prediction_error],
            index = ['k', 'local_density_threshold', 'silhouette', 'prediction_error'],
            columns = ['stats'])


def load_stablity_error_data(output_directory, run_name, local_neighborhood_size=0.30, 
   density_threshold=2.0, components = [30, 50, 60, 80, 100, 200, 250, 300]):

    cnmf_obj = cnmf.cNMF(output_dir=output_directory, name=run_name)

    stats = []
    norm_counts = sc.read(cnmf_obj.paths['normalized_counts'])
    
    for k in components:
        rf_usages_path = '{output_directory}/{run_name}/{run_name}.usages.k_{k}.dt_{sel_thresh}.consensus.txt'.format(
                                                                                output_directory=output_directory,
                                                                                run_name = run_name,
                                                                                k=k,
                                                                                sel_thresh = f"{density_threshold}".replace('.','_'))

        stats.append(load_stablity_error_data_(k, cnmf_obj, norm_counts, rf_usages_path, density_threshold, local_neighborhood_size))

    arr = np.array(stats).squeeze()
    df = pd.DataFrame(arr, columns=['k', 'local_density_threshold', 'silhouette', 'prediction_error'])

    print("min stablity is", df['silhouette'].min())
    print("max stablity is", df['silhouette'].max())

    print("min error is",df['prediction_error'].min())
    print("max error is",df['prediction_error'].max())

    return df
'''