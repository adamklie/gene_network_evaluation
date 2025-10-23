import muon as mu 
import scanpy as sc
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import scanpy as sc
import anndata as ad
from pathlib import Path
from PIL import Image
import mygene
import os
from PyPDF2 import PdfMerger
import glob
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from adjustText import adjust_text
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import multiprocessing as mp
from multiprocessing import Pool, Manager
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
#plt.rcParams["axes.spines.bottom"] = False
#plt.rcParams["axes.spines.left"] = False
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts (editable text)
plt.rcParams['ps.fonttype'] = 42   # For EPS as well



plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'legend.fontsize': 16,        # Legend text size
    'legend.title_fontsize': 16,  # Legend title size
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.titlesize': 20
})

import sys
# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/cNMF_benchmarking/cNMF_benchmarking_pipeline')

from .utilities import convert_adata_with_mygene, convert_with_mygene, rename_list_gene_dictionary, rename_adata_gene_dictionary



# plot gene expression UMAP given adata
def plot_umap_per_gene(mdata, Target_Gene, file_to_dictionary = None, ax=None,
 color='purple', save_path=None, save_name=None, figsize=(8,6), show=False):

    if file_to_dictionary is None:
        renamed = mdata['rna'].copy()
    else:
        renamed = rename_adata_gene_dictionary(mdata['rna'], file_to_dictionary)


    # Check if query gene exists
    if Target_Gene not in renamed.var_names.values:
        raise ValueError(f"Gene {Target_Gene} not found in mdata")
    
    # Check if gene exists
    gene_name_list = renamed.var_names.tolist()
    if Target_Gene not in gene_name_list:
        print("gene name is not found in adata")
        return None
    
    # Set color
    colors = ['lightgrey', color]
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    title = f'{Target_Gene} Expression'
    
    # If no axis provided, create new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False
    
    # Plot on the provided/created axis
    plt.sca(ax)  # Set current axis to ensure scanpy uses correct figure -CC change
    sc.pl.umap(renamed, color=Target_Gene, title=title, cmap=cmap, ax=ax, show=False)
    
    # Rasterize ONLY the scatter points (collections) in this axis
    for collection in ax.collections:
        collection.set_rasterized(True)
    
    ax.set_title(title, fontsize=18, fontweight='bold', loc='center')
    ax.set_xlabel('UMAP 1', fontsize=10, fontweight = 'bold')
    ax.set_ylabel('UMAP 2', fontsize=10, fontweight = 'bold')

    # Only save if standalone and save_path provided
    if standalone and save_path and save_name:
        fig.savefig(f"{save_path}/{save_name}.svg", format='svg', bbox_inches='tight', dpi=300)
    
    # Only show/close if standalone
    if standalone:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax



# plot guide UMAP given mdata with guide assignment
def plot_umap_per_gene_guide(mdata, Target_Gene, ax=None, color='red', save_path=None, 
save_name=None, figsize=(8,6), show=False):

    # Extract gRNA per cell and merge the gRNA targeting the same gene 
    X = mdata['cNMF'].obsm["guide_assignment"].T
    df = pd.DataFrame(X.toarray if hasattr(X, "toarray") else X,
                    index =  mdata['cNMF'].uns["guide_targets"]
                    )
    df_merge = df.groupby(df.index).sum()

    adata = ad.AnnData(X=df_merge.T.values, obs=mdata['rna'].obs ) 
    adata.var_names = df_merge.index
    adata.obsm['X_pca'] = mdata['rna'].obsm['X_pca']
    adata.obsm['X_umap'] = mdata['rna'].obsm['X_umap']
    
    # Check if gene exists
    gene_name_list = adata.var_names.tolist()
    if Target_Gene not in gene_name_list:
        raise ValueError(f"Gene {Target_Gene} not found in guide assignment")
    
        return None
    
    # Set color
    colors = ['lightgrey', color]
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    title = f'{Target_Gene} Perturbation'
    
    # If no axis provided, create new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False
    
    # Plot on the provided/created axis
    plt.sca(ax)  # Set current axis to ensure scanpy uses correct figure -CC change

    sc.pl.umap(adata, color=Target_Gene, title=title, cmap=cmap, ax=ax, show=False)
    
    # Rasterize ONLY the scatter points (collections) in this axis
    for collection in ax.collections:
        collection.set_rasterized(True)
    
    ax.set_title(title, fontsize=18, fontweight='bold', loc='center')
    ax.set_xlabel('UMAP 1', fontsize=10, fontweight = 'bold')
    ax.set_ylabel('UMAP 2', fontsize=10, fontweight = 'bold')

    # Only save if standalone and save_path provided
    if standalone and save_path and save_name:
        fig.savefig(f"{save_path}/{save_name}.svg", format='svg', bbox_inches='tight', dpi=300)
    
    # Only show/close if standalone
    if standalone:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax



# read the cNMF gene matrix (in txt), plot top x gene in the program 
def plot_top_program_per_gene(mdata,  Target_Gene, file_to_dictionary = None,top_program=10, 
                               species="human", ax=None, save_path=None, save_name=None, 
                               figsize=(5,8), show=False):

    # read cNMF gene program matrix
    X =  mdata["cNMF"].varm["loadings"] 

    # rename gene
    if file_to_dictionary is None:
        df_renamed = pd.DataFrame(data=X, columns = mdata["rna"].var_names)
    else:
        renamed_gene_list = rename_list_gene_dictionary( mdata["rna"].var_names, file_to_dictionary)
        df_renamed = pd.DataFrame(data=X, columns = renamed_gene_list)

    
    # Check if gene exists
    if Target_Gene not in df_renamed.columns:
        raise ValueError(f"Gene {Target_Gene} not found in mdata")
    
    
    # sort top x program
    df_sorted = df_renamed[Target_Gene].nlargest(top_program)
    df_sorted = df_sorted.sort_values(ascending=True)

    
    # If no axis provided, create new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(df_sorted)), df_sorted.values, color='#808080', alpha=0.8)
    
    # Customize the plot
    ax.set_title(f"Top Loading Program for {Target_Gene}",fontsize=12, fontweight='bold', loc='center')
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted.index, fontsize=10)
    ax.set_xlabel('Gene Loading Score (z-score)',fontsize=10, fontweight='bold') #, loc='center')
    ax.set_ylabel(f'Program Name',fontsize=10, fontweight='bold', loc='center')
    
    # Format x-axis
    ax.set_xlim(0, max(df_sorted.values) * 1.1)
    ax.ticklabel_format(style='scientific', axis='x' )#scilimits=(0,0))
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    # Only save if standalone and save_path provided
    if standalone and save_path and save_name:
        fig.savefig(f"{save_path}/{save_name}.svg", format='svg', bbox_inches='tight', dpi=300)
    
    # Only show/close if standalone
    if standalone:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax



# plot dotplot given a target gene 
def perturbed_gene_dotplot(mdata,  Target_Gene, file_to_dictionary=None, groupby='sample', save_name=None,
                          save_path=None, figsize=(3, 2), show=False, ax=None):
    # read in adata
    if file_to_dictionary is None:
        renamed = mdata['rna']
    else:
        renamed = rename_adata_gene_dictionary(mdata['rna'],file_to_dictionary)

    renamed.var_names_make_unique()


    # Check if query gene exists
    if Target_Gene not in renamed.var_names.values:
        raise ValueError(f"Gene {Target_Gene} not found in mdata")
    
    
    if save_name is None:
        save_name = f"{Target_Gene} Expression by days"
    
    # Create the dotplot
    if ax is None:
        # Standalone mode - let scanpy create its own figure
        dp = sc.pl.dotplot(renamed, Target_Gene, groupby=groupby,
                          figsize=figsize, swap_axes=True, dendrogram=False,
                          show=False, return_fig=True)
        dp.make_figure()
        fig = dp.fig
        ax = dp.ax_dict['mainplot_ax']
    else:
        # Gridspec mode - use provided ax
        fig = ax.get_figure()
        dp = sc.pl.dotplot(renamed, Target_Gene, groupby=groupby,
                          swap_axes=True, dendrogram=False, show=False,
                          return_fig=True, ax=ax)
        dp.make_figure()
    
    ax.set_title(f"{Target_Gene} Expression", fontsize=14, fontweight='bold', loc='center')
    ax.set_ylabel('Gene', fontsize=10, fontweight='bold', loc='center')
    ax.set_xlabel('Day', fontsize=10, fontweight='bold', loc='center')
    
    # Get labels and set ticks properly
    label = list(mdata['rna'].obs[groupby].cat.categories)  # Use categories instead
    
    # Set both ticks and labels together
    tick_positions = range(len(label))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(label, fontsize=8)

    
    # save fig (only in standalone mode)
    if save_name and save_path and ax is None:
        fig.savefig(f"{save_path}/{save_name}.png", format='png', bbox_inches='tight', dpi=300)  # Changed to png
    
    # Control whether to display the plot (only in standalone mode)
    if ax is None:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax



# plot barplot for up/down regulated genes log2FC given results from perturbation analysis, return genes in df
def plot_log2FC(perturb_path, Target, tagert_col_name="target_name", plot_col_name="program_name", 
                log2fc_col='log2FC', num_item=5, p_value=0.05, save_path=None, save_name=None, 
                figsize=(5, 4), show=False, ax=None, Day =""):

    # read df
    df = pd.read_csv(perturb_path, sep="\t")
    
    # Check if query gene exists
    if Target not in df[tagert_col_name].values:
        print(f"Gene {Target} not found in mdata")
        return 

    # Sort by log2FC
    df_sorted = df.loc[df[tagert_col_name] == Target]
    df_sorted = df_sorted[df_sorted['adj_pval'] < p_value]
    df_sorted = df_sorted.sort_values(by=log2fc_col, ascending=False)
    # Get top and bottom gene
    top_df = df_sorted.head(num_item)
    bottom_df = df_sorted.tail(num_item)
    # Combine and add category
    top = top_df.copy()
    bottom = bottom_df.copy()
    # Combine data
    plot_data = pd.merge(top, bottom, how='outer')
    plot_data = plot_data.sort_values(by=log2fc_col, ascending=False)
    
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Create horizontal bar plot
    colors = ['red' if x > 0 else 'blue' for x in plot_data[log2fc_col]]
    bars = ax.barh(range(len(plot_data)), plot_data[log2fc_col], color=colors, alpha=0.7)
    
    # Customize the plot
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data[plot_col_name],fontsize=10) 


    ax.set_xlabel('Effect on Program Expression (log2 fold-change)',fontsize=10, fontweight='bold', loc='center')
    ax.set_ylabel( "Program Name",fontsize=10, fontweight='bold', loc='center')
    ax.set_title(f"Program Log2 Fold-Change with {Target} {Day}",fontsize=14, fontweight='bold', loc='center')

    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)


    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Upregulated'),
                      Patch(facecolor='blue', alpha=0.7, label='Downregulated')]
    ax.legend(handles=legend_elements) #loc='lower right')
    
    # Adjust layout
    ax.grid(axis='x', alpha=0.3)
    
    # Save if path provided (only when ax is not provided)
    if save_path and ax is None:
        fig.savefig(f'{save_path}/{save_name}.svg', format='svg', bbox_inches='tight', dpi=300)
    
    # Control whether to display the plot (only when ax is not provided)
    if ax is None:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax, plot_data
 



# plot one volcone plot given perturbation analysis, return genes in df
def plot_volcano(perturb_path, Target, tagert_col_name="target_name", plot_col_name="program_name", 
                 down_thred_log=-0.05, up_thred_log=0.05, p_value=0.05, save_path=None, 
                 save_name=None, figsize=(5, 4), show=False, ax=None, Day =""):
                 
    df = pd.read_csv(perturb_path, sep="\t")
       
    # Check if query gene exists
    if Target not in df[tagert_col_name].values:
        print(f"Gene {Target} not found in mdata")
        return 

    df_program = df.loc[df[tagert_col_name] == Target]
    
    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot all points
    ax.scatter(x=df_program['log2FC'], 
               y=df_program['adj_pval'].apply(lambda x: -np.log10(x)), 
               s=10, label="Not significant", color="grey")
    
    # Highlight down- or up-regulated genes
    down = df_program[(df_program['log2FC'] <= down_thred_log) & 
                      (df_program['adj_pval'] <= p_value)]
    up = df_program[(df_program['log2FC'] >= up_thred_log) & 
                    (df_program['adj_pval'] <= p_value)]
    
    ax.scatter(x=down['log2FC'], 
               y=down['adj_pval'].apply(lambda x: -np.log10(x)), 
               s=10, label="Down-regulated", color="blue")
    ax.scatter(x=up['log2FC'], 
               y=up['adj_pval'].apply(lambda x: -np.log10(x)), 
               s=10, label="Up-regulated", color="red")
    
    # Add text labels
    texts = []
    for i, r in up.iterrows():
        texts.append(ax.text(x=r['log2FC'], y=-np.log10(r['adj_pval']), 
                            s=r[plot_col_name], fontsize=10, zorder=10))
    for i, r in down.iterrows():
        texts.append(ax.text(x=r['log2FC'], y=-np.log10(r['adj_pval']), 
                            s=r[plot_col_name], fontsize=10, zorder=10))
    
    # Adjust text to avoid overlaps
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5), ax=ax)
    
    
    # Set labels and lines
    ax.set_title(f"Volcano Plot for {Target} {Day}",fontsize=14, fontweight='bold', loc='center')
    ax.set_xlabel('Effect on Program Expression (log2 fold-change)',fontsize=10, fontweight='bold', loc='center')
    ax.set_ylabel("Adjusted p-value, -log10",fontsize=10, fontweight='bold', loc='center')
    ax.axvline(down_thred_log, color="grey", linestyle="--")
    ax.axvline(up_thred_log, color="grey", linestyle="--")
    ax.axhline(-np.log10(p_value), color="grey", linestyle="--")
    ax.legend()
    ax.set_title(f"Volcano Plot for {Target} {Day}",fontsize=14, fontweight='bold', loc='center')

    
    # Save if path provided (only when ax is not provided)
    if save_path and ax is None:
        fig.savefig(f'{save_path}/{save_name}.svg', format='svg', bbox_inches='tight', dpi=300)
    
    # Control whether to display the plot (only when ax is not provided)
    if ax is None:
        if show:
            plt.show()
        else:
            plt.close(fig)
        
    return ax, pd.merge(up, down, how='outer'), texts   




# given mdata, list of programs to plot, plot dotplot for programs, split by days
def programs_dotplot(mdata, Target, groupby="sample", program_list=None,
                     save_path=None, save_name=None, figsize=(5, 4), show=False, ax=None, Day=""):
 
    # make anndata from program loadings
    adata_new = mdata['cNMF']

    gene_list = adata_new.var_names.tolist()
    
    if program_list is not None:
        gene_list = list(map(str, program_list))
    
    if not gene_list:

        print("No significant Gene")
        # Handle empty gene list
        if ax is None:
            blank_img = np.ones((300, 200, 3), dtype=np.uint8) * 255
            img = Image.fromarray(blank_img)
            if save_path and save_name:
                full_path = f"{save_path}/{save_name}.png"
                img.save(full_path)
            return None
        else:
            ax.text(0.5, 0.5, 'No programs to display',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{Target} Perturbed Program Loading Scores {Day}", 
                        fontweight='bold', loc='center')
            return ax
    
    gene_list = gene_list[::-1]
    
    # Create dotplot
    if ax is None:
        # Standalone mode - let scanpy create its own figure
        dp = sc.pl.dotplot(adata_new, gene_list, groupby=groupby, swap_axes=True, figsize=figsize,
                          dendrogram=False, show=False, return_fig=True)
        dp.make_figure()
        fig = dp.fig
        ax = dp.ax_dict['mainplot_ax']
    else:
        # Gridspec mode - use provided ax
        fig = ax.get_figure()
        dp = sc.pl.dotplot(adata_new, gene_list, groupby=groupby, swap_axes=True, figsize=figsize,
                          dendrogram=False, show=False, return_fig=True, ax=ax)
        dp.make_figure()
    
    ax.set_title(f"{Target} Perturbed Program Loading Scores {Day}", 
                fontsize=14, fontweight='bold', loc='center')
    ax.set_ylabel('Program Name', fontsize=10, fontweight='bold', loc='center')
    ax.set_xlabel("Day", fontsize=10, fontweight='bold', loc='center')
    
    # Get labels and set ticks properly
    label = list(adata_new.obs[groupby].cat.categories)
    
    # Set both ticks and labels together
    tick_positions = range(len(label))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(label, fontsize=8)
    
    # Save if path provided (only when ax is not provided)
    if save_path and save_name and ax is None:
        fig.savefig(f'{save_path}/{save_name}.svg', format='svg', bbox_inches='tight', dpi=300)
    
    # Control whether to display the plot (only in standalone mode)
    if ax is None:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax
 
 
 

# plot top x programs for perturbed gene given Target gene and gene loading file path
def analyze_correlations(mdata, Target, file_to_dictionary=None, top_num=5, save_path=None, 
                         save_name=None, figsize=(10, 8), show=False, ax=None):
        
    X =  mdata["cNMF"].varm["loadings"] 

    # rename gene
    if file_to_dictionary is None:
        df_rename = pd.DataFrame(data=X, columns = mdata["rna"].var_names)
    else:
        renamed_gene_list = rename_list_gene_dictionary( mdata["rna"].var_names, file_to_dictionary)
        df_rename = pd.DataFrame(data=X, columns = renamed_gene_list)

    # check if gene exists
    if Target not in df_rename.columns:
        raise ValueError(f"Gene {Target} not found in mdata")


    # Calculate correlation matrix
    correlation_matrix = df_rename.corr()
    
    # Get correlations with the target program
    target_correlations = correlation_matrix[Target]
    target_correlations = target_correlations.drop(Target)  # Remove self-correlation
    
    # Sort correlations
    sorted_correlations = target_correlations.sort_values(ascending=False)
    
    # Get top and bottom gene
    top = sorted_correlations.head(top_num)
    bottom = sorted_correlations.tail(top_num)
    
    # Combine for plotting
    combined_correlations = pd.concat([bottom, top])
    combined_correlations = combined_correlations.sort_values(ascending=False)
    
    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Create horizontal bar plot
    colors = ['blue' if x < 0 else 'red' for x in combined_correlations]
    bars = ax.barh(range(len(combined_correlations)), combined_correlations.values, 
                   color=colors, alpha=0.7)
    
    # Customize the plot
    ax.set_title(f"Gene Loading Correlation for {Target}",fontsize=14, fontweight='bold', loc='center')
    ax.set_ylabel('Gene Name',fontsize=10, fontweight='bold', loc='center')
    ax.set_xlabel('Correlation Coefficient',fontsize=10, fontweight='bold', loc='center')
    
    # Set y-axis labels
    ax.set_yticks(range(len(combined_correlations)))
    ax.set_yticklabels(combined_correlations.index, rotation=45, ha='right')
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Positive correlation'),
                      Patch(facecolor='blue', alpha=0.7, label='Negative correlation')]
    ax.legend(handles=legend_elements )# loc='lower right')
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    
    # Adjust layout (only in standalone mode)
    if ax is None:
        plt.tight_layout()
    
    # Save if path provided (only in standalone mode)
    if save_path and ax is None:
        fig.savefig(f'{save_path}/{save_name}.svg', format='svg', bbox_inches='tight', dpi=300)
    
    # Control whether to display the plot (only in standalone mode)
    if ax is None:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax, combined_correlations




# plot the waterfall plot for genes that have simliar program loading scores when perturbed 
def create_gene_correlation_waterfall(perturb_path, Target_Gene, top_num=5, save_path=None, 
                         save_name=None, figsize=(3, 5), show=False, ax=None, Day =""):

    # Read data
    df = pd.read_csv(perturb_path, sep='\t', index_col=0)
    
    # Check if query gene exists
    if Target_Gene not in df.index:
        raise ValueError(f"Gene {Target_Gene} not found in perturb analysis")
    
    # Get log2FC values for query gene across all programs
    df_gene = df.loc[Target_Gene]
    df_gene_log2fc = df_gene[['log2FC','program_name']].sort_values(by = "program_name")
    df_gene_log2fc.drop("program_name", axis = 1, inplace = True)
    
    correlations = {}
    for gene in df.index:
        if gene != Target_Gene:
            gene_values = df.loc[gene]
            gene_log2fc = gene_values[['log2FC','program_name']].sort_values(by = "program_name")
            gene_log2fc.drop("program_name", axis = 1, inplace = True)
            corr, _ = pearsonr(df_gene_log2fc['log2FC'], gene_log2fc['log2FC'])
            correlations[gene] = corr
    
    # Convert to DataFrame and sort
    corr_df = pd.DataFrame(list(correlations.items()),
                           columns=['gene', 'correlation'])
    corr_df = corr_df.sort_values('correlation', ascending=False)
    
    # Get top N positive and bottom N negative correlations for labeling
    top_positive = corr_df.head(top_num)
    top_negative = corr_df.tail(top_num)
    genes_to_label = set(top_positive['gene'].tolist() + top_negative['gene'].tolist())
    
    # Use ALL correlations for plotting
    plot_data = corr_df.reset_index(drop=True)

    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    
    # Plot only the markers (no line)
    ax.scatter(range(len(plot_data)), plot_data['correlation'], 
               color='gray', s=16)
    
    # Add horizontal reference line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Remove x-axis labels
    ax.set_xticks([])

    
    # Highlight the labeled genes with larger markers and add annotations
    texts = []
    for idx, row in plot_data.iterrows():
        if row['gene'] in genes_to_label:
            ax.plot(idx, row['correlation'], 'o', color='darkgray', markersize=8, zorder=5)
            # Add gene name annotation
            text = ax.text(idx, row['correlation'], row['gene'],
                          fontsize=10,
                          style='italic',
                          ha='center')
            texts.append(text)
    
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=ax, fontsize=12) 
    
    ax.set_title(f"Gene Perturbation Correlation for {Target_Gene} {Day}",fontsize=14, fontweight='bold', loc='center')
    ax.set_ylabel(f'Correlation of Program\nExpression with\n{Target_Gene} Perturbation',fontsize=10, fontweight='bold', loc='center')
    ax.set_xlabel('Perturbed Genes',fontsize=10, fontweight='bold', loc='center')

    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Adjust layout (only in standalone mode)
    if ax is None:
        plt.tight_layout()
    
    # Save if path provided (only in standalone mode)
    if save_path and ax is None:
        fig.savefig(f'{save_path}/{save_name}.svg', format='svg', bbox_inches='tight', dpi=300)
    

    # Control whether to display the plot (only in standalone mode)
    if ax is None:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    
    return ax, texts




# graph one big svg or pdf 
def create_comprehensive_plot(
    mdata,
    perturb_path,
    file_to_dictionary,
    Target_Gene,
    top_program=10,
    species="human", 
    groupby='sample',
    tagert_col_name="target_name",
    plot_col_name="program_name",
    log2fc_col='log2FC',
    top_num=5,
    down_thred_log=-0.05,
    up_thred_log=0.05,
    p_value=0.05,
    save_path=None,
    save_name=None,
    figsize=None,
    samples=None,
    square_plots=True,
    show=True,
    PDF=True
):
    """
    Create a comprehensive multi-panel figure for analyzing gene perturbations.
    
    Parameters:
    -----------
    samples : list, optional
        List of sample/day identifiers. If None, defaults to ['D0', 'sample_D1', 'sample_D2', 'sample_D3'].
        Can be any length to accommodate different numbers of timepoints.
    
    figsize : tuple, optional
        Figure size (width, height). If None, automatically calculated based on:
        - square_plots=True: calculates height to make all plots square
        - square_plots=False: uses legacy calculation (25, 4 + num_samples * 4)
    
    square_plots : bool, default=True
        If True, automatically adjusts figsize to make all plots square-shaped.
        If False, uses the legacy auto-sizing calculation.
        Ignored if figsize is explicitly provided.
    
    ... (other parameters remain the same as original)
    """
    
    # Set default samples if not provided
    if samples is None:
        samples = ['D0', 'sample_D1', 'sample_D2', 'sample_D3']
    
    # Calculate figure dimensions
    num_samples = len(samples)
    
    # Set matplotlib backend to non-interactive to reduce memory usage
    plt.ioff()
    
    # Close any existing figures to prevent memory leaks
    plt.close('all')
    
    # Create figure with custom gridspec
    # 1 header row + n sample rows
    num_rows = 1 + num_samples
    fig = plt.figure(figsize=figsize)
    
    # Create gridspec with dynamic number of rows using 27 columns for flexibility
    gs = gridspec.GridSpec(num_rows, 27, figure=fig, 
                        hspace=0.4,  # Vertical spacing between rows
                        wspace=0.5,  # Horizontal spacing between columns
                        left=0.10,   # Left margin
                        right=0.90)  # Right margin
    
    # First row: 5 header plots
    ax1 = fig.add_subplot(gs[0, 0:5])    # UMAP Expression
    ax21 = fig.add_subplot(gs[0, 6:11])   # Top program
    ax3 = fig.add_subplot(gs[0, 12:16])  # Perturbed gene dotplot
    ax4 = fig.add_subplot(gs[0, 17:22])  # Correlation analysis
    ax2 = fig.add_subplot(gs[0, 23:27])  # UMAP Perturbation
    
    # Initialize list with header axes
    axes = [ax1, ax2, ax3, ax4, ax21]
    
    # Dynamically create axes for each sample
    # Each sample row has 4 plots: Log2FC, Volcano, Dotplot, Waterfall
    for sample_idx in range(num_samples):
        row_idx = sample_idx + 1  # Start from row 1 (after header)
        
        ax_log2fc = fig.add_subplot(gs[row_idx, 1:6])    # Log2FC
        ax_volcano = fig.add_subplot(gs[row_idx, 7:12])  # Volcano
        ax_dotplot = fig.add_subplot(gs[row_idx, 13:18]) # Dotplot
        ax_waterfall = fig.add_subplot(gs[row_idx, 20:25]) # Waterfall
        
        axes.extend([ax_log2fc, ax_volcano, ax_dotplot, ax_waterfall])

    # Plot 1: UMAP
    ax1 = plot_umap_per_gene(
        mdata=mdata,
        file_to_dictionary=file_to_dictionary,
        Target_Gene=Target_Gene, 
        figsize=(4, 3),
        ax=ax1
    )
    ax1.set_title(f"{Target_Gene} Expression", fontsize=18, fontweight='bold', loc='center')
    ax1.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax1.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    
    # Plot 21: UMAP for gRNA
    ax21 = plot_umap_per_gene_guide(
        mdata=mdata,
        Target_Gene=Target_Gene, 
        figsize=(4, 3),
        ax=ax21
    )
    ax21.set_title(f"{Target_Gene} Perturbation", fontsize=18, fontweight='bold', loc='center')
    ax21.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax21.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')

    # Plot 2: Top program 
    ax2 = plot_top_program_per_gene(
        mdata=mdata,
        file_to_dictionary=file_to_dictionary,
        Target_Gene=Target_Gene,
        top_program=top_program,
        species=species, 
        ax=ax2,
        figsize=(2, 4)
    )
    ax2.set_title(f"Top Loading Program for {Target_Gene}", fontsize=20, fontweight='bold', loc='center')
    ax2.set_xlabel('Gene Loading Score (z-score)', fontsize=14, fontweight='bold')
    ax2.set_ylabel(f'Program Name', fontsize=14, fontweight='bold', loc='center')
    
    # Plot 3: Perturbed gene dotplot
    ax3 = perturbed_gene_dotplot(
        mdata=mdata,
        file_to_dictionary=file_to_dictionary,
        Target_Gene=Target_Gene,
        groupby=groupby,
        figsize=(3, 2),
        ax=ax3
    )
    ax3.set_title(f"{Target_Gene} Expression", fontsize=20, fontweight='bold', loc='center')
    ax3.set_ylabel('Gene', fontsize=14, fontweight='bold', loc='center')
    ax3.set_xlabel('Day', fontsize=14, fontweight='bold', loc='center')
    
    # Plot 4: Correlation analysis
    ax4, corr_data = analyze_correlations(
        mdata=mdata,
        file_to_dictionary=file_to_dictionary,
        Target=Target_Gene,
        top_num=top_num,
        figsize=(4, 3),
        ax=ax4
    )
    ax4.set_title(f"Gene Loading Correlation for {Target_Gene}", fontsize=20, fontweight='bold', loc='center')
    ax4.set_ylabel('Gene Name', fontsize=14, fontweight='bold', loc='center')
    ax4.set_xlabel('Correlation Coefficient', fontsize=14, fontweight='bold', loc='center')

    # Loop through samples and create 4 plots per sample (Log2FC, Volcano, Dotplot, Waterfall)
    ax_index = 5  # Start after header axes (0-4)
    
    for idx, samp in enumerate(samples):
        file_name = f"{perturb_path}_{samp}_perturbation_association.txt" 
        Day = f'Day {idx}'

        # Plot 1: Log2FC plot
        current_ax = axes[ax_index]
        current_ax, df = plot_log2FC( 
            perturb_path=file_name, 
            Target=Target_Gene, 
            tagert_col_name=tagert_col_name, 
            plot_col_name=plot_col_name, 
            p_value=p_value,
            figsize=(4, 3),
            ax=current_ax,
            Day=Day
        )
        ax_index += 1
        current_ax.set_xlabel('Effect on Program Expression (log2 fold-change)', fontsize=14, fontweight='bold', loc='center')
        current_ax.set_ylabel("Program Name", fontsize=14, fontweight='bold', loc='center')
        current_ax.set_title(f"Program Log2 Fold-Change, {Day} \n {Target_Gene}", fontsize=20, fontweight='bold', loc='center')

        # Plot 2: Volcano plot
        current_ax = axes[ax_index]
        current_ax, program, text = plot_volcano(
            perturb_path=file_name, 
            Target=Target_Gene, 
            down_thred_log=down_thred_log, 
            up_thred_log=up_thred_log, 
            p_value=p_value,
            figsize=(5, 3), 
            ax=current_ax,
            Day=Day
        )
        ax_index += 1
        current_ax.set_title(f"Volcano Plot, {Day} \n {Target_Gene}", fontsize=20, fontweight='bold', loc='center')
        current_ax.set_xlabel('Effect on Program Expression (log2 fold-change)', fontsize=14, fontweight='bold', loc='center')
        current_ax.set_ylabel("Adjusted p-value, -log10", fontsize=14, fontweight='bold', loc='center')
        for t in text:
            t.set_fontsize(14)
        adjust_text(text, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=current_ax, fontsize=14) 

        # Plot 3: Programs dotplot
        current_ax = axes[ax_index]
        current_ax = programs_dotplot(
            mdata=mdata, 
            program_list=df["program_name"].tolist(),
            Target=Target_Gene,
            figsize=(5, 3),
            ax=current_ax,
            Day=Day
        )
        ax_index += 1
        current_ax.set_title(f"Perturbed Program Loading Scores, {Day} \n {Target_Gene}", fontsize=18, fontweight='bold', loc='center')
        current_ax.set_ylabel('Program Name', fontsize=14, fontweight='bold', loc='center')
        current_ax.set_xlabel("Day", fontsize=14, fontweight='bold', loc='center')
        
        # Plot 4: Waterfall plot 
        current_ax = axes[ax_index]
        current_ax, text = create_gene_correlation_waterfall(
            perturb_path=file_name,
            Target_Gene=Target_Gene,
            ax=current_ax,
            Day=Day,
            figsize=(3, 4),
        )
        ax_index += 1
        current_ax.set_title(f"Gene Perturbation Correlation, {Day} \n {Target_Gene}", fontsize=18, fontweight='bold', loc='center')
        current_ax.set_ylabel(f'Correlation of Program\nExpression with\n{Target_Gene} Perturbation', fontsize=14, fontweight='bold', loc='center')
        current_ax.set_xlabel('Perturbed Genes', fontsize=14, fontweight='bold', loc='center')

        for t in text:
            t.set_fontsize(14)
        adjust_text(text, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=current_ax, fontsize=14) 

    # Add main title
    fig.suptitle(f'Comprehensive Analysis: {Target_Gene}', fontweight='bold', y=0.995, fontsize=30)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path and save_name and PDF:
        full_path = f"{save_path}/{save_name}.pdf"
        print(f"Saving figure to {full_path}...")
        fig.savefig(full_path, format='pdf', bbox_inches='tight', dpi=300)
    
    if save_path and save_name and not PDF:
        full_path = f"{save_path}/{save_name}.svg"
        print(f"Saving figure to {full_path}...")
        fig.savefig(full_path, format='svg', bbox_inches='tight', dpi=300)
    
    # Control whether to display the plot
    if show:
        plt.show()
    
    # Always close the figure to prevent memory leak
    plt.close(fig)




# parallel run - single genes
def process_single_gene(Target_Gene,     
                        mdata,
                        perturb_path,
                        file_to_dictionary,
                        top_program=10,
                        species="human", 
                        groupby='sample',
                        tagert_col_name="target_name",
                        plot_col_name="program_name",
                        log2fc_col='log2FC',
                        top_num=5,
                        down_thred_log=-0.05,
                        up_thred_log=0.05,
                        p_value=0.05,
                        save_path=None,
                        save_name=None,
                        figsize=None,
                        samples=None,
                        square_plots=True,
                        show=True,
                        PDF=True):

    try:
        print(f"Processing gene: {Target_Gene}")

        create_comprehensive_plot(
            mdata=mdata,
            perturb_path=perturb_path,
            Target_Gene=Target_Gene,
            file_to_dictionary=file_to_dictionary,
            top_program=top_program,
            species=species,
            groupby=groupby,
            tagert_col_name=tagert_col_name,
            plot_col_name=plot_col_name,
            log2fc_col=log2fc_col,
            top_num=top_num,
            down_thred_log=down_thred_log,
            up_thred_log=up_thred_log,
            p_value=p_value,
            save_path=save_path,
            save_name=gene,
            figsize=figsize,
            samples=samples,
            square_plots=square_plots,
            show=show,
            PDF=PDF
        )

        return f"Success: {Target_Gene}"

    except Exception as e:
        print(f"Error processing {Target_Gene}: {str(e)}")
        return f"Error: {Target_Gene} - {str(e)}"




# parallel run - multi genes
def parallel_gene_processing(perturbed_gene_list, 
                        mdata,
                        perturb_path,
                        file_to_dictionary,
                        top_program=10,
                        species="human", 
                        groupby='sample',
                        tagert_col_name="target_name",
                        plot_col_name="program_name",
                        log2fc_col='log2FC',
                        top_num=5,
                        down_thred_log=-0.05,
                        up_thred_log=0.05,
                        p_value=0.05,
                        save_path=None,
                        save_name=None,
                        figsize=None,
                        samples=None,
                        square_plots=True,
                        show=True,
                        PDF=True,
                        n_processes=-1):

    if n_processes == -1:
        n_processes = os.cpu_count()


    # Create partial function with fixed parameters
    process_func = partial(
            process_single_gene,
            mdata=mdata,
            perturb_path=perturb_path,
            Target_Gene=Target_Gene,
            file_to_dictionary=file_to_dictionary,
            top_program=top_program,
            species=species,
            groupby=groupby,
            tagert_col_name=tagert_col_name,
            plot_col_name=plot_col_name,
            log2fc_col=log2fc_col,
            top_num=top_num,
            down_thred_log=down_thred_log,
            up_thred_log=up_thred_log,
            p_value=p_value,
            save_path=save_path,
            save_name=gene,
            figsize=figsize,
            samples=samples,
            square_plots=square_plots,
            show=show,
            PDF=PDF
        )

    print(f"Starting parallel processing of {len(perturbed_gene_list)} genes using {n_processes} processes...")

    # Use multiprocessing Pool for SLURM compatibility
    with ThreadPoolExecutor(max_workers=n_processes) as executor:
        results = list(executor.map(process_func, perturbed_gene_list))

    print("Parallel processing completed!")

    return results
  
  