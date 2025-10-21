import os
import gin
import argparse
import yaml

import mudata
import muon
import anndata
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
import sys
import mygene
import shutil
from pathlib import Path

# torch consensus NMF  -> https://github.com/ymo6/torch_based_cNMF
# consensus NMF ->     https://github.com/dylkot/cNMF

def init_cnmf_obj(output_dir=None, run_name=None):
    
    from cnmf import cNMF

    # Compute cNMF and create prog anndata
    cnmf_obj = cNMF(output_dir=output_dir, name=run_name)

    return cnmf_obj


def run_cnmf_factorization( mode="torch_cnmf",
                            output_dir=None, 
                            run_name ="consensus_NMF",
                            counts_fn=None,
                            components=[30,50,60,80,100,200,250,300], 
                            densify=False,
                            tpm_fn=None,
                            genes_file=None,
                            n_iter=10, 
                            init="random",
                            algo="halsvar",
                            beta_loss='frobenius',
                            num_highvar_genes=2000, 
                            seed=14,
                            total_workers=-1, 
                            sel_thresh=[0.2, 2.0],
                            tol=1e-4,
                        
                        # torch_cnmf specific parameters
                            use_gpu=True,
                            n_jobs=-1,
                            batch_mode = "batch",
                            alpha_usage=0, 
                            alpha_spectra=0,
                            l1_ratio_usage=0, 
                            l1_ratio_spectra=0, 
                            online_usage_tol=0.005,
                            online_spectra_tol=0.005, 
                            fp_precision='float',
                            batch_max_iter=1000, 
                            batch_hals_tol=0.005, 
                            batch_hals_max_iter=1000,
                            online_max_pass=200, 
                            online_chunk_size=5000, 
                            online_chunk_max_iter=1000,
                            shuffle_cells=False,
                            sk_cd_refit=True):

    if mode=='torch_cnmf':
                    
        # Compute cNMF and create prog anndata

        cnmf_obj = init_cnmf_obj(output_dir=output_dir, run_name=run_name)
        cnmf_obj.prepare(counts_fn=counts_fn, 
                        components=components,
                        n_iter=n_iter, 
                        densify=densify, 
                        tpm_fn=tpm_fn, 
                        num_highvar_genes=num_highvar_genes, 
                        genes_file=genes_file,
                        init=init,
                        beta_loss=beta_loss, 
                        algo=algo, 
                        tol=tol,
                        n_jobs=n_jobs, 
                        seed=seed, 
                        use_gpu=use_gpu,
                        mode = batch_mode,
                        alpha_usage=alpha_usage, 
                        alpha_spectra=alpha_spectra,
                        l1_ratio_usage=l1_ratio_usage, 
                        l1_ratio_spectra=l1_ratio_spectra, 
                        online_usage_tol=online_usage_tol,
                        online_spectra_tol=online_spectra_tol, 
                        fp_precision=fp_precision,
                        batch_max_iter=batch_max_iter, 
                        batch_hals_tol=batch_hals_tol, 
                        batch_hals_max_iter=batch_hals_max_iter,
                        online_max_pass=online_max_pass, 
                        online_chunk_size=online_chunk_size, 
                        online_chunk_max_iter=online_chunk_max_iter,
                        shuffle_cells=shuffle_cells,
                        sk_cd_refit=sk_cd_refit)
        cnmf_obj.factorize()
    
    elif mode=='standard':

        # Compute cNMF and create prog anndata
        cnmf_obj = init_cnmf_obj(output_dir=output_dir, name=run_name)
        cnmf_obj.prepare(counts_fn=counts_fn, components=components, n_iter=n_iter, 
                        seed=seed, num_highvar_genes=num_highvar_genes, beta_loss=beta_loss,
                        tol=tol, init = init, max_NMF_iter = batch_max_iter)

                        
        # FIXME: Doesn't seem to work in multithreading
        cnmf_obj.factorize(total_workers=total_workers)

    return cnmf_obj


def run_cnmf_consensus(cnmf_obj=None, output_dir=None, run_name=None, 
                       components=[7,8,9,10], sel_thresh=[0.01, 0.05, 2.0]):
    
    #TODO: Usually this step has to be rerun with a manually chosen density threshold.

    if cnmf_obj is None:
        cnmf_obj = init_cnmf_obj(output_dir=output_dir, name=run_name)

    for k in tqdm(components, desc='Running cNMF'):
        for thresh in sel_thresh:
            cnmf_obj.consensus(k=k, density_threshold=thresh, show_clustering=True)


@gin.configurable
def run_consensus_NMF_(K=10, 
                       output_dir=None, 
                       mode="torch_cnmf",
                       run_name ="consensus_NMF",
                       counts_fn=None,
                       components=[30,50,60,80,100,200,250,300], 
                       densify=False,
                       tpm_fn=None,
                       genes_file=None,
                       n_iter=10, 
                       init="random",
                       algo="halsvar",
                       beta_loss='frobenius',
                       num_highvar_genes=2000, 
                       seed=14,
                       total_workers=-1, 
                       sel_thresh=[0.2, 2.0],
                       tol=1e-4,
                    
                       # torch_cnmf specific parameters
                        use_gpu=True,
                        n_jobs=-1,
                        batch_mode = "batch",
                        alpha_usage=0, 
                        alpha_spectra=0,
                        l1_ratio_usage=0, 
                        l1_ratio_spectra=0, 
                        online_usage_tol=0.005,
                        online_spectra_tol=0.005, 
                        fp_precision='float',
                        batch_max_iter=1000, 
                        batch_hals_tol=0.005, 
                        batch_hals_max_iter=1000,
                        online_max_pass=200, 
                        online_chunk_size=5000, 
                        online_chunk_max_iter=1000,
                        shuffle_cells=False,
                        sk_cd_refit=True):

    
    """
    Perform gene program inference using consensus NMF.

    Supports both standard cNMF and torch-based cNMF implementations.
    https://github.com/dylkot/cNMF (standard)
    https://github.com/ymo6/torch_based_cNMF (torch_cnmf)

    Args:
        mdata (MuData): 
            MuData object containing anndata of data and cell-level metadata.
        
        # I/O Parameters
        output_dir (str, default='./'): 
            Path to directory to store outputs.
        run_name(str, default='consensus_NMF'): 
            File that stores the outputs.
        counts_fn (str, default=None): 
            File that contains input data. 


        # General cNMF Parameters
        mode (str, default='torch_cnmf'): 
            Use torch-based cNMF ('torch_cnmf') or standard cNMF ('standard').
        inplace (bool, default=True): 
            Update the mudata object inplace or return a copy.
        sel_thresh (list, default=[0.2, 2.0]): 
            Density thresholds for filtering NMF runs during consensus step.
        component (list, default=[30, 50, 100]): 
            Values of K (number of components) to run NMF for.
        n_iter (int, default=10): 
            Number of iterations for factorization.
        seed (int, default=14): 
            Random seed for reproducibility.
        beta_loss (str, default='frobenius'): 
            Beta loss metric. Options: 'frobenius' (L2), 'kullback-leibler' (KL), 'itakura-saito' (IS).
        num_highvar_genes (int, default=2000): 
            Number of highly variable genes to use if genes_file is None.
        
        # Data Processing Parameters
        densify (bool, default=False): 
            Convert sparse data to dense format.
        tpm_fn (str, default=None): 
            Path to TPM data file. If None, computed from counts.
        genes_file (str, default=None): 
            Path to file containing high-variance genes list.
        
        # Algorithm Parameters
        init (str, default='random'): 
            Initialization method: 'random', 'nndsvd', 'nndsvda', 'nndsvdar'.
        algo (str, default='halsvar'): 
            Algorithm choice: 'mu' (multiplicative update), 'halsvar' (HALS variant).
        tol (float, default=1e-4): 
            Tolerance for convergence check.
        total_workers (int, default=1): 
            Number of CPU threads to run each NMF. If -1, use all available.

        
        # Torch-specific Parameters (only used when mode='torch_cnmf')
        use_gpu (bool, default=True): 
            Whether to use GPU acceleration (torch_cnmf only).
        batch_mode (str, default='batch'): 
            Learning mode: 'batch' or 'online'. Online only works with beta_loss='frobenius'.
        n_jobs (int, default=1): 
            Number of CPU threads to use for torch-cnmf. If -1, use all available.
        
        # Regularization Parameters (torch_cnmf)
        alpha_usage (float, default=0): 
            L2 regularization parameter for usage matrix (W).
        alpha_spectra (float, default=0): 
            L2 regularization parameter for spectra matrix (H).
        l1_ratio_usage (float, default=0): 
            L1 penalty ratio for W (0-1). L2 ratio = (1 - l1_ratio_usage).
        l1_ratio_spectra (float, default=0): 
            L1 penalty ratio for H (0-1). L2 ratio = (1 - l1_ratio_spectra).
        
        # Batch Learning Parameters (torch_cnmf)
        fp_precision (str, default='float'): 
            Numeric precision: 'float' (torch.float) or 'double' (torch.double).
        batch_max_iter (int, default=500): 
            Maximum iterations for batch learning.
        batch_hals_tol (float, default=0.05): 
            Tolerance for HALS - maximal relative change threshold.
        batch_hals_max_iter (int, default=200): 
            Maximum iterations for H & W updates. Set to 1 for standard HALS.
        
        # Online Learning Parameters (torch_cnmf)
        online_max_pass (int, default=20): 
            Maximum number of passes through all data in online mode.
        online_chunk_size (int, default=5000): 
            Mini-batch size for online learning.
        online_chunk_max_iter (int, default=200): 
            Maximum iterations for H or W updates in online learning.
        online_usage_tol (float, default=0.05): 
            Tolerance for updating W in each chunk during online learning.
        online_spectra_tol (float, default=0.05): 
            Tolerance for updating H in each chunk during online learning.
        shuffle_cells (bool, default=False): 
            Shuffle cells before online learning (recommended for online mode).
        sk_cd_refit (bool, default=True): 
            Use scikit-learn coordinate descent refit if True, torch refit if False.

    Returns:
        MuData or None: If not inplace, returns modified MuData object.
    
    """

    cnmf_obj = run_cnmf_factorization(mode=mode,
                                      output_dir=output_dir, 
                                      run_name=run_name, 
                                      counts_fn=counts_fn,
                                      components=components, 
                                      n_iter=n_iter, 
                                      seed=seed,
                                      total_workers=total_workers, 
                                      num_highvar_genes=num_highvar_genes, 
                                      beta_loss=beta_loss, 
                                      use_gpu=use_gpu, 
                                      densify=densify, 
                                      tpm_fn=tpm_fn, 
                                      genes_file=genes_file, 
                                      init=init,
                                      algo=algo, 
                                      tol=tol, 
                                      n_jobs=n_jobs,
                                      batch_mode=batch_mode,
                                      alpha_usage=alpha_usage, 
                                      alpha_spectra=alpha_spectra,
                                      l1_ratio_usage=l1_ratio_usage, 
                                      l1_ratio_spectra=l1_ratio_spectra,
                                      online_usage_tol=online_usage_tol, 
                                      online_spectra_tol=online_spectra_tol,
                                      fp_precision=fp_precision, 
                                      batch_max_iter=batch_max_iter,
                                      batch_hals_tol=batch_hals_tol, 
                                      batch_hals_max_iter=batch_hals_max_iter,
                                      online_max_pass=online_max_pass, 
                                      online_chunk_size=online_chunk_size,
                                      online_chunk_max_iter=online_chunk_max_iter, 
                                      shuffle_cells=shuffle_cells)

    cnmf_obj.combine()
    cnmf_obj.k_selection_plot()

    # Plot & store for many 
    run_cnmf_consensus(cnmf_obj=cnmf_obj, 
                       components=components, 
                       sel_thresh=sel_thresh)
    
    return cnmf_obj, K, components 


def run_consensus_NMF(mdata,
                      work_dir='./',
                      scratch_dir=None,  
                      run_name="consensus_NMF",
                      prog_key='consensus_NMF', 
                      data_key='rna', 
                      layer='X', 
                      config_path=None, 
                      mode = "torch_cnmf",
                      inplace=True,
                      sel_thresh = [0.2, 2.0],
                      output_all_k = True,
                      output_all_thresh=True):


    # Load method specific parameters
    try: gin.parse_config_file(config_path)
    except: raise ValueError('gin config file could not be found')

    if not inplace:
        mdata = mudata.MuData({data_key: mdata[data_key].copy()})

    # Create output directory for cNMF results
    if work_dir is not None:
        try: os.makedirs(work_dir, exist_ok=True)
        except: raise ValueError('Work directory location does not exist.')

    # Store temporary anndata
    if scratch_dir==None:
        scratch_dir=work_dir

    # Create temp anndata 
    if layer=='X':
        counts_path = os.path.join(f"{scratch_dir}/{run_name}", '{}_temp.h5ad'.format(data_key))
        temp_data = mdata[data_key].copy()
        temp_data.var_names_make_unique()
        temp_data.write(counts_path)
    else:
        temp_data = anndata.AnnData(data=mdata[data_key].layers[layer], 
                                    obs=mdata[data_key].obs,
                                    var=mdata[data_key].var)
        counts_path = os.path.join(f"{scratch_dir}/{run_name}", '{}_{}_temp.h5ad'.format(data_key, layer))
        temp_data.var_names_make_unique()
        temp_data.write(counts_path)

  
    # Compute cNMF and create prog anndata
    cnmf_obj, K, components = \
    run_consensus_NMF_(output_dir=work_dir, 
                       counts_fn=counts_path,
                       run_name=run_name,
                       mode=mode,
                       sel_thresh=sel_thresh)

    # Create new anndata object
    usage, spectra_scores, spectra_tpm, top_genes = \
    cnmf_obj.load_results(K=K, density_threshold=min(sel_thresh))

    mdata.mod[prog_key] = anndata.AnnData(X=usage, obs=mdata[data_key].obs)
    mdata[prog_key].uns['loadings'] = spectra_tpm
    mdata[prog_key].uns['loadings_zscore'] = spectra_scores
    mdata[prog_key].uns['loadings_genes'] = top_genes

    # Store outputs in mdata
    if not output_all_k:
        components = [K]
    if not output_all_thresh:
        sel_thresh = [min(sel_thresh), max(sel_thresh)]
    adatas = {}
    for k in tqdm(components, desc='Storing output'):
        for thresh in sel_thresh:
            usage, spectra_scores, spectra_tpm, top_genes = \
            cnmf_obj.load_results(K=k, density_threshold=thresh)

            adata_ = anndata.AnnData(X=usage, obs=mdata[data_key].obs)
            adata_.uns['loadings'] = spectra_tpm
            adata_.uns['loadings_zscore'] = spectra_scores
            adata_.uns['loadings_genes'] = top_genes

            adatas[prog_key+'_{}_{}'.format(k, thresh)] = adata_
    
    adatas[data_key] = mdata[data_key]
    adatas[prog_key] = mdata[prog_key]

    mdata = mudata.MuData(adatas)

    if not inplace: return mdata


# compile results into correct format for downstream Evaluation pipeline
def compile_results(output_directory, run_name, sel_thresh = 2.0, components = [30, 50, 60, 80, 100, 200, 250, 300] ):
       
    for k in components:

        scores = pd.read_csv('{output_directory}/{run_name}/{run_name}.usages.k_{k}.dt_{sel_thresh}.consensus.txt'.format(
                                                                                        output_directory=output_directory,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = str(sel_thresh).replace('.','_')),
                                                                                        sep='\t', index_col=0)

        loadings = pd.read_csv('{output_directory}/{run_name}/{run_name}.spectra.k_{k}.dt_{sel_thresh}.consensus.txt'.format(
                                                                                        output_directory=output_directory,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = str(sel_thresh).replace('.','_')),
                                                                                        sep='\t', index_col=0)
        

        os.makedirs((f'{output_directory}/{run_name}/loading'), exist_ok=True)


        scores.to_csv('{output_directory}/{run_name}/loading/cNMF_scores_{k}_{sel_thresh}.txt'.format(
                                                                                        output_directory=output_directory,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = sel_thresh), sep='\t')
                                                                                        
        loadings.T.to_csv('{output_directory}/{run_name}/loading/cNMF_loadings_{k}_{sel_thresh}.txt'.format(     
                                                                                        output_directory=output_directory,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = sel_thresh), sep='\t')

        adata_ = anndata.read_h5ad('{output_directory}/{run_name}/cnmf_tmp/{run_name}.tpm.h5ad'.format(
                                                                                        output_directory=output_directory,
                                                                                        run_name = run_name,
                                                                                        k=k ))
        adata_.var_names_make_unique()
        adata_.obs_names_make_unique()

        prog_data = anndata.AnnData(X=scores.values, obs=adata_.obs)
        prog_data.varm['loadings'] = loadings.values
        prog_data.uns['var_names'] = loadings.columns.values


        # Make adata
        os.makedirs((f'{output_directory}/{run_name}/prog_data'), exist_ok=True)
        prog_data.write(f'{output_directory}/{run_name}/prog_data/NMF_{k}_{sel_thresh}.h5ad'.format(
                                                                                output_directory=output_directory,
                                                                                run_name = run_name,
                                                                                k=k,
                                                                                sel_thresh = str(sel_thresh).replace('.','_')))

        # Make mdata
        mdata = muon.MuData({'rna': adata_, 'cNMF': prog_data})

        if "guide_names" in adata_.uns:
            mdata['cNMF'].uns["guide_names"] = adata_.uns["guide_names"]

        if "guide_targets" in adata_.uns:
            mdata['cNMF'].uns["guide_targets"] = adata_.uns["guide_targets"]

        if "guide_assignment" in adata_.obsm:
            mdata['cNMF'].obsm["guide_assignment"] = adata_.obsm["guide_assignment"]

        os.makedirs((f'{output_directory}/{run_name}/mdata'), exist_ok=True)
        mdata.write(f'{output_directory}/{run_name}/adata/cNMF_{k}_{sel_thresh}.h5mu'.format(
                                                                                output_directory=output_directory,
                                                                                run_name = run_name,
                                                                                k=k,
                                                                                sel_thresh = str(sel_thresh).replace('.','_')))


# given a df from cNMF, return top 300 gene for each program in df
def get_top_indices_fast(df, gene_num = 300):
 
    # Get column names
    col_names = df.columns.values
    
    # Use argsort to get indices of top 300 values per row
    # argsort sorts in ascending order, so we use [:, -300:] and reverse
    top_indices = np.argsort(df.values, axis=1)[:, -gene_num:][:, ::-1]
    
    # Map indices to column names
    top_col_names = col_names[top_indices]
    
    # Create result DataFrame
    result_df = pd.DataFrame(
        top_col_names,
        index=df.index,
        columns=[f'top_{i+1}' for i in range(gene_num)]
    )

    result_df.index = [f'Program_{i}' for i in range(1,len(result_df)+1)]
    
    return result_df


# annotate genes in excel given a df with rows for each program, cols for genes    
def annotate_genes_to_excel(df, output_file='gene_annotations.xlsx'):
    
    # Initialize MyGene
    mg = mygene.MyGeneInfo()
    
    # Dictionary to store results for each column
    all_annotations = {}
    
    # Process each column
    for row_idx in df.index:
        # Get unique genes from column (remove NaN)
        genes = df.loc[row_idx].dropna().unique().tolist()
        
        if len(genes) == 0:
            print(f"Column '{row_idx}': No genes found")
            continue
        
        print(f"Annotating column '{row_idx}': {len(genes)} genes...")
        
        # Query MyGene for annotations
        results = mg.querymany(
            genes, 
            scopes='symbol,alias,ensembl.gene',  # Multiple search scopes
            fields='symbol,name,entrezgene,summary,type_of_gene',
            species='human',
            returnall=True
        )
        
        # Process results
        annotation_list = []
        for query_gene, gene_info in zip(genes, results['out']):
            # Handle cases where gene is not found or multiple matches
            if 'notfound' in gene_info and gene_info['notfound']:
                annotation_list.append({
                    'Input_Gene': query_gene,
                    'Gene_Symbol': 'NOT FOUND',
                    'Gene_Name': 'NOT FOUND',
                    'Entrez_ID': '',
                    'Type': '',
                    'Summary': ''
                })
            else:
                annotation_list.append({
                    'Input_Gene': query_gene,
                    'Gene_Symbol': gene_info.get('symbol', query_gene),
                    'Gene_Name': gene_info.get('name', ''),
                    'Entrez_ID': gene_info.get('entrezgene', ''),
                    'Type': gene_info.get('type_of_gene', ''),
                    'Summary': gene_info.get('summary', '')
                })
        
        # Create DataFrame for this column
        all_annotations[row_idx] = pd.DataFrame(annotation_list)

    # Export to Excel with multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for row_idx, annotations_df in all_annotations.items():
            sheet_name = str(row_idx)
            # Truncate sheet name if too long (Excel limit is 31 chars)
            sheet_name = sheet_name[:31] if len(sheet_name) > 31 else sheet_name
            annotations_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return all_annotations



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mudataObj_path', type=str, required=True)  
    parser.add_argument('--work_dir', default='./', type=str)
    parser.add_argument('--run_name', default='consensus_NMF', type=str)
    parser.add_argument('--scratch_dir', default='./', type=str)
    parser.add_argument('-pk', '--prog_key', default='consensus_NMF', type=str) 
    parser.add_argument('-dk', '--data_key', default='rna', type=str) 
    parser.add_argument('--layer', default='X', type=str)
    parser.add_argument('--config_path', default='./consensus_NMF_config.gin', type=str)
    parser.add_argument('--mode', default='standard', type=str, choices=['torch_cnmf', 'standard'])
    parser.add_argument('--output', action='store_false')
    parser.add_argument('--sel_thresh', nargs='*', type=float, default=None)
    parser.add_argument('--output_all_k', action='store_false')
    parser.add_argument('--output_all_thresh', action='store_false')


    args = parser.parse_args()

    # load data
    mdata = mudata.read(args.mudataObj_path)

    if args.sel_thresh is None:
        sel_thresh_value = [0.2, 2.0]
    else:
        sel_thresh_value = args.sel_thresh

    # run cNMF
    run_consensus_NMF(mdata, work_dir=args.work_dir,run_name=args.run_name, scratch_dir=args.scratch_dir, 
                      prog_key=args.prog_key, data_key=args.data_key, layer=args.layer, 
                      config_path=args.config_path, inplace=args.output, sel_thresh = sel_thresh_value, mode=args.mode,
                      output_all_k = args.output_all_k, output_all_thresh=args.output_all_thresh)

    # save all parameters used
    args_dict = vars(args)
    with open(f'{args.work_dir}/{args.run_name}/config.yml', 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False, width=1000)
