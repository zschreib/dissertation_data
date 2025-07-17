from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import time
from scipy import sparse
import warnings
from scipy.stats import hypergeom
import statsmodels.stats.multitest as smm
from joblib import Parallel, delayed
from itertools import combinations
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def read_tsv_file(file_path):
    return pd.read_csv(file_path, sep='\t', dtype={'ID': str})

def load_metadata(metadata_file, database: str):
    """Return a metadata DataFrame already filtered by `database` (or all if 'All')."""
    usecols = ['ID', 'Database']
    meta = pd.read_csv(metadata_file, sep='\t', usecols=usecols, dtype={'ID': str})

    if database and database != 'All':
        before = len(meta)
        meta = meta[meta['Database'] == database]
        print(f"Filtered metadata {before}→{len(meta)} rows for database={database}")
    else:
        print(f"No database filter applied. Metadata has {len(meta)} rows.")

    counts = meta.groupby('ID')['Database'].nunique()
    bad = counts[counts > 1].index
    if len(bad):
        warnings.warn(
            f"{len(bad)} IDs appear in more than one Database in metadata; "
            "those IDs will be dropped"
        )
        meta = meta[~meta['ID'].isin(bad)]

    return meta

def load_rep_block_metadata(metadata_file: str) -> pd.DataFrame:
    """
    Load only the REP entries from your metadata,
    returning a small table keyed by Cluster_Number.
    """
    usecols = [
        'Cluster_ID',
        'Cluster_Number',
        'Uniref_tophit_function',
        'ORF_category'
    ]
    meta = pd.read_csv(metadata_file, sep='\t', usecols=usecols, dtype={'Cluster_Number': int})
    rep = meta[meta['Cluster_ID'] == 'REP']
    return rep.set_index('Cluster_Number')[['Uniref_tophit_function', 'ORF_category']]


def _extract_canonical_blocks(arr: np.ndarray, cb: int):
    n = arr.size
    if n < cb:
        return set()
    win = np.lib.stride_tricks.sliding_window_view(arr, window_shape=cb)
    return {min(tuple(w), tuple(w[::-1])) for w in win}


def find_conserved_blocks(df, cb: int, min_samples: int, n_jobs: int = 1):
    """
    df: DataFrame with 'ID' and 'Block_Content'
    cb: window size
    min_samples: threshold for a block to be 'conserved'
    n_jobs: 1 for serial, -1 or >1 to parallelize extraction
    """
    ids = df['ID'].tolist()
    contents = df['Block_Content'].tolist()
    arrays = [np.fromstring(s, dtype=int, sep=',') for s in contents]

    #Extract all blocks and count them
    counter = Counter()
    if n_jobs == 1:
        sample_block_sets = []
        for arr in tqdm(arrays, desc="Extracting blocks"):
            blks = _extract_canonical_blocks(arr, cb)
            sample_block_sets.append(blks)
            counter.update(blks)
    else:
        sample_block_sets = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_extract_canonical_blocks)(arr, cb) for arr in arrays
        )
        for blks in sample_block_sets:
            counter.update(blks)

    #Build the conserved‐block→column index map
    block_to_idx = {}
    for blk, ct in counter.items():
        if ct >= min_samples:
            block_to_idx[blk] = len(block_to_idx)

    #Build the presence matrix
    row_idx, col_idx = [], []
    final_sample_sets = {}

    for i, sid in enumerate(tqdm(ids, desc="Building matrix")):
        blks = sample_block_sets[i]
        conserved_blks = {b for b in blks if b in block_to_idx}
        final_sample_sets[sid] = conserved_blks
        for b in conserved_blks:
            j = block_to_idx[b]
            row_idx.append(i)
            col_idx.append(j)

    n_rows = len(ids)
    n_cols = len(block_to_idx)
    if col_idx:
        max_j = max(col_idx)
        if max_j >= n_cols:
            warnings.warn(
                f"max column index {max_j} >= allocated columns {n_cols}; "
                f"expanding to {max_j + 1}"
            )
            n_cols = max_j + 1

    presence = sparse.coo_matrix(
        (np.ones(len(row_idx), int), (row_idx, col_idx)),
        shape=(n_rows, n_cols),
        dtype=int
    ).tocsr()

    conserved_ids = [ids[i] for i in sorted(set(row_idx))]

    return {
        'all_samples': ids,
        'sample_sets': final_sample_sets,
        'block_counts': counter,
        'block_to_idx': block_to_idx,
        'conserved_ids': conserved_ids,
        'presence_matrix': presence,
        'total_windows': sum(len(s) for s in sample_block_sets),
        'unique_blocks': len(counter),
        'conserved_blocks': len(block_to_idx),
        'samples_with_conserved': len(conserved_ids),
    }

def build_bipartite_graph(sample_sets: dict,
                          block_to_idx: dict,
                          rep_meta: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    for sid in sorted(sample_sets):
        G.add_node(sid, type='sample', bipartite=0)

    for blk in sorted(block_to_idx):
        name = "_".join(map(str, blk))
        G.add_node(name, type='block', bipartite=1)

        clusters = list(blk)
        funcs = [
            rep_meta.loc[c, 'Uniref_tophit_function']
            for c in clusters
            if c in rep_meta.index
        ]
        cats = [
            rep_meta.loc[c, 'ORF_category']
            for c in clusters
            if c in rep_meta.index
        ]

        G.nodes[name]['clusters'] = ",".join(map(str, clusters))
        G.nodes[name]['functions'] = ",".join(funcs)
        G.nodes[name]['categories'] = ",".join(cats)

    for sid in sorted(sample_sets):
        for blk in sorted(sample_sets[sid]):
            if blk in block_to_idx:
                name = "_".join(map(str, blk))
                G.add_edge(sid, name)

    return G

def detect_syntenome(G: nx.Graph,
                     sample_sets: dict) -> dict:
    """
    sample_sets: { sample_id: set(block1, block2, …) }
    G: bipartite graph

    1) Build sample–sample weighted projection by
       iterating over each block’s sample list.
    2) Run Louvain on that smaller graph.
    3) Write the 'syntenome' attribute back onto G’s sample nodes.
    """
    #Invert sample_sets → block → [samples]
    block_to_samples = defaultdict(list)
    for sid, blks in sample_sets.items():
        for blk in blks:
            block_to_samples[blk].append(sid)

    #Accumulate pairwise co-occurrence counts
    weight_dict = defaultdict(int)
    for blk, samples in tqdm(block_to_samples.items(), desc="Projecting samples"):
        # only pairs of samples that share this block
        for u, v in combinations(samples, 2):
            # ensure an ordering so (u,v) == (v,u)
            if u > v:
                u, v = v, u
            weight_dict[(u, v)] += 1

    #Build the sample–sample graph
    S = nx.Graph()
    S.add_nodes_from(sample_sets.keys())
    for (u, v), w in weight_dict.items():
        S.add_edge(u, v, weight=w)

    #Louvain
    try:
        import community.community_louvain as community_louvain
    except ImportError:
        raise ImportError("Install python-louvain (`pip install python-louvain`) to detect syntenome.")
    partition = community_louvain.best_partition(S, weight='weight', random_state=SEED)

    nx.set_node_attributes(G, partition, 'syntenome')
    return partition

def build_block_jaccard_graph(G_full: nx.Graph,
                              sample_sets: dict,
                              threshold: float = 0.0) -> nx.Graph:
    """
    From the full bipartite graph G_full and its sample→blocks map,
    build a block–block graph where edge weights = Jaccard index,
    and only edges with jaccard >= threshold are kept.

    Jaccard index between two cluster blocks A and B:
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    #Invert to block → set(samples)
    block_to_samps = defaultdict(set)
    for samp, blks in sample_sets.items():
        for blk in blks:
            block_to_samps[blk].add(samp)

    #Copy over block nodes (and their attributes) into a new graph
    Gj = nx.Graph()
    for node, data in G_full.nodes(data=True):
        if data.get('type') == 'block':
            Gj.add_node(node, **data)

    #Compute pairwise Jaccard and add edges
    blocks = list(block_to_samps)
    for i, b1 in enumerate(blocks):
        s1 = block_to_samps[b1]
        for b2 in blocks[i + 1:]:
            s2 = block_to_samps[b2]
            union = len(s1 | s2)
            if union == 0:
                continue
            jci = len(s1 & s2) / union
            if jci >= threshold:
                n1 = "_".join(map(str, b1))
                n2 = "_".join(map(str, b2))
                Gj.add_edge(n1, n2, jaccard=jci)

    return Gj

def enrich_block_in_syntenome(G: nx.Graph,
                              partition: dict) -> pd.DataFrame:
    """
    For each syntenome and each block node in G, perform a conditional one-tailed
    hypergeometric 144349_with_92161.fa to assess statistical enrichment or depletion.

    The 144349_with_92161.fa compares the proportion of syntenome samples connected to a block
    against the proportion in the full sample population:
        - If the block is more common in the syntenome than expected, an enrichment
          (right-tailed) 144349_with_92161.fa is performed.
        - If the block is less common, a depletion (left-tailed) 144349_with_92161.fa is performed.
        - If it’s exactly equal, record a ‘neutral’ result with p=1.

    Benjamini–Hochberg FDR correction is applied across all tests.
    """

    # ensure every sample node is in a syntenome to avoid data skew.
    all_samples = sorted(n for n, d in G.nodes(data=True) if d.get('type') == 'sample')
    missing = set(all_samples) - set(partition.keys())
    if missing:
        warnings.warn(f"Partition missing for samples {sorted(missing)}; these will be skipped")
    samples = [s for s in all_samples if s in partition]

    M = len(samples)
    data = []

    for com in sorted(set(partition[s] for s in samples)):
        comm_samps = [s for s in samples if partition[s] == com]
        N = len(comm_samps)

        for blk in sorted(n for n, d in G.nodes(data=True) if d.get('type') == 'block'):
            K = sum(1 for s in samples if G.has_edge(s, blk))
            k = sum(1 for s in comm_samps if G.has_edge(s, blk))

            ratio_syn = k / N if N else 0
            ratio_pop = K / M if M else 0

            if ratio_syn > ratio_pop:
                p, direction = hypergeom.sf(k - 1, M, K, N), 'enriched'
            elif ratio_syn < ratio_pop:
                p, direction = hypergeom.cdf(k, M, K, N), 'depleted'
            else:
                p, direction = 1.0, 'neutral'
            #rename table fields
            data.append({
                'syntenome': com,
                'block': blk,
                'n_in_syn': k,
                'n_in_pop': K,
                'syn_size': N,
                'pop_size': M,
                'direction': direction,
                'p_value': p
            })

    df = pd.DataFrame(data)
    rej, qvals, _, _ = smm.multipletests(df['p_value'], alpha=0.05, method='fdr_bh')
    df['q_value'] = qvals
    df['significant'] = rej
    return df

def prune_graph_for_drawing(G: nx.Graph) -> nx.Graph:
    """
    Return a copy of G with sample nodes that connect to ≤2 block removed,
    and any block nodes that become isolated also removed.
    """
    G_vis = G.copy()
    # drop samples with only one or zero edges
    for n, d in list(G_vis.nodes(data=True)):
        if d.get('type') == 'sample' and G_vis.degree(n) <= 1:
            G_vis.remove_node(n)
    # drop blocks that lost all their samples
    for n, d in list(G_vis.nodes(data=True)):
        if d.get('type') == 'block' and G_vis.degree(n) == 0:
            G_vis.remove_node(n)
    return G_vis

def write_jaccard_tsv(G: nx.Graph, tsv_path: str):
    """
    Write every undirected edge in G as three-column TSV:
    block1<TAB>block2<TAB>jaccard_score
    """
    with open(tsv_path, 'w') as out:
        out.write("block1\tblock2\tjaccard_score\n")
        for u, v, data in G.edges(data=True):
            # ensure a consistent ordering
            b1, b2 = sorted([u, v])
            j = data.get('jaccard', 0.0)
            out.write(f"{b1}\t{b2}\t{j:.6f}\n")

def write_graphml(G: nx.Graph, output_file: str):
    """Dump the full bipartite graph (with all attributes) to GraphML."""
    nx.write_graphml(G, output_file)


def main(input_file, metadata_file, output: str, cb: int, min_samples: int, jaccard: int, database: str):
    start_time = time.time()

    print("Reading input…")
    df = read_tsv_file(input_file)

    print("Loading metadata…")
    meta = load_metadata(metadata_file, database)

    valid_ids = set(meta['ID'])
    df = df[df['ID'].isin(valid_ids)].reset_index(drop=True)

    results = find_conserved_blocks(df, cb, min_samples, n_jobs=-1)

    orig = len(results['sample_sets'])
    results['sample_sets'] = {
        sid: blks
        for sid, blks in results['sample_sets'].items()
        if blks
    }
    dropped = orig - len(results['sample_sets'])
    if dropped:
        print(f"Dropping {dropped} isolated samples with no conserved blocks")

    results['all_samples'] = list(results['sample_sets'])

    rep_meta = load_rep_block_metadata(metadata_file)

    # General stats
    print(f"Total samples processed after metadata‐filter: {len(df)}")
    print(f"Total windows extracted:                 {results['total_windows']}")
    print(f"Unique cluster‐blocks found:             {results['unique_blocks']}")
    print(f"Blocks ≥{min_samples}:                    {results['conserved_blocks']}")
    print(f"Samples with ≥1 conserved block:         {results['samples_with_conserved']}")

    G = build_bipartite_graph(
        sample_sets=results['sample_sets'],
        block_to_idx=results['block_to_idx'],
        rep_meta=rep_meta
    )

    partition = detect_syntenome(
        G,
        sample_sets=results['sample_sets']
    )

    J = build_block_jaccard_graph(
        G_full=G,
        sample_sets=results['sample_sets'],
        threshold=jaccard
    )

    # Enrichment
    block_enrich = enrich_block_in_syntenome(G, partition)

    # Outputs
    G_vis = prune_graph_for_drawing(G)
    write_graphml(G_vis, output + '.graphml')
    write_graphml(J, output + '_block_jaccard.graphml')
    write_jaccard_tsv(J, output + "_block_jaccard.tsv")

    block_enrich.to_csv(output + '_block_enrichment.tsv', sep='\t', index=False)

    # 1) Build a map: syntenome → set of enriched blocks in that syntenome
    enriched_map = (
        block_enrich
        .query("significant & direction=='enriched'")
        .groupby('syntenome')['block']
        .apply(set)
        .to_dict()
    )

    # 2) For each sample, find its enriched block-neighbors and syntenome
    rows = []
    for sample in sorted(partition):
        com = partition[sample]
        # all block-nodes adjacent to this sample
        blocks = [nbr for nbr in G.neighbors(sample)
                  if G.nodes[nbr].get('type') == 'block']
        # filter to only those enriched in this sample’s syntenome
        enriched = [b for b in blocks if b in enriched_map.get(com, set())]
        rows.append({
            'Sample_ID': sample,
            'Enriched_Cluster_Blocks': "(" + ", ".join(enriched) + ")",
            'Syntenome': com
        })

    # 3) Write it out
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output + '_sample_enriched_blocks.tsv', sep='\t', index=False)

    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze BATS genome synteny using k-mers')
    parser.add_argument('--input', required=True, help='Input TSV file with cluster block data')
    parser.add_argument('--metadata', required=True, help='Metadata TSV file')
    parser.add_argument('--output', default='network_output', help='Output prefix for network files')
    parser.add_argument('--cb', type=int, default=3, help='Cluster-block size. Default = 3')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='Minimum genomes for conserved cluster-block. Default = 5')
    parser.add_argument('--jaccard', type=float, default=0.25,
                        help='Jaccard threshold for cluster-block co-occurrence. Default = 0.25')
    parser.add_argument('--database', default='All',
                        help="Database to filter by (exact match). Use 'All' to disable filtering.")

    args = parser.parse_args()
    main(args.input, args.metadata, args.taxonomy, args.output, args.cb, args.min_samples, args.jaccard, args.database)
