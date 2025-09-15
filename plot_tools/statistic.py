import numpy as np
import healpy as hp
from matplotlib.path import Path
from astropy.table import Table
from plot_util import *

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

from joblib import load as joblib_load
import torch

filename = '/home/YukaYamada/output/PFS/property/autumn_property.fits'
autumn = Table.read(filename)

filename = '/home/YukaYamada/output/PFS/property/spring_property.fits'
spring = Table.read(filename)

def mask_edge():
    # Healpix pixel centers
    nside = 256
    ra, dec = hp.pix2ang(nside, spring['healpix'], lonlat=True)

    ra = np.array(ra)
    dec = np.array(dec)

    # 四角形のポリゴン（axs[0]の赤線内）
    polygon = np.array([
        [129, -1.0],
        [129, 4.2],
        [224.5, 4.2],
        [224.5, -1.0],
        [129, -1.0]
    ])

    # 点群がポリゴン内にあるかどうかを判定
    spring_path = Path(polygon)
    points = np.vstack([ra, dec]).T
    spring_mask = spring_path.contains_points(points)

    # 選ばれた pixel index
    selected_pix = spring['healpix'][spring_mask]

    # Healpix pixel centers
    nside = 256
    ra, dec = hp.pix2ang(nside, autumn['healpix'], lonlat=True)

    ra = np.array(ra)
    ra = ra - 360*(ra>300)
    dec = np.array(dec)

    # 四角形のポリゴン（axs[0]の赤線内）
    polygon = np.array([
        [-28, 0],
        [-28, 6.1],
        [-20, 6.1],
        [-20, 5.0],
        [3.0, 5.0],
        [3.0, 4.2],
        [10.0, 4.2],
        [10.0, 5.0],
        [22.5, 5.0],
        [22.5, 4.2],
        [38, 4.2],
        [38, -5.9],
        [30, -5.9],
        [30, 0],
        [-28, 0]
    ])

    # 点群がポリゴン内にあるかどうかを判定
    autumn_path = Path(polygon)
    points = np.vstack([ra, dec]).T
    autumn_mask = autumn_path.contains_points(points)

    # 選ばれた pixel index
    selected_pix = autumn['healpix'][autumn_mask]
    return autumn[autumn_mask], spring[spring_mask]

def jackknife(autumn, spring, figure=False):
    """
    autumn: pandas dataframe
    spring: pandas dataframe
    """
    autumn_df = autumn.dropna(subset=['target'])
    ra, dec = hp.pix2ang(nside=256, ipix = autumn_df['healpix'], lonlat=True)
    ra = ra-360*(ra>300)
    
    array = -1*np.ones(len(ra))

    for i in range(8):
        array[(ra>=np.min(ra)+i*8.5)&(ra<np.min(ra)+(i+1)*8.5)] = i
    array[(ra>=np.min(ra)+i*8.5)&(dec>=-1)] = 7
    array[(ra>25)&(dec<-1)] = 8
    autumn_df['jackknife'] = array

    spring_df = spring.dropna(subset=['target'])
    ra, dec = hp.pix2ang(nside=256, ipix = spring_df['healpix'], lonlat=True)
    
    array = np.ones(len(ra))
    for i in range(11):
        array[(ra>=np.min(ra)+i*9)&(ra<np.min(ra)+(i+1)*9)] = i+9
    
    spring_df['jackknife'] = array
    
    ##############################plot
    if figure:
        colors_hex = [
            '#e41a1c',  # 赤
            '#377eb8',  # 青
            '#4daf4a',  # 緑
            '#984ea3',  # 紫
            '#ff7f00',  # オレンジ
            '#ffff33',  # 黄色
            '#a65628',  # 茶
            '#f781bf',  # ピンク
            '#000000',  # グレー
            '#66c2a5',  # 青緑
            '#fc8d62',  # オレンジピンク
            '#8da0cb',  # 青紫
            '#e78ac3',  # ピンク紫
            '#a6d854',  # 黄緑
            '#ffd92f',  # 黄色2
            '#e5c494',  # ベージュ
            '#b3b3b3',  # 薄グレー
            '#1b9e77',  # 深緑
            '#d95f02',  # 橙赤
            '#7570b3'   # 紫青
            ]

        # カラーマップとして登録
        custom_cmap = ListedColormap(colors_hex)

        fig, axs = plt.subplots(
            2, 1, figsize=(12, 8),
            gridspec_kw={"height_ratios": [1, 1], "hspace": 0.2}
            )

        coll1 = plot_map(spring_df, 'jackknife', 'spring', axs[0], vmin=0, vmax=20, cmap = custom_cmap)
        coll2 = plot_map(autumn_df, 'jackknife', 'autumn', axs[1], vmin=0, vmax=20, cmap = custom_cmap)

        axs[1].set_xlabel('RA [deg]', fontsize=20)
    
        axs[0].grid()
        axs[0].yaxis.label.set_position((axs[0].yaxis.label.get_position()[0], 0.35))
        axs[1].grid()
        path = '/home/YukaYamada/output/PFS/figure/'
        plt.subplots_adjust(top=0.7, bottom=0.3, left=0.12, right=0.95)
        plt.savefig(path+'jackknife.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(path+'jackknife.png', bbox_inches='tight', dpi=300)
        plt.show()
    
    return autumn_df, spring_df
    
    
###############################################################################################################

def jackknife_ang_ratio(
    df, key, mean_density, target_col, bin_edges_deg, 
    nside=256, nest=False, eps=1e-15
):
    """
    Estimator: ( <w_sg>^2 ) / <w_ss>,  where
      w_sg(i,j) = 0.5 * ( delta_s_i * delta_i_j + delta_s_j * delta_i_i )
      w_ss(i,j) = delta_s_i * delta_s_j
    Averages <...> are pairwise, area-pair weighted: w_pair = area_i * area_j

    Parameters
    ----------
    df : pandas.DataFrame
        Columns must include ['healpix','area','jackknife', target_col, key].
        If there are multiple rows per healpix, they will be aggregated (area-weighted).
    key : str
        Name of the systematics column (scalar per pixel).
    mean_density : float
        Global mean of target_col (same definition you used upstream).
    target_col : str
        Name of the target density column (scalar per pixel).
    bin_edges_deg : 1D array-like
        Angular bin edges in degrees (monotonic increasing).
    nside : int
        HEALPix nside (default 256).
    nest : bool
        True if 'healpix' indices are NESTED. Default False (RING).
    eps : float
        Small number to stabilize divisions.

    Returns
    -------
    x : ndarray
        Bin centers [deg].
    est_mean : ndarray
        Estimator values per bin: (<w_sg>^2)/<w_ss>
    jk_std : ndarray
        Jackknife standard deviation per bin (leave-one-out regions).
    """

    # ---- 0) Aggregate to unique healpix (area-weighted) to avoid duplicates ----
    if df['healpix'].duplicated().any():
        g = df.groupby('healpix', as_index=False)
        agg = g.apply(lambda d: pd.Series({
            'area': d['area'].sum(),
            'jackknife': d['jackknife'].iloc[0],  # assume consistent region per pixel
            target_col: np.average(d[target_col], weights=d['area']),
            key: np.average(d[key], weights=d['area'])
        })).reset_index(drop=True)
    else:
        agg = df[['healpix','area','jackknife',target_col,key]].copy()

    # ---- 1) Prepare deltas ----
    # delta_i: target density contrast (w.r.t provided mean_density)
    delta_i = agg[target_col].to_numpy() / float(mean_density) - 1.0

    # delta_s: systematics contrast relative to area-weighted global mean
    glob_mean_key = np.average(agg[key].to_numpy(), weights=agg['area'].to_numpy())
    delta_s = agg[key].to_numpy() / glob_mean_key - 1.0
    
    #random
    delta_r = np.random.normal(loc = np.mean(delta_i), scale = np.std(delta_i), size = len(agg['healpix']))

    hpix = agg['healpix'].to_numpy()
    area = agg['area'].to_numpy()
    jk_label = agg['jackknife'].to_numpy()

    # ---- 2) Geometry ----
    # Get angular vectors of pixel centers
    theta, phi = hp.pix2ang(nside, hpix, nest=nest)
    vecs = hp.ang2vec(theta, phi)

    # Binning prep
    bin_edges_deg = np.asarray(bin_edges_deg)
    nbins = len(bin_edges_deg) - 1
    bin_centers = 0.5 * (bin_edges_deg[:-1] + bin_edges_deg[1:])
    radius_deg = float(bin_edges_deg.max())
    radius_rad = np.deg2rad(radius_deg)

    # Map from pixel id -> row index
    hpix2idx = {int(p): i for i, p in enumerate(hpix)}

    # ---- 3) Precompute unique neighbor pairs and their assigned bin ----
    # We only use i<j to avoid double counting.
    pair_bins = []   # bin index per pair
    pair_i = []      # i index
    pair_j = []      # j index

    for i, p in enumerate(hpix):
        neigh = hp.query_disc(nside, vecs[i], radius_rad, inclusive=False, nest=nest)
        # Keep only neighbors present in our catalog and with j>i (unique pair)
        valid_js = []
        for pj in neigh:
            j = hpix2idx.get(int(pj), -1)
            if j > i:
                valid_js.append(j) #neighbor hielpix ID in dictionary (j>i)
        if not valid_js:
            continue

        # Angular distances (deg) between i and its valid neighbors
        # hp.rotator.angdist can take broadcasting; we stack vecs[j] properly:
        vj = vecs[np.array(valid_js)]
        ang_rad = hp.rotator.angdist(vecs[i], vj.T)  # returns array of angles
        ang_deg = np.rad2deg(ang_rad)

        # Digitize into bins
        b = np.digitize(ang_deg, bin_edges_deg) - 1 #digitize neighboring pixel distances
        ok = (b >= 0) & (b < nbins)
        if np.any(ok):
            jj = np.array(valid_js)[ok]
            bb = b[ok]
            pair_i.append(np.full_like(jj, i))
            pair_j.append(jj)
            pair_bins.append(bb)

    if len(pair_bins) == 0:
        # No pairs formed within radius -> return NaNs
        est = np.full(nbins, np.nan)
        jk = np.full(nbins, np.nan)
        return bin_centers, est, jk

    pair_i = np.concatenate(pair_i)
    pair_j = np.concatenate(pair_j)
    pair_bins = np.concatenate(pair_bins)

    # ---- 4) Helper to accumulate per-bin sums and compute estimator ----
    def accumulate_and_estimate(mask_pairs):
        # mask_pairs: boolean array over pairs to include
        if not np.any(mask_pairs):
            return np.full(nbins, np.nan)

        ii = pair_i[mask_pairs]
        jj = pair_j[mask_pairs]
        bb = pair_bins[mask_pairs]

        # Pair weights (area product)
        w_pair = area[ii] * area[jj]

        # w_sg for each pair
        wsg_pair = 0.5 * (delta_s[ii] * delta_i[jj] + delta_s[jj] * delta_i[ii])
        #wsg_pair = (delta_s[ii] * delta_i[jj] - delta_s[ii] * delta_r[jj] - delta_i[jj] * delta_r[ii] + delta_r[ii] * delta_r[jj]) / (delta_r[ii] * delta_r[jj] + eps) 
        # w_ss for each pair
        wss_pair = delta_s[ii] * delta_s[jj]
        #wss_pair = (delta_s[ii] * delta_s[jj] - delta_s[ii] * delta_r[jj] - delta_s[jj] * delta_r[ii] + delta_r[ii] * delta_r[jj])/ (delta_r[ii] * delta_r[jj] + eps)

        # Weighted sums per bin
        sum_w = np.bincount(bb, minlength=nbins)
        sum_wsg = np.bincount(bb, weights=wsg_pair, minlength=nbins)
        sum_wss = np.bincount(bb, weights=wss_pair, minlength=nbins)

        # Averages
        with np.errstate(invalid='ignore', divide='ignore'):
            avg_wsg = sum_wsg / (sum_w)
            avg_wss = sum_wss / (sum_w)

        # Estimator per bin: (<w_sg>^2)/<w_ss>
        #est = (avg_wsg ** 2) / (avg_wss)
        est = avg_wsg

        # (任意) 物理的に不安定な領域の警告は呼び出し側で行うとよい
        return est

    # ---- 5) Full-sample estimate ----
    full_mask = np.ones_like(pair_bins, dtype=bool)
    est_full = accumulate_and_estimate(full_mask)

    # ---- 6) Jackknife (leave-one-region-out) ----
    unique_jk = np.unique(jk_label)
    jk_vals = []

    # どのペアがどのJKラベルに属するか（i, j のどちらかが除外ラベルに含まれるときに除外）
    # 事前に各ラベルごとに「除外すべき pair マスク」を作ると速い
    idx2jk = jk_label  # alias
    for lab in unique_jk:
        # keep pairs where BOTH endpoints are NOT the excluded label
        keep = (idx2jk[pair_i] != lab) & (idx2jk[pair_j] != lab)
        est_j = accumulate_and_estimate(keep)
        jk_vals.append(est_j)

    jk_vals = np.vstack(jk_vals)  # shape: (n_jack, nbins)

    # Jackknife standard deviation
    m = jk_vals.shape[0]
    # jackknife mean of estimates
    jk_mean = np.nanmean(jk_vals, axis=0)
    jk_std = np.sqrt((m - 1) * np.nanmean((jk_vals - jk_mean[None, :]) ** 2, axis=0))

    return bin_centers, est_full, jk_std

########################################################################
def jackknife_dens(df, key, target_col, mean_density, n_jack=20, bins=10):
    df = df.copy()
    df['jackknife_block'] = np.random.randint(0, n_jack, size=len(df))

    # bin を最初に固定する
    _, fixed_bins = np.histogram(df[key], bins=bins)
    bin_centers = (fixed_bins[:-1] + fixed_bins[1:]) / 2.0

    means_all = []
    
    df[key + '_bin'] = pd.cut(df[key], bins=fixed_bins)


    summary = df.groupby(key + '_bin').apply(lambda x: pd.Series({
        'count': len(x[target_col]),
        'mean': np.sum(x[target_col] * x['area']) / np.sum(x['area']) / mean_density - 1.0,
        
        })).reset_index()

    for i in range(n_jack):
        jack_df = df[df['jackknife_block'] != i].copy()
        jack_df[key + '_bin'] = pd.cut(jack_df[key], bins=fixed_bins)

        def stats(x):
            area_sum = np.sum(x['area'])
            if area_sum == 0:
                return pd.Series({'mean': np.nan})
            mu = np.sum(x[target_col] * x['area']) / area_sum
            return pd.Series({'mean': mu / mean_density - 1.0})

        _summary = jack_df.groupby(key + '_bin').apply(stats).reset_index()
        means_all.append(_summary['mean'].values)

    means_all = np.array(means_all)
    mean_estimate = np.nanmean(means_all, axis=0)
    jk_std = np.sqrt((n_jack - 1) * np.nanmean((means_all - np.array(summary['mean'])[None, :])**2, axis=0))

    return bin_centers[summary['count']>50], summary['mean'][summary['count']>50], jk_std[summary['count']>50]

#################################################################################################################
def jackknife_PS(df, key, mean_density, target_col, n_jack=20):
    nside = 256
    area = hp.nside2pixarea(nside,degrees=True)
    
    full_healpix = np.arange(hp.nside2npix(nside))
    full = pd.DataFrame({'healpix':full_healpix})
    
    s_mean = np.sum(df[key]*df['area'])/np.sum(df['area'])
    delta_s = df[key]/s_mean - 1.0
    
    delta_i = df[target_col]/mean_density - 1.0
    
    data = pd.DataFrame({
        'healpix':df['healpix'],
        'delta_s':delta_s,
        'delta_i':delta_i,
        'jackknife':df['jackknife']
    })
    
    full_data = pd.merge(full, data, on='healpix', how='left')
    
    map_s = hp.ma(full_data['delta_s'])
    map_s.mask = np.isnan(full_data['delta_s'])
    
    map_i = hp.ma(full_data['delta_i'])
    map_i.mask = np.isnan(full_data['delta_i'])
    
    cl_is = hp.anafast(map_s, map_i, lmax = 100)
    cl_ss = hp.anafast(map_s, map_s, lmax = 100)
    
    x = np.arange(101)
    y = cl_is**2/cl_ss
    
    #data = devide_autumn(data)
    
    jacks = []
    
    for i in range(n_jack):
        full_data = pd.merge(full, data[data['jackknife']!=i], on='healpix', how='left')
    
        map_s = hp.ma(full_data['delta_s'])
        map_s.mask = np.isnan(full_data['delta_s'])
    
        map_i = hp.ma(full_data['delta_i'])
        map_i.mask = np.isnan(full_data['delta_i'])
    
        cl_is = hp.anafast(map_s, map_i, lmax = 100)
        cl_ss = hp.anafast(map_s, map_s, lmax = 100)
    
        _y = cl_is**2/cl_ss
        jacks.append(_y)
        
    jacks = np.array(jacks)
    jk_std = np.sqrt((n_jack - 1) * np.nanmean((jacks - y[None, :])**2, axis=0))
    
    
    
    return x, y, jk_std

