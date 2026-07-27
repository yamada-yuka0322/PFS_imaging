import numpy as np
import healpy as hp
from matplotlib.path import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

import pandas as pd

from pfsimaging import Loader as loader

from astropy.table import Table

import pymaster as nmt

nside = 256
area = hp.nside2pixarea(nside,degrees=True)

def mask_edge(autumn, spring):
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
    #polygon = np.array([
        #[127, -3.0],
        #[127, 6.0],
        #[227, 6.0],
        #[227, -3.0],
        #[127, -3.0]
        #])

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

def cut_edge(targets):
    # Healpix pixel centers
    ra = np.array(targets['RA'])
    dec = np.array(targets['DEC'])

    # 四角形のポリゴン（axs[0]の赤線内）
    polygon = np.array([
        [129, -1.0],
        [129, 4.2],
        [224.5, 4.2],
        [224.5, -1.0],
        [129, -1.0]
    ])
    #polygon = np.array([
        #[127, -3.0],
        #[127, 6.0],
        #[227, 6.0],
        #[227, -3.0],
        #[127, -3.0]
        #])

    # 点群がポリゴン内にあるかどうかを判定
    spring_path = Path(polygon)
    points = np.vstack([ra, dec]).T
    spring_mask = spring_path.contains_points(points)

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

    return targets[spring_mask | autumn_mask]

def mean_density(table):
    """
    function to calculate the area weighted mean density
    
    input
    ------------------------------------------------------
    table: astropy table
    must included the target density 'target' and the effective area 'area' for each healpixel.
    
    output
    -----------------------------
    mu: float
    effective area weighted mean density
    """
    m = ~np.isnan(table['target']) & np.isfinite(table['target']) & ~np.isnan(table['area']) & np.isfinite(table['area'])
    if not np.any(m):
        return 0
    area_sum = np.sum(table['area'][m])
    if area_sum == 0:
        return 0
    mu = np.sum(table['target'][m] * table['area'][m]) / area_sum
    return mu
    

def jackknife(table, figure=False):
    """
    function to assign jacknife index on each healpixels
    
    input
    ------------------------------------------------------
    table: 
    """
    table['jackknife'] = -1*np.ones(len(table))
    ra, dec = hp.pix2ang(nside=256, ipix = table['healpix'], lonlat=True)
    ra = ra - 360.0*(ra > 300.0) #ra range(-60 ~ 300)
    
    isAutumn = (ra>-35.0) & (ra < 45.0)
    isSpring = (ra > 120.0) & (ra < 230.0) & (dec > -4.0) & (dec < 6.0)
    
    array = -1*np.ones(len(ra[isAutumn]))

    for i in range(8):
        array[(ra[isAutumn]>=np.min(ra[isAutumn])+i*8.5)&(ra[isAutumn]<np.min(ra[isAutumn])+(i+1)*8.5)] = i
    array[(ra[isAutumn]>=np.min(ra[isAutumn])+i*8.5)&(dec[isAutumn]>=-1)] = 7
    array[(ra[isAutumn]>25)&(dec[isAutumn]<-1)] = 8
    table['jackknife'][isAutumn] = array

    #spring_df = spring.dropna(subset=['target'])
    
    array = np.ones(len(ra[isSpring]))
    for i in range(11):
        array[(ra[isSpring]>=np.min(ra[isSpring])+i*9)&(ra[isSpring]<np.min(ra[isSpring])+(i+1)*9)] = i+9
    
    table['jackknife'][isSpring] = array
    
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
        #custom_cmap = ListedColormap(colors_hex)

        #fig, axs = plt.subplots(
            #2, 1, figsize=(12, 8),
            #gridspec_kw={"height_ratios": [1, 1], "hspace": 0.2}
            #)

        #coll1 = plot_map(spring_df, 'jackknife', 'spring', axs[0], vmin=0, vmax=20, cmap = custom_cmap)
        #coll2 = plot_map(autumn_df, 'jackknife', 'autumn', axs[1], vmin=0, vmax=20, cmap = custom_cmap)

        #axs[1].set_xlabel('RA [deg]', fontsize=20)
    
        #axs[0].grid()
        #axs[0].yaxis.label.set_position((axs[0].yaxis.label.get_position()[0], 0.35))
        #axs[1].grid()
        #path = '/home/YukaYamada/output/PFS/figure/'
        #plt.subplots_adjust(top=0.7, bottom=0.3, left=0.12, right=0.95)
        #plt.savefig(path+'jackknife.pdf', bbox_inches='tight', dpi=300)
        #plt.savefig(path+'jackknife.png', bbox_inches='tight', dpi=300)
        #plt.show()
    
    return table
    
    
###############################################################################################################

def jackknife_ang_ratio(
    table, key, target_col, bin_edges_deg, 
    nside=256, eps=1e-15, ratio = True
):
    """
    Estimator: ( <w_sg>^2 ) / <w_ss>,  where
      w_sg(i,j) = 0.5 * ( delta_s_i * delta_i_j + delta_s_j * delta_i_i )
      w_ss(i,j) = delta_s_i * delta_s_j
    Averages <...> are pairwise, area-pair weighted: w_pair = area_i * area_j

    Parameters
    ----------
    table : table
        Columns must include ['healpix','area', target_col, key].
    key : str
        Name of the imaging systematics column.
    target_col: string
        name of the target density column 
        (either 'target' for the observed target or 'weighted_target' for the weighted target density)
    bin_edges_deg : 1D array-like
        Angular bin edges in degrees (monotonic increasing).
    nside : int
        HEALPix nside (default 256).
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

    _mean_density = mean_density(table)
    table = jackknife(table) #apply jacknife ID

    # ---- 1) Prepare deltas ----
    # delta_i: target density contrast (w.r.t provided mean_density)
    delta_i = np.array(table[target_col]) / float(_mean_density) - 1.0

    # delta_s: systematics contrast relative to area-weighted global mean
    glob_mean_key = np.average(np.array(table[key]), weights=np.array(table['area']))
    delta_s = np.array(table[key]) / glob_mean_key - 1.0
    
    #random
    delta_r = np.random.normal(loc = np.mean(delta_i), scale = np.std(delta_i), size = len(table['healpix']))

    hpix = np.array(table['healpix'])
    area = np.array(table['area'])
    jk_label = np.array(table['jackknife'])

    # ---- 2) Geometry ----
    # Get angular vectors of pixel centers
    theta, phi = hp.pix2ang(nside, hpix, nest=False)
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
        neigh = hp.query_disc(nside, vecs[i], radius_rad, inclusive=False, nest=False)
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
        # w_ss for each pair
        wss_pair = delta_s[ii] * delta_s[jj]
        wgg_pair = delta_i[ii] * delta_i[jj]

        # Weighted sums per bin
        sum_w = np.bincount(bb, minlength=nbins)
        sum_wsg = np.bincount(bb, weights=wsg_pair, minlength=nbins)
        sum_wgg = np.bincount(bb, weights=wgg_pair, minlength=nbins)
        sum_wss = np.bincount(bb, weights=wss_pair, minlength=nbins)

        # Averages
        with np.errstate(invalid='ignore', divide='ignore'):
            avg_wsg = sum_wsg / (sum_w)
            avg_wss = sum_wss / (sum_w)
            avg_wgg = sum_wgg / (sum_w)

        # Estimator per bin: (<w_sg>^2)/<w_ss>
        if ratio:
            est = (avg_wsg ** 2) / (avg_wss)
            
        else:
            est = avg_wsg

        # (任意) 物理的に不安定な領域の警告は呼び出し側で行うとよい
        return est, avg_wgg

    # ---- 5) Full-sample estimate ----
    full_mask = np.ones_like(pair_bins, dtype=bool)
    est_full, _ = accumulate_and_estimate(full_mask)

    # ---- 6) Jackknife (leave-one-region-out) ----
    unique_jk = np.unique(jk_label)
    jk_vals = []

    # どのペアがどのJKラベルに属するか（i, j のどちらかが除外ラベルに含まれるときに除外）
    # 事前に各ラベルごとに「除外すべき pair マスク」を作ると速い
    idx2jk = jk_label  # alias
    for lab in unique_jk:
        # keep pairs where BOTH endpoints are NOT the excluded label
        keep = (idx2jk[pair_i] != lab) & (idx2jk[pair_j] != lab)
        _, est_j = accumulate_and_estimate(keep) ##stacking "w_gg" to calculate sigma(w_gg)
        jk_vals.append(est_j)

    jk_vals = np.vstack(jk_vals)  # shape: (n_jack, nbins)

    # Jackknife standard deviation
    m = jk_vals.shape[0]
    # jackknife mean of estimates
    jk_mean = np.nanmean(jk_vals, axis=0)
    jk_std = np.sqrt((m - 1) * np.nanmean((jk_vals - jk_mean[None, :]) ** 2, axis=0))

    return bin_centers, est_full, jk_std

########################################################################
def jackknife_dens(
        table,
        key,
        target_col='target',
        n_jack=20,
        bins=10,
        min_count=100,
):
    """
    calculate target density as a function of imaging attribute. (with jackknife error bar)

    Parameters
    ----------
    table : astropy.table.Table
        table including Imaging attribute、target density、effective area。

    key : str
        imaging attribute name。

    target_col : str, default='target'
        Target density column name。
        ex: 'target', 'weighted_target'

    n_jack : int, default=20
        Jackknife region count。

    bins : int or array-like, default=10
        Bin count or bin edge。

    min_count : int, default=100
        minimum datapoint within single bin。

    Returns
    -------
    bin_centers : numpy.ndarray
        center of each bin

    mean : numpy.ndarray
        target density fluctuation
        n / mean_density - 1。

    std : numpy.ndarray
        jackkinfe error
    """
    #rng = np.random.default_rng()
    #table['jackknife'] = rng.integers(n_jack, size=len(table))
    table = jackknife(table)

    required_cols = {key, target_col, 'area', 'jackknife'}
    missing_cols = required_cols - set(table.colnames)

    if missing_cols:
        raise KeyError(
            f"Required columns are missing: {sorted(missing_cols)}"
        )

    mean_dens = mean_density(table)

    key_values = np.asarray(table[key], dtype=float)
    target_values = np.asarray(table[target_col], dtype=float)
    area_values = np.asarray(table['area'], dtype=float)
    jack_ids = np.asarray(table['jackknife'])


    valid = (
        np.isfinite(key_values)
        & np.isfinite(target_values)
        & np.isfinite(area_values)
        & (area_values > 0)
    )

    if not np.any(valid):
        raise ValueError("No valid data points were found.")

    # Bin edge
    if np.isscalar(bins):
        _, fixed_bins = np.histogram(
            key_values[valid],
            bins=int(bins),
        )
    else:
        fixed_bins = np.asarray(bins, dtype=float)

    n_bins = len(fixed_bins) - 1

    if n_bins < 1:
        raise ValueError("At least one bin is required.")

    bin_centers = 0.5 * (fixed_bins[:-1] + fixed_bins[1:])

    bin_ids = np.digitize(
        key_values,
        fixed_bins,
        right=False,
    ) - 1

    bin_ids[key_values == fixed_bins[-1]] = n_bins - 1

    valid &= (bin_ids >= 0) & (bin_ids < n_bins)

    def calculate_binned_mean(selection):
        """
        calculate mean density for each attribute bin
        """
        counts = np.zeros(n_bins, dtype=int)
        means = np.full(n_bins, np.nan, dtype=float)

        for bin_index in range(n_bins):
            m = (
                selection
                & valid
                & (bin_ids == bin_index)
            )

            counts[bin_index] = np.count_nonzero(m)

            if counts[bin_index] == 0:
                continue

            area_sum = np.sum(area_values[m])

            if not np.isfinite(area_sum) or area_sum <= 0:
                continue

            weighted_density = (
                np.sum(target_values[m] * area_values[m])
                / area_sum
            )

            means[bin_index] = weighted_density / mean_dens - 1.0

        return counts, means

    # full mean
    full_selection = np.ones(len(table), dtype=bool)
    counts, means = calculate_binned_mean(full_selection)
    
    # Leave-one-region-out jackknife
    jackknife_means = np.full(
        (n_jack, n_bins),
        np.nan,
        dtype=float,
    )
    
    for i in range(n_jack):
        selection = jack_ids != i
        _, jackknife_means[i] = calculate_binned_mean(selection)

    theta_bar = np.nanmean(jackknife_means, axis=0)

    squared_difference = (
        jackknife_means - theta_bar[None, :]
    ) ** 2

    # jackknife error
    jk_var = (
        (n_jack - 1) / n_jack
        * np.nansum(squared_difference, axis=0)
    )

    std = np.sqrt(jk_var)

    # remove bin with datacount < min_count
    output_mask = (
        (counts >= min_count)
        & np.isfinite(means)
        & np.isfinite(std)
    )

    return (
        bin_centers[output_mask],
        means[output_mask],
        std[output_mask],
    )


def density_poisson_error(
    table,
    key,
    target_col='target',
    bins=10,
    min_count=100,
    use_binomial=True,
):
    """
    Imaging attribute binごとのtarget density fluctuationと
    counting-statistics errorを計算する。

    Parameters
    ----------
    table : astropy.table.Table
        must include
        key, target_col, area, total

        target_col : target density
        area       : effective area
        total      : random count before applying mask

    key : str
        Imaging attribute name

    target_col : str, default='target'
        Target density column name

    bins : int or array-like, default=10
        bin count

    min_count : int, default=100
        required minimum number of pixels included in a single bin 

    use_binomial : bool, default=True
        True:
            treat N_ran | N_ran_tot as binominal distribution
        False:
            treat N_ran and N_ran_totas independent poisson distribution

    Returns
    -------
    bin_centers : numpy.ndarray
        center of each bin

    mean : numpy.ndarray
        target density fluctuation counted for each bin
        n_bin / n_global - 1

    error : numpy.ndarray
        counting-statistics error。
    """
    required = {key, target_col, 'area', 'total'}
    missing = required - set(table.colnames)

    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    imaging = np.asarray(table[key], dtype=float)
    density_pixel = np.asarray(table[target_col], dtype=float)
    effective_area_pixel = np.asarray(table['area'], dtype=float)
    random_total = np.asarray(table['total'], dtype=float)

    valid = (
        np.isfinite(imaging)
        & np.isfinite(density_pixel)
        & np.isfinite(effective_area_pixel)
        & np.isfinite(random_total)
        & (effective_area_pixel >= 0)
        & (random_total > 0)
        & (effective_area_pixel <= area)
    )

    if not np.any(valid):
        return (
            np.array([]),
            np.array([]),
            np.array([]),
        )

    # density × effective area = target count
    target_count = density_pixel * effective_area_pixel

    # unmasked fraction
    unmasked_fraction = (
        effective_area_pixel / area
    )

    # N_ran = f_unmasked × N_ran,total
    random_unmasked = (
        unmasked_fraction * random_total
    )

    valid &= (
        np.isfinite(target_count)
        & np.isfinite(random_unmasked)
        & (target_count >= 0)
        & (random_unmasked >= 0)
        & (random_unmasked <= random_total)
    )

    # Global mean density
    global_target_count = np.sum(target_count[valid])
    global_effective_area = np.sum(effective_area_pixel[valid])

    if global_effective_area <= 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
        )

    global_mean_density = (
        global_target_count / global_effective_area
    )

    # Bin edge
    if np.isscalar(bins):
        fixed_bins = np.histogram_bin_edges(
            imaging[valid],
            bins=int(bins),
        )
    else:
        fixed_bins = np.asarray(bins, dtype=float)

    n_bins = len(fixed_bins) - 1
    bin_centers = 0.5 * (
        fixed_bins[:-1] + fixed_bins[1:]
    )

    bin_id = np.digitize(
        imaging,
        fixed_bins,
        right=False,
    ) - 1

    bin_id[imaging == fixed_bins[-1]] = n_bins - 1

    mean = np.full(n_bins, np.nan)
    error = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        m = (
            valid
            & (bin_id == i)
        )

        counts[i] = np.count_nonzero(m)

        if counts[i] == 0:
            continue

        total_target = np.sum(target_count[m])
        total_effective_area = np.sum(
            effective_area_pixel[m]
        )

        if total_effective_area <= 0:
            continue

        density = total_target / total_effective_area

        # n_bin / n_global - 1
        mean[i] = (
            density / global_mean_density - 1.0
        )

        # Actual unweighted countにだけ適用可能
        target_variance = total_target

        if use_binomial:
            # Var(a_eff,j)
            area_variance_pixel = (
                area ** 2
                * unmasked_fraction[m]
                * (1.0 - unmasked_fraction[m])
                / random_total[m]
            )
        else:
            nran = random_unmasked[m]
            nran_tot = random_total[m]

            area_variance_pixel = (
                area**2
                * (
                    nran / nran_tot**2
                    + nran**2 / nran_tot**3
                )
            )

        effective_area_variance = np.sum(
            area_variance_pixel
        )

        density_variance = (
            target_variance
            / total_effective_area**2
            +
            total_target**2
            * effective_area_variance
            / total_effective_area**4
        )

        density_error = np.sqrt(density_variance)

        # fluctuation n/n_global - 1 の誤差
        # global mean density自体の誤差はここでは無視
        error[i] = (
            density_error / global_mean_density
        )

    output = (
        (counts >= min_count)
        & np.isfinite(mean)
        & np.isfinite(error)
    )

    return (
        bin_centers[output],
        mean[output],
        error[output],
    )

#################################################################################################################
def jackknife_PS(table, key, target_col, n_jack=20):
    nside = 256
    area = hp.nside2pixarea(nside,degrees=True)
    
    _mean_density = mean_density(table)
    table = jackknife(table) #apply jacknife ID
    
    s_mean = np.sum(table[key]*table['area'])/np.sum(table['area'])
    delta_s = table[key]/s_mean - 1.0
    
    delta_i = table[target_col]/_mean_density - 1.0
    
    _table = Table({
        'healpix': table['healpix'],
        "delta_i": delta_i,
        "delta_s": delta_s
    })
    
    lmin = 10
    lmax = 3 * nside

    edges = np.unique(
        np.logspace(
            np.log10(lmin),
            np.log10(lmax),
            11,
        ).astype(int)
    )

    b = nmt.NmtBin.from_edges(
        edges[:-1],
        edges[1:],
    )

    x = b.get_effective_ells()
    
    y = calculate_PS(
        table,
        key,
        target_col,
        b,
        nside,
    )

    jacks = []

    for i in range(n_jack):
        jack_table = table[table["jackknife"] != i]

        y_jack = calculate_PS(
            jack_table,
            key,
            target_col,
            b,
            nside,
        )

        jacks.append(y_jack)

    jacks = np.asarray(jacks)

    jack_mean = np.nanmean(jacks, axis=0)

    jk_std = np.sqrt(
        (n_jack - 1)
        * np.nanmean(
            (jacks - jack_mean[None, :])**2,
            axis=0,
        )
    )
    return x, y, jk_std

def calculate_PS(table, key, target_col, bins, nside):
    npix = hp.nside2npix(nside)

    healpix = np.asarray(table["healpix"], dtype=int)
    area = np.asarray(table["area"], dtype=float)
    target = np.asarray(table[target_col], dtype=float)
    systematics = np.asarray(table[key], dtype=float)

    valid = (
        np.isfinite(area)
        & np.isfinite(target)
        & np.isfinite(systematics)
        & (area > 0)
    )

    healpix = healpix[valid]
    area = area[valid]
    target = target[valid]
    systematics = systematics[valid]

    # target_colが既に面密度の場合
    mean_target = np.sum(target * area) / np.sum(area)

    # imaging propertyの面積加重平均
    mean_systematics = np.sum(systematics * area) / np.sum(area)

    delta_i = target / mean_target - 1.0
    delta_s = systematics / mean_systematics - 1.0

    mask = np.zeros(npix, dtype=float)
    mask[healpix] = 1.0

    map_i = np.zeros(npix, dtype=float)
    map_s = np.zeros(npix, dtype=float)

    map_i[healpix] = delta_i
    map_s[healpix] = delta_s

    field_i = nmt.NmtField(mask, [map_i])
    field_s = nmt.NmtField(mask, [map_s])

    cl_is = nmt.compute_full_master(
        field_i,
        field_s,
        bins,
    )[0]

    cl_ss = nmt.compute_full_master(
        field_s,
        field_s,
        bins,
    )[0]

    result = np.full_like(cl_is, np.nan)

    good = (
        np.isfinite(cl_is)
        & np.isfinite(cl_ss)
        & (cl_ss > 0)
    )

    result[good] = cl_is[good] ** 2 / cl_ss[good]

    return result