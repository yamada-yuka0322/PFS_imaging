from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors

import numpy as np
import healpy as hp

import sys
sys.path.append("../src")

from bias import *
from Loader import *
from Regression import *

from sklearn.preprocessing import StandardScaler
import pandas as pd

import torch

from statistic import *

names = {
    'gseeing':"$\it{g}$-seeing",
    'rseeing':"$\it{r}$-seeing",
    'iseeing':"$\it{i}$-seeing",
    'zseeing':"$\it{z}$-seeing",
    'yseeing':"$\it{y}$-seeing",
    
    'g_depth':"$\it{g}$-depth",
    'r_depth':"$\it{r}$-depth",
    'i_depth':"$\it{i}$-depth",
    'z_depth':"$\it{z}$-depth",
    'y_depth':"$\it{y}$-depth",
    
    'star':"star",
    'star_log':'log(stellar density)',
    'extinction':"extinction",
    'desi_extinction':"extinction"
}

def plot_map(df, target, field, vmin=None, vmax=None, cmap='viridis', colorbar=True):
    """
    input
    ------------------------------------------------------
    df: pandas dataframe
    including "target" and "healpix" column
    
    target: string
    column name of target data
    
    field: string (spring/ field)
    
    ax: matplotlib ax
    to plot
    """
    
    fig, ax = plt.subplots(figsize=(12,4))
    
    if field =='spring':
        ramin = 125
        ramax = 230
        ax.set_ylim(-4, 7)
        
        yticks = np.array([0, 5])
        xticks = np.array([140, 160, 180, 200, 220])
    if field=='autumn':
        ramin=-35
        ramax=45
        ax.set_ylim(-8, 8)
        yticks = np.array([-5, 0, 5])
        xticks = np.array([-20, 0, 20, 40])
        
    nside = 256
    npix = hp.nside2npix(nside)

    valid_pix = df['healpix']
    values = df[target]

    verts = []
    colors = []

    for pix, val in zip(valid_pix, values):
        boundary = hp.boundaries(nside, pix, step=1, nest=False)  # shape=(2, N)
        vec = hp.ang2vec(*hp.pix2ang(nside, pix))  # enter
        theta, phi = hp.vec2ang(boundary.T)
        ra, dec = hp.pix2ang(nside, pix, lonlat=True)  #center coordinate
        ra_boundary, dec_boundary = hp.vec2ang(boundary.T, lonlat=True)

        # ra -180~180
        ra_boundary = np.degrees(phi)
        if field=='autumn':
            ra_boundary = np.where(ra_boundary > 180, ra_boundary - 360, ra_boundary)
        dec_boundary = 90 - np.degrees(theta)

        verts.append(list(zip(ra_boundary, dec_boundary)))
        colors.append(val)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    coll = PolyCollection(
        verts, array=np.array(colors), cmap=cmap,
        edgecolors='none', linewidth=0, norm=norm  # ← norm
    )
    ax.add_collection(coll)

    ax.set_xlim(ramin, ramax)  # RA range
    ax.set_ylabel('Dec [deg]', fontsize=20)
    ax.set_aspect('equal')

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.tick_params(labelsize=20)
    # --- Colorbar---
    if colorbar:
        cbar = fig.colorbar(coll, ax=ax, shrink=0.5, orientation='vertical')
        cbar.set_label(target, fontsize=18)
        cbar.ax.tick_params(labelsize=14)
    
    return coll
    
###############################################################################################################
def Plot_Ang(autumn, spring, targets, weights = [], field='all'):
    data = prepare_data(autumn, spring, cut_edge=True, do_jackknife=True, target_list = weights)
    if (field=='autumn'):
        data = data[data['jackknife'] <= 8]
    if (field=='spring'):
        data = data[data['jackknife'] > 8]
        
    _data = data.copy()
    for name, target in targets.items():
        _data = get_target_density(target, _data, name = name)
        
    keys = ["gseeing", "rseeing", "iseeing", "zseeing", "yseeing", "g_depth", "r_depth", "i_depth", "z_depth", "y_depth", "desi_extinction", "star_log"]
    fig, axes = plt.subplots(2, 6, figsize=(30, 10))
    
    cmap = plt.get_cmap("tab10")
    bins = np.linspace(0, 5, 11)
    
    for i, key in enumerate(keys):
        ax = axes[i // 6, i % 6]
        
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(['0.0', '1.0', '2.0', '3.0', '4.0'], fontsize=20)
        ax.set_xlabel('[deg]', fontsize=20)

        ax.yaxis.set_tick_params(labelsize=20)
    
        if 'seeing' in key:
            xlabel = 'arcsec'
        if 'depth' in key:
            xlabel = 'mag'
        if 'extinction' in key:
            xlabel = 'mag'
        if 'star' in key:
            _data[key] = np.log10(_data['star'])
            xlabel = 'log deg$^{-2}$'

        for j, name in enumerate(targets.keys()):
            mean_density =  np.sum(_data[name]*_data['area'])/np.sum(_data['area'])
            x, mean, std = jackknife_ang_ratio(_data, key, mean_density, name, bins)
            mean *= 1e3
            std *= 1e3
            ax.fill_between(x, mean - std, mean + std, color=cmap(j), alpha=0.3)
            ax.plot(x, mean, color=cmap(j), label=name)
            
        _x = np.linspace(0.0, 5.0, 25)
        _y = np.zeros(25)
        ax.plot(_x, _y, ls='--', color="cyan")
        
        ax.text(0.95, 0.15, names[key], ha='right', va='top', transform=ax.transAxes, fontsize=20)
    
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        
        if i%6==0:
            #ax.set_ylabel(r'$(\omega^{g,s})^2/\omega^{s,s}$', fontsize=20)
            ax.set_ylabel(r'$\omega^{g,s} \times 10^3$', fontsize=20)
        #else:
            #ax.set_yticklabels([])
        if i < 6:
            ax.set_xticklabels([])
            ax.set_xlabel("")
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=20)
    plt.tight_layout()
    #path = '/home/YukaYamada/output/PFS/figure/'
    #plt.subplots_adjust(top=0.7, bottom=0.3, left=0.12, right=0.95)
    #plt.savefig(path+'ang_corr.pdf', bbox_inches='tight', dpi=300)
    #plt.savefig(path+'ang_corr.png', bbox_inches='tight', dpi=300)
    #plt.show()
    
    return
    
def plot_ang(autumn, spring, target_list = [], field="all"):
    """
    input
    -------------------------------------------------------
    df: pandas dataframe
    must include ["healpix","jackknife", "target", "area", keys]
    """
    
    data = prepare_data(autumn, spring, cut_edge=True, do_jackknife=True, target_list = target_list)
    mean_density = np.sum(data['target']*data['area'])/np.sum(data['area'])
    
    keys = ["gseeing", "rseeing", "iseeing", "zseeing", "yseeing", "g_depth", "r_depth", "i_depth", "z_depth", "y_depth", "desi_extinction", "star_log"]
    fig = plt.figure(figsize=(30, 10))
    bins = np.linspace(0, 5, 11)
    for i, key in enumerate(keys):
        ax = fig.add_subplot(2, 6, i+1 )
    
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(['0.0', '1.0', '2.0', '3.0', '4.0'], fontsize=20)
        ax.set_xlabel('[deg]', fontsize=20)

        ax.yaxis.set_tick_params(labelsize=20)
    
        x, mean, std = jackknife_ang_ratio(data, key, mean_density, 'target', bins)
        mean *= 1e3
        std *= 1e3
        plt.fill_between(x, mean - std, mean + std, color='black', alpha=0.3)
        plt.plot(x, mean, color='black', label='No Weight')
        
        if "NN" in target_list:
            x, mean, std = jackknife_ang_ratio(data, key, mean_density, 'target_nn', bins)
            mean *= 1e3
            std *= 1e3
            plt.fill_between(x, mean - std, mean + std, color='blue', alpha=0.3)
            plt.plot(x, mean, color='blue', label='NN')
            
        if "linear" in target_list:
            x, mean, std = jackknife_ang_ratio(data, key, mean_density, 'target_lin', bins)
            mean *= 1e3
            std *= 1e3
            plt.fill_between(x, mean - std, mean + std, color='red', alpha=0.3)
            plt.plot(x, mean, color='red', label='Linear')
            
        if "quad" in target_list:
            x, mean, std = jackknife_ang_ratio(data, key, mean_density, 'target_quad', bins)
            mean *= 1e3
            std *= 1e3
            plt.fill_between(x, mean - std, mean + std, color='green', alpha=0.3)
            plt.plot(x, mean, color='green', label='Quadratic')
            
        if "mock" in target_list:
            x, mean, std = jackknife_ang_ratio(data, key, mean_density, 'mock', bins)
            mean *= 1e3
            std *= 1e3
            plt.fill_between(x, mean - std, mean + std, color='gold', alpha=0.3)
            plt.plot(x, mean, color='gold', label='mock')
            
            x, mean, std = jackknife_ang_ratio(data, key, mean_density, 'weighted', bins)
            mean *= 1e3
            std *= 1e3
            plt.fill_between(x, mean - std, mean + std, color='purple', alpha=0.3)
            plt.plot(x, mean, color='purple', label='mock')
            
        _x = np.linspace(0.0, 5.0, 25)
        _y = np.zeros(25)
        plt.plot(_x, _y, ls='--', color="cyan")
        
        ax.text(0.95, 0.15, names[key], ha='right', va='top', transform=ax.transAxes, fontsize=20)
    
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        
        if i%6==0:
            #ax.set_ylabel(r'$(\omega^{g,s})^2/\omega^{s,s}$', fontsize=20)
            ax.set_ylabel(r'$\omega^{g,s} \times 10^3$', fontsize=20)
        #else:
            #ax.set_yticklabels([])
        if i < 6:
            ax.set_xticklabels([])
            ax.set_xlabel("")
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=20)
    plt.tight_layout()
    #path = '/home/YukaYamada/output/PFS/figure/'
    #plt.subplots_adjust(top=0.7, bottom=0.3, left=0.12, right=0.95)
    #plt.savefig(path+'ang_corr.pdf', bbox_inches='tight', dpi=300)
    #plt.savefig(path+'ang_corr.png', bbox_inches='tight', dpi=300)
    #plt.show()
    
    return

#####################################################################################################
def Plot_Dens(autumn, spring, targets, weights = [], field='all'):
    data = prepare_data(autumn, spring, cut_edge=True, do_jackknife=True, target_list = weights)
    if (field=='autumn'):
        data = data[data['jackknife'] <= 8]
    if (field=='spring'):
        data = data[data['jackknife'] > 8]
        
    _data = data.copy()
    for name, target in targets.items():
        _data = get_target_density(target, _data, name = name)
        
    keys = ["gseeing", "rseeing", "iseeing", "zseeing", "yseeing", "g_depth", "r_depth", "i_depth", "z_depth", "y_depth", "desi_extinction", "star_log"]
    fig, axes = plt.subplots(2, 6, figsize=(30, 10), sharey=True)
    
    cmap = plt.get_cmap("tab10")
    
    for i, key in enumerate(keys):
        ax = axes[i // 6, i % 6]
    
        if 'seeing' in key:
            xlabel = 'arcsec'
        if 'depth' in key:
            xlabel = 'mag'
        if 'extinction' in key:
            xlabel = 'mag'
        if 'star' in key:
            _data[key] = np.log10(_data['star'])
            xlabel = 'log deg$^{-2}$'

        for j, name in enumerate(targets.keys()):
            mean_density =  np.sum(_data[name]*_data['area'])/np.sum(_data['area'])
            bc, mean, std = jackknife_dens(_data, key, name, mean_density)
            ax.plot(bc, mean, color=cmap(j), label = name)
            ax.fill_between(bc, mean - std, mean + std, color=cmap(j), alpha=0.3)

        #ax.set_title(key, fontsize=20)
        ax.set_ylim(-0.2, 0.2)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
    
        ax.text(0.5, 0.85, names[key], ha='right', va='top', transform=ax.transAxes, fontsize=20)

    
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        
        if i%6==0:
            ax.set_ylabel(r'$\bar{d}(s_k)/\bar{d} - 1$', fontsize=20)
        
        ax.set_xlabel(xlabel, fontsize=20)
        
        _y = np.zeros(len(bc))
        ax.plot(bc, _y, ls='--', color="cyan")

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=20)
    plt.subplots_adjust(wspace=0.0, hspace=0.2, bottom=0.15) 
    plt.tight_layout()
    return
        
    
def plot_dens(autumn, spring, target_list = [], field="all"):
    
    data = prepare_data(autumn, spring, cut_edge=True, do_jackknife=True, target_list = target_list)
    mean_density = np.sum(data['target']*data['area'])/np.sum(data['area'])
    
    keys = ["gseeing", "rseeing", "iseeing", "zseeing", "yseeing", "g_depth", "r_depth", "i_depth", "z_depth", "y_depth", "desi_extinction", "star_log"]
    fig, axes = plt.subplots(2, 6, figsize=(30, 10), sharey=True)
    
    for i, key in enumerate(keys):
        ax = axes[i // 6, i % 6]
    
        if 'seeing' in key:
            xlabel = 'arcsec'
        if 'depth' in key:
            xlabel = 'mag'
        if 'extinction' in key:
            xlabel = 'mag'
        if 'star' in key:
            xlabel = 'deg$^{-2}$'

        # No correction
        bc, mean, std = jackknife_dens(data, key, 'target', mean_density, n_jack=20)
        ax.plot(bc, mean, color='black', label = 'No Correction')
        ax.fill_between(bc, mean - std, mean + std, color='black', alpha=0.3)
    
        if ('NN' in target_list):
            bc, mean, std = jackknife_dens(data, key, 'target_nn', mean_density, n_jack=20)
            ax.plot(bc, mean, color='blue', label = 'NN')
            ax.fill_between(bc, mean - std, mean + std, color='blue', alpha=0.3)
        
        if ('linear' in target_list):
            bc, mean, std = jackknife_dens(data, key, 'target_lin', mean_density, n_jack=20)
            ax.plot(bc, mean, color='red', label = 'Linear')
            ax.fill_between(bc, mean - std, mean + std, color='red', alpha=0.3)
    
        if ('quad' in target_list):
            bc, mean, std = jackknife_dens(data, key, 'target_quad', mean_density, n_jack=20)
            ax.plot(bc, mean, color='green', label = 'Quadratic')
            ax.fill_between(bc, mean - std, mean + std, color='green', alpha=0.3)

        #ax.set_title(key, fontsize=20)
        ax.set_ylim(-0.2, 0.2)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
    
        ax.text(0.5, 0.85, names[key], ha='right', va='top', transform=ax.transAxes, fontsize=20)

    
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        
        if i%6==0:
            ax.set_ylabel(r'$\bar{d}(s_k)/\bar{d} - 1$', fontsize=20)
        
        ax.set_xlabel(xlabel, fontsize=20)
        
        _y = np.zeros(len(bc))
        ax.plot(bc, _y, ls='--', color="cyan")

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=20)
    plt.subplots_adjust(wspace=0.0, hspace=0.2, bottom=0.15) 
    plt.tight_layout()
    return

#####################################################################################################
def Plot_PS(autumn, spring, targets, weights = [], field='all'):
    data = prepare_data(autumn, spring, cut_edge=True, do_jackknife=True, target_list = weights)
    if (field=='autumn'):
        data = data[data['jackknife'] <= 8]
    if (field=='spring'):
        data = data[data['jackknife'] > 8]
        
    _data = data.copy()
    for name, target in targets.items():
        _data = get_target_density(target, _data, name = name)
    
    keys = ["gseeing", "rseeing", "iseeing", "zseeing", "yseeing", "g_depth", "r_depth", "i_depth", "z_depth", "y_depth", "desi_extinction", "star_log"]
    fig, axes = plt.subplots(2, 6, figsize=(30, 10), sharey=True)
    cmap = plt.get_cmap("tab10")
        
    for i, key in enumerate(keys):
        ax = axes[i // 6, i % 6]  # 行列から1次元インデックスを取り出す
        ax.set_ylim(3e-12, 1e-4)
        #ax.set_title(key, fontsize=20)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        
        if 'star' in key:
            _data[key] = np.log10(_data['star'])
        
        for j, name in enumerate(targets.keys()):
            mean_density =  np.sum(_data[name]*_data['area'])/np.sum(_data['area'])
            bc, mean, std = jackknife_PS(_data, key, mean_density, name)
            ax.plot(bc, mean, color=cmap(j), label = name)
            ax.fill_between(bc, mean, mean + std, color=cmap(j), alpha=0.3)
        
        ax.set_xlabel('$l$', fontsize=20)
    
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        
        if i%6==0:
            ax.set_ylabel(r'$(C_l^{g,s})^2/C_l^{s,s}$', fontsize=20)
        if i < 6:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        ax.text(0.9, 0.9, names[key], ha='right', va='top', transform=ax.transAxes, fontsize=20)

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=20)
    plt.subplots_adjust(wspace=0.0, hspace=0.0, bottom=0.15) 
    plt.tight_layout()
    return


def plot_PS(autumn, spring, target_list = [], random=True, field="all"):
    data = prepare_data(autumn, spring, cut_edge=True, do_jackknife=True, target_list = target_list)
    mean_density = np.sum(data['target']*data['area'])/np.sum(data['area'])
    std = np.std(data['target'])
    
    keys = ["gseeing", "rseeing", "iseeing", "zseeing", "yseeing", "g_depth", "r_depth", "i_depth", "z_depth", "y_depth", "desi_extinction", "star_log"]
    fig, axes = plt.subplots(2, 6, figsize=(30, 10), sharey=True)
    if random:
        data['random'] = mean_density + np.random.normal(loc = 0, scale = std, size = len(data["target"]))
        random_mean = []
        random_std = []
        
    for i, key in enumerate(keys):
        ax = axes[i // 6, i % 6]  # 行列から1次元インデックスを取り出す
        ax.set_ylim(3e-12, 1e-4)
        #ax.set_title(key, fontsize=20)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
    
        bc, mean, std = jackknife_PS(data, key, mean_density, 'target', n_jack=20)
        ax.plot(bc, mean, color='black', label = 'No Correction')
        ax.fill_between(bc, mean, mean + std, color='black', alpha=0.3)
        
        if('NN' in target_list):
            bc, mean, std = jackknife_PS(data, key, mean_density, 'target_nn', n_jack=20)
            ax.plot(bc, mean, color='blue', label = 'NN')
            
        if('linear' in target_list):
            bc, mean, std = jackknife_PS(data, key, mean_density, 'target_lin', n_jack=20)
            ax.plot(bc, mean, color='red', label = 'Linear')
            
        if('quad' in target_list):
            bc, mean, std = jackknife_PS(data, key, mean_density, 'target_quad', n_jack=20)
            ax.plot(bc, mean, color='green', label = 'Quadtratic')
            
        if ('mock' in target_list):
            bc, mean, std = jackknife_PS(data, key, mean_density, 'mock', n_jack=20)
            ax.plot(bc, mean, color='gold', label = 'mock')
            ax.fill_between(bc, mean, mean + std, color='gold', alpha=0.3)
            
        if random:
            bc, mean, std = jackknife_PS(data, key, mean_density, 'random', n_jack=20)
            random_mean.append(mean)
            random_std.append(std)

        ax.set_xlabel('$l$', fontsize=20)
    
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        
        if i%6==0:
            ax.set_ylabel(r'$(C_l^{g,s})^2/C_l^{s,s}$', fontsize=20)
        if i < 6:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        ax.text(0.9, 0.9, names[key], ha='right', va='top', transform=ax.transAxes, fontsize=20)
    
    if random:
        random_mean = np.log10(random_mean)
        random_std = np.log10(random_std)
        mean = 10**np.mean(random_mean, axis = 0)
        std = 10**np.mean(random_std, axis = 0)
        for i, key in enumerate(keys):
            ax = axes[i // 6, i % 6]  # 行列から1次元インデックスを取り出す
            ax.plot(bc, mean, ls='--', color='cyan')
            ax.fill_between(bc, mean, mean + std, color='cyan', alpha=0.3)


    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=20)
    plt.subplots_adjust(wspace=0.0, hspace=0.0, bottom=0.15) 
    plt.tight_layout()
    return
    
#####################################################################################################

def prepare_data(autumn, spring, cut_edge, do_jackknife, target_list = [], field = 'all'):
    if cut_edge:
        _autumn, _spring = mask_edge(autumn, spring)
        autumn_df = _autumn.to_pandas()
        spring_df = _spring.to_pandas()
    else:
        autumn_df = autumn.to_pandas()
        spring_df = spring.to_pandas()
    
    if do_jackknife:
        autumn_df, spring_df = jackknife(autumn_df, spring_df)
    
    cleaned_df = pd.concat([autumn_df, spring_df], ignore_index=True)
    # ===== NN 重み =====
    keys = ["gseeing", "rseeing", "iseeing", "zseeing", "yseeing", "g_depth", "r_depth", "i_depth", "z_depth", "y_depth", "desi_extinction", "star_log"]
    
    out = pd.DataFrame({'healpix': cleaned_df['healpix'].to_numpy()})
    if ('NN' in target_list)or('mock' in target_list):
        cleaned_weight = nn_weights(cleaned_df, cleaned_df['healpix'], keys)
        w_nn = np.asarray(cleaned_weight['nn_weights'])
        out['nn'] = w_nn
        out['target_nn'] = cleaned_df['target'].to_numpy() * w_nn
        if 'mock' in target_list:
            mean_density = np.sum(cleaned_df['target']*cleaned_df['area'])/np.sum(cleaned_df['area'])
            std = np.std(cleaned_df["target"])
            mock = 1 / np.array(w_nn) * (mean_density + np.random.normal(loc = 0, scale = std, size = len(cleaned_weight["target"])))
            weighted = mock * np.array(w_nn)
            out['mock'] = mock
            out['weighted'] = weighted

    # ===== 線形 重み =====
    if 'linear' in target_list:
        cleaned_weight, lin_coef = linear_weights(cleaned_df, cleaned_df['healpix'], keys)
        w_lin = np.asarray(cleaned_weight['lin_weights'])
        out['lin'] = w_lin
        out['target_lin'] = cleaned_df['target'].to_numpy() * w_lin

    # ===== 二次(Quadratic) 重み =====
    if 'quad' in target_list:
        cleaned_weight_q = quadratic_weights(cleaned_df, cleaned_df['healpix'], keys)
        w_quad = np.asarray(cleaned_weight_q['quad_weights'])
        out['quad'] = w_quad
        out['target_quad'] = cleaned_df['target'].to_numpy() * w_quad

    # ===== 最後に一度だけ結合 =====
    merged = cleaned_df.merge(out, on='healpix', how='inner')
    
    if(field == 'spring'):
        pixel = spring_df['pixel']
        merged = merged[merged['healpix'].isin(pixel)]
    elif(field == 'autumn'):
        pixel = autumn_df['pixel']
        merged = merged[merged['healpix'].isin(pixel)]

    return merged
            
            
        
        
    
    