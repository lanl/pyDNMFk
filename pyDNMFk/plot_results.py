# @author: Manish Bhattarai
import matplotlib
from matplotlib import pyplot as plt
from .data_io import *


def plot_err(err):
    """Plots the relative error for NMF decomposition as a function of number of iterations"""
    idx = np.linspace(1, len(err), len(err))
    plt.plot(idx, err)
    plt.xlabel('Iterations')
    plt.ylabel('Relative error')
    plt.title('Relative error vs Iterations')
    plt.savefig('Error_plot.png')
    plt.show()


def read_plot_factors(factors_path, pgrid):
    """Reads the factors W and H and Plots them"""
    W, H = read_factors(factors_path, pgrid)
    plot_W(W)
    plt.savefig(factors_path + 'W.png')
    plot_W(H.T)
    plt.savefig(factors_path + 'H.png')


def plot_W(W):
    """Reads a factor and plots into subplots for each component"""
    m, k = W.shape

    params = {'legend.fontsize': 60,
              'axes.labelsize': 60,
              'axes.titlesize': 60,
              'xtick.labelsize': 60,
              'mathtext.fontset': 'cm',
              'mathtext.rm': 'serif',
              "xtick.bottom": False,
              "ytick.left": False,
              }
    matplotlib.rcParams.update(params)

    f, axes = plt.subplots(nrows=k, sharex=True, figsize=(60, 40))

    plt.subplots_adjust(hspace=0.001, bottom=0.2)

    # colors=["blue", "red"]
    colors = plt.rcParams["axes.prop_cycle"]()
    W = W.T
    for i in range(k):
        c = next(colors)["color"]
        axes[i].plot(W[i], label="W[{}]".format(i), color=c, linewidth=5.0)
        axes[i].legend(loc=4, prop={'size': 50})
        axes[i].tick_params(axis="y", labelsize=30)

    plt.xlabel('Features')

    # create subplot just for placing the ylabel centered on all plots
    shadowaxes = f.add_subplot(111, xticks=[], yticks=[], frame_on=False)
    shadowaxes.set_ylabel('W Components')
    shadowaxes.yaxis.set_label_coords(-0.05, 0.5)
    plt.savefig('Results_W.png', bbox_inches='tight')
    plt.show()


def plot_results(startProcess, endProcess, stepProcess,RECON, RECON1, SILL_MIN, out_put, name):
    """Plots the relative error and Silhouette results for estimation of k"""
    ######################################## Plotting ####################################################
    t = range(startProcess, endProcess + 1,stepProcess)
    fig, ax1 = plt.subplots(num=None, figsize=(10, 6), dpi=300, facecolor='w', edgecolor='k')
    title = 'Num'
    color = 'tab:red'
    ax1.set_xlabel('Total Signatures')
    ax1.set_ylabel('Mean L2 %', color=color)
    ax1.set_title(title)
    lns1 = ax1.plot(t, RECON, marker='o', linestyle=':', color=color, label='Mean L2 %')
    lns3 = ax1.plot(t, RECON1, marker='X', linestyle=':', color='tab:green', label="Relative error %")

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.xaxis.set_ticks(np.arange(min(t), max(t) + 1, 1))
    # ax1.axvspan(shadow_start, shadow_end, alpha=0.20, color='#ADD8E6')
    # ax1.axvspan(shadow_alternative_start,  shadow_alternative_end, alpha=0.20, color='#696969')
    # manipulate the y-axis values into percentage 
    vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    # ax1.legend(loc=0)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Minimum Stability', color=color)  # we already handled the x-label with ax1
    lns2 = ax2.plot(t, SILL_MIN, marker='s', linestyle="-.", color=color, label='Minimum Stability')
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.legend(loc=1)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()

    # added these three lines
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    plt.savefig(out_put + '/' + name + '_selection_plot.pdf')

    plt.close()


def box_plot(dat, respath):
    """Plots the boxplot from the given data and saves the results"""
    dat.plot.bar()
    plt.xlabel('operation')
    plt.ylabel('timing(sec)')
    plt.savefig(respath + 'timing.png')

    # plt.show()


def timing_stats(fpath):
    """Reads the timing stats dictionary from the stored file and parses the data. """
    import copy
    data = pd.read_csv(fpath).iloc[0, 1:]
    breakdown_level_2 = {'init': ['__init__', 'init_factors'],
                         'data_io': ['read', 'create_folder_dir', 'save_factors', 'save_cluster_results'],
                         'sample': ['randM'], 'dist_compute': ['compute_global_dim', \
                                                               'global_gram', 'AH_glob', 'ATW_glob',
                                                               'normalize_features', 'dist_norm', 'relative_err',
                                                               'sum_axis', 'UHT_glob', 'WTU_glob'],
                         'dist_comm': ['cart_2d_collect_factors', 'gather_W_H'], \
                         'clustering': ['normalize_by_W', 'greedy_lsa', 'change_order', 'dist_feature_ordering', 'mad',
                                        'dist_silhouettes', 'column_err', 'pvalueAnalysis']}
    breakdown_level_1 = {'init': 'init_factors', 'dist_io': ['read', 'save_factors', 'save_cluster_results'],
                         'sampling': 'randM',
                         'clustering': ['dist_custom_clustering', 'mad', 'dist_silhouettes', 'pvalueAnalysis'],
                         'compute': 'fit'}
    results = {}

    ''''Data parsing'''
    breakdown_level_1_dat = copy.deepcopy(breakdown_level_1)
    breakdown_level_2_dat = copy.deepcopy(breakdown_level_2)

    for key, val in data.to_dict().items():
        for keys, vals in breakdown_level_1.items():
            try:
                if type(vals) == str:  # Only one val
                    if vals == key:
                        breakdown_level_1_dat[keys] = val
                else:  # Multiple val
                    idx = [key == v for v in vals].index(1)
                    breakdown_level_1_dat[keys][idx] = val
            except:
                continue
        for keys, vals in breakdown_level_2.items():
            try:
                if type(vals) == str:
                    if vals == key:
                        breakdown_level_2_dat[keys] = val
                else:
                    idx = [key == v for v in vals].index(1)
                    breakdown_level_2_dat[keys][idx] = val
            except:
                continue
    return breakdown_level_1_dat, breakdown_level_2_dat


def plot_timing_stats(fpath, respath):
    ''' Plots the timing stats for the MPI operation.
    fpath: Stats data path
    respath: Path to save graph'''
    res1, res2 = timing_stats(fpath)
    # print('res1',res1)
    for i, j in res1.items():
        if type(j) == float:
            res1[i] = [j]
    tmp = dict([(i, sum(j)) for i, j in res1.items()])
    box_plot(pd.DataFrame([tmp]).loc[0, :], respath)
