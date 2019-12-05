import numpy as np
import os
import argparse
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
# import seaborn as sns
# sns.set()

def plot_prop(props,prop_names,prop_title,plot_path,xlabel=None):

    figure = plt.figure(figsize=(16,9))
    for i,prop in enumerate(props):
        plt.plot(prop,label=prop_names[i])

    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel("steps")
    plt.ylabel(prop_title)
    plt.legend()
    plt.grid(True)
    plt.title(prop_title)
    plt.savefig(os.path.join(plot_path,"{}.png".format(prop_title)),bbox_inches='tight')
    figure.clf()
    plt.close()

def plot_x_v_y(x,y,prop_title,plot_path):

    figure = plt.figure(figsize=(16,9))
    plt.plot(x,y,'-x')

    plt.xlabel("q1")
    plt.ylabel("q2")
    plt.grid(True)
    plt.title(prop_title)
    plt.savefig(os.path.join(plot_path,"{}.png".format(prop_title)),bbox_inches='tight')
    figure.clf()
    plt.close()


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', dest='algo', type=str,
                        default="lqr", help="lqr or ilqr ")

    return parser.parse_args()

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
