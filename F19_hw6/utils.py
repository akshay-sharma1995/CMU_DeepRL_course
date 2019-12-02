import numpy as np
import os
import argparse
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
sns.set()

def plot_prop(props,prop_names,prop_title,plot_path):

    figure = plt.figure(figsize=(16,9))
    for i,prop in enumerate(props):
        plt.plot(prop,label=prop_names[i])

    plt.xlabel("steps")
    plt.ylabel(prop_title)
    plt.legend()
    plt.title(prop_title)
    plt.savefig(os.path.join(plot_path,"{}.png".format(prop_title)))
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
