import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def plot_rewards(reward_list, postfix):
    folder = './results/'
    filename = 'rewards_' + postfix
    if not os.path.exists(folder):
        os.mkdir(folder)
    plt.plot(np.arange(len(reward_list)), reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.savefig(os.path.join(folder, filename))
    plt.close()


def plot_actions(action_list, postfix):
    folder = './results/'
    filename = 'actions_' + postfix
    if not os.path.exists(folder):
        os.mkdir(folder)
    plt.plot(np.arange(len(action_list)), action_list)
    plt.xlabel('Episode')
    plt.ylabel('action')
    plt.savefig(os.path.join(folder, filename))
    plt.close()


def plot_dif(dif_list, postfix):
    folder = './results/'
    filename = 'dif_' + postfix
    if not os.path.exists(folder):
        os.mkdir(folder)
    plt.plot(np.arange(len(dif_list)), dif_list)
    plt.xlabel('Episode')
    plt.ylabel('QTable Update Difference')
    plt.savefig(os.path.join(folder, filename))
    plt.close()