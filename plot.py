import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_csv(filenames:list):
    sns.set(style="darkgrid")
    for file in filenames:
        reward = pd.read_csv(file)
        plt.plot(reward['reward'].values, label = file)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    filenames = ['Pendulum_v0_experiment','Pendulum_v0_experiment_1']
    plot_csv(filenames)