import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def plot_data(data: pd.DataFrame, save_path=None, file_name: str=None, file_ext=".png",
              title=None, x_name="Epoch", y_name="Value",
              figsize=(10,5)):
    plt.figure(figsize=figsize)
    
    ax = plt.plot(data[x_name], data[y_name])
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    if save_path:
        assert file_name is not None, "file_name is required when saving the plot"
        file_name = file_name.split(".")[0] + file_ext
        path = os.path.join(save_path, file_name)
        plt.savefig(path, facecolor="w",
                    bbox_inches="tight", dpi=300)
        
    plt.clf()
        

def plot_data_scatter(x, y, save_path=None, file_name: str=None, file_ext=".png",
              title=None, x_name="Feature 1", y_name="Feature 2",
              figsize=(10,5)):
    plt.figure(figsize=figsize)
    
    ax = plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    if save_path:
        assert file_name is not None, "file_name is required when saving the plot"
        file_name = file_name.split(".")[0] + file_ext
        path = os.path.join(save_path, file_name)
        plt.savefig(path, facecolor="w",
                    bbox_inches="tight", dpi=300)
        
    plt.clf()