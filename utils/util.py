import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

## Default frequency
def plot_dist(data, col, fig=None):
    n_customer=len(data)
    n_default=data[col].sum()

    ##Plotting
    if fig==None: plt.figure(figsize=(7,4))
    sns.set_context('notebook', font_scale=1.2)
    sns.countplot(col,data=data, palette="Blues")
    plt.title('CREDIT Default', size=14)