import matplotlib.pyplot as plt
from time import time
import numpy as np
class Presenter:
    def __init__(self):
        pass

    def plot(self, x, y, r=3,**kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # Now, we draw our points with a gradient of colors.
        ax.scatter(x, y, linewidths=0,
                   marker='o', s=10, cmap=plt.cm.jet)
        ax.axis('equal')
        # ax.set_axis_off()
        plt.show()