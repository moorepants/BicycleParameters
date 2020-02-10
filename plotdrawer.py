import os
import bicycleparameters as bp
import numpy as np

speeds = np.linspace(0, 10, num=100)

    # Clunkily draws and saves default plots 
def save_plot():
    bike = bp.Bicycle('Silver', pathToData=os.getcwd()+'\\data')  # Alter Bicycle name
    #bike.add_rider('Jason', reCalc=True)
    plot = bike.plot_eigenvalues_vs_speed(speeds, show=False)
    plot.savefig(os.getcwd()+'\\assets\\eigen-plots\\defaults\\Silver.png')  # Alter png name

def draw_plot():
    bike = bp.Bicycle('', pathToData=os.getcwd()+'\\data')
    plot = bike.plot_eigenvalues_vs_speed(speeds, show=True)
save_plot()


