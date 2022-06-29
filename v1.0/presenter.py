import matplotlib.pyplot as plt

from utils import *
from src import *
import imageio
import sys
def presenter(savepoint):
    data = load("./results/"+str(savepoint)+".npz")
    qx = data['qx']
    qy = data['qy']
    px = data['px']
    py = data['py']
    theta = data['theta']
    print("=========Load savepoint successfully=========")
    display(qx, qy)
    plt.title(savepoint)
    # plt.show()

def presenter_circle(savepoint):
    data = load("./results/"+str(savepoint)+".npz")
    qx = data['qx']
    qy = data['qy']
    px = data['px']
    py = data['py']
    theta = data['theta']
    print("=========Load savepoint successfully=========")
    display_circle(qx, qy)
    plt.title(savepoint)
    # plt.show()

def presenter_label(savepoint,range='all'):
    data = load("./results/"+str(savepoint)+".npz")
    qx = data['qx']
    qy = data['qy']
    px = data['px']
    py = data['py']
    theta = data['theta']
    print("=========Load savepoint successfully=========")
    if range == 'all':
        for i, txt in enumerate(qx):
            plt.annotate(i,(qx[i],qy[i]))
    else:
        for i in range:
            plt.annotate(i,(qx[i],qy[i]))

def presenter_arrow(savepoint, range='all',step=5e-6):
    data = load("./results/"+str(savepoint)+".npz")
    qx = data['qx']
    qy = data['qy']
    data1 = load("./results/"+str(savepoint)+".npz")
    px = data1['px']
    py = data1['py']
    theta = data['theta']
    print("=========Load savepoint successfully=========")
    if range == 'all':
        for i,_ in enumerate(qx):
            plt.arrow(qx[i],qy[i],step*px[i],step*py[i])
    else:
        for i in range:
            plt.arrow(qx[i],qy[i],step*px[i],step*py[i])

def generate_figure(filename):
    presenter(filename[:-4])
    plt.savefig("./fig/"+filename[:-3]+"png")
    plt.cla()
    plt.clf()
    plt.close('all')


def generate_all_figures(folder_name):
    import os
    import gc
    for filename in sorted(os.scandir(folder_name), key=lambda s: s.path.lower()):
        presenter(filename.path[10:-4])
        plt.savefig("./fig/"+filename.path[9:-3]+"png")
        plt.cla()
        plt.clf()
        plt.close('all')
        gc.collect()


def create_gif():
    import os
    print('Creating gif\n')
    with imageio.get_writer('mygif.avi', mode='I',fps=120) as writer:
        for filename in sorted(os.scandir("./fig"), key=lambda s: int(s.path[6:-4])):
            print(filename.path)
            image = imageio.imread(filename.path)
            writer.append_data(image)
    print('Gif saved\n')



def show_savepoint(savepoint):
    presenter(savepoint)
    presenter_label(savepoint,[])

    presenter_circle(savepoint)
    presenter_label(savepoint,[241,40,41,141,140])
    # presenter_arrow(savepoint)
    plt.show()


if __name__=="__main__":
    show_savepoint(9900)
    # generate_all_figures("./results")
    # generate_figure(sys.argv[1])
    # create_gif()
    pass