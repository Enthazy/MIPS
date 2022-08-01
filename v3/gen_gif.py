import imageio
import sys

def create_gif(start,end):
    import os
    print('Creating gif\n')
    with imageio.get_writer('mygif.avi', mode='I', fps=48) as writer:
        for filename in sorted(os.scandir("./fig"), key=lambda s: int(s.path[6:-4])):
            if int(start) <= int(filename.path[6:-4]) <= int(end):
                print(filename.path)
                image = imageio.imread(filename.path)
                writer.append_data(image)
    print('Gif saved\n')


if __name__ == "__main__":
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    name = str(sys.argv[3])
    folderpath = str(sys.argv[4])
    savepath = str(sys.argv[5])

    import os
    os.makedirs(savepath, exist_ok=True)
    startpoint=len(folderpath)
    print('Creating gif\n')
    with imageio.get_writer(savepath+name, mode='I', fps=48) as writer:
        for filename in sorted(os.scandir(folderpath), key=lambda s: int(s.path[startpoint:-4])):
            if int(start) <= int(filename.path[startpoint:-4]) <= int(end):
                print(filename.path)
                image = imageio.imread(filename.path)
                writer.append_data(image)
    print('Gif saved\n')
