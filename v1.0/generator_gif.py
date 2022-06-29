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
    create_gif(sys.argv[1],sys.argv[2])
    pass
