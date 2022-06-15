from src import *

def main():
    def init(N):
        coords = generate_points_with_min_distance(n=N, shape=(Lx*0.98, Ly*0.98), min_dist=2)
        posx = np.array(coords[:,0])
        posy = np.array(coords[:,1])
        theta = np.random.randn(N)

        vecx = np.zeros(N)
        vecy = np.zeros(N)
        return posx, posy, vecx, vecy, theta

    # Hyper-parameters
    epoch = int(1e4)

    N = 15625  # number of particles
    M = 70
    Lx = 260 # box size x
    Ly = 260 # box size y
    step = 5e-6
    Pe = 100 #Peclet number
    is_save = True
    is_load = False
    is_show = False
    savepoint=0

    # Initialization ===============================================
    posx, posy, vecx, vecy, theta = init(N)
    print(posx.shape)
    grid = grid_init(M)

    if is_load:
        savepoint = 4600
        data = load("./results/state"+str(savepoint)+".npz")
        posx = data['px']
        posy = data['py']
        vecx = data['vx']
        vecy = data['vy']
        theta = data['theta']
        print("=========Load savepoint successfully=========")
    if is_show:
        display(posx, posy)
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    # Calculate physical quantities
    print("folding ratio is: ",cal_folding(N,Lx,Ly))
    print("Pe value is: ",Pe)

    grid = grid_seperation(grid, posx, posy, M, Lx, Ly)
    # Run
    set_seed(714)
    for _ in range(epoch):
        t1 = time()
        posx, posy, vecx, vecy, theta=run(step, grid, posx, posy, vecx, vecy, theta, Pe, N, M, Lx, Ly)
        if _%5==0:
            grid = grid_seperation(grid, posx, posy, M, Lx, Ly)
        t2 = time()

        if _%100==0 or _<200:
            print("iteration: ", _, "time: ", t2-t1)

        if is_save and (_ % 100==0):
            data = {'px': posx,
                    'py': posy,
                    'vx': vecx,
                    'vy': vecy,
                    'theta': theta
                    }
            save("./results/state"+str(savepoint+_)+".npz", data)
        if is_show and (_ % 100==0):
            display(posx, posy)
            plt.show(block=False)
            plt.pause(3)
            plt.close()
    return posx, posy

if __name__== "__main__":
    # import pdb
    # try:
    #     main()
    # except Exception as e:
    #     pdb.set_trace()
    main()