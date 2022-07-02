from src import *

def main():
    def init(N):
        coords = generate_points_with_min_distance(n=N, shape=(Lx*0.98, Ly*0.98), min_dist=1)
        qx = np.array(coords[:,0])
        qy = np.array(coords[:,1])
        theta = np.random.randn(N)

        px = np.zeros(N)
        py = np.zeros(N)
        return qx, qy, px, py, theta

    # Hyper-parameters
    epoch = int(3e3)

    N = 10000  # number of particles
    M = 60
    Lx = 120 # box size x
    Ly = 120 # box size y
    step = 5e-6
    Pe = 120 #Peclet number
    is_save = False
    is_load = False
    is_show = False
    savepoint=0
    np.random.seed(714)

    # Initialization ===============================================
    qx, qy, px, py, theta = init(N)
    grid = grid_init(M)

    if is_load:
        savepoint = 1703300
        data = load("./results/"+str(savepoint)+".npz")
        qx = data['qx']
        qy = data['qy']
        px = data['px']
        py = data['py']
        theta = data['theta']
        print("=========Load savepoint successfully=========")
    if is_show:
        display(qx, qy)
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    # Calculate physical quantities
    print("folding ratio is: ",cal_folding(N,Lx,Ly))
    print("Pe value is: ",Pe)

    grid = grid_seperation(grid, qx, qy, M, Lx, Ly)
    # Run
    for _ in range(epoch):
        set_seed(savepoint+_)
        t1 = time()
        qx, qy, px, py, theta=run(step, grid, qx, qy, px, py, theta, Pe, N, M, Lx, Ly)
        if _%10==0:
            grid = grid_seperation(grid, qx, qy, M, Lx, Ly)
        t2 = time()

        if _%100==0 or _<200:
            print("iteration: ", _, "time: ", t2-t1)

        if is_save and ((_ % 1000==0)
                        # or (_>1000 and _%5==0)
                        # or (_>800)
        ):
            data = {'qx': qx,
                    'qy': qy,
                    'px': px,
                    'py': py,
                    'theta': theta,
                    # 'grid':np.ndarray(grid)
                    }
            save("./results/"+str(savepoint+_)+".npz", data)
        if is_show and (_ % 1000==0):
            display(qx, qy)
            plt.show(block=False)
            plt.pause(3)
            plt.close()
    return qx, qy

if __name__== "__main__":
    # import pdb
    # try:
    #     main()
    # except Exception as e:
    #     pdb.set_trace()
    T1 = time()
    main()
    T2 = time()
    print("time: ", T2-T1)
