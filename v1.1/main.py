from src import *
from time import time

step_unit = 1e-8

def main():
    import sys
    # Hyper-parameters
    epoch = np.int64(sys.argv[1])  # how many savesfile will be generated
    savetime = np.int64(sys.argv[2])  # iterations in each epoch
    # how many iterations for each savefile
    grid_update_time = int(20)  # how many iterations for each grid update

    N = np.int64(sys.argv[3])  # number of particles
    M = np.int64(sys.argv[4])  # number of grids
    step = np.float64(float(sys.argv[5]) * step_unit) # time step
    Lx = np.int64(sys.argv[6])  # box size x
    Ly = np.int64(sys.argv[6])  # box size y
    Pe = np.int64(sys.argv[7])  # Peclet number
    is_save = True
    is_load = False
    savepoint = 0
    seed = 123 #np.random.randint(1000)
    np.random.seed(seed)

    # Initialization ========================================================================================
    def init(N):
        coords = generate_points_with_min_distance(n=N, shape=(Lx * 0.98, Ly * 0.98), min_dist=1)
        qx = np.array(coords[:, 0])
        qy = np.array(coords[:, 1])
        qtheta = np.random.randn(N)

        ax = np.copy(qx)
        ay = np.copy(qy)

        # px = np.zeros(N).astype(np.float64)
        # py = np.zeros(N).astype(np.float64)
        px = np.random.uniform(-1,1,N) * Pe / 2
        py = np.random.uniform(-1,1,N) * Pe / 2
        return qx, qy, qtheta, px, py, ax, ay

    qx, qy, qtheta, px, py, ax, ay = init(N)
    grid = grid_init(M)

    if is_load:
        savepoint = "50_0.684"
        data = load("./results/" + str(savepoint) + ".npz")
        qx = data['qx']
        qy = data['qy']
        px = data['px']
        py = data['py']
        qtheta = data['qtheta']
        print("=========Load savepoint successfully=========")

    # Calculate physical quantities
    folding_frac = int(cal_folding(N, Lx, Ly)*100)
    print("folding ratio is: ", folding_frac)
    print("Pe value is: ", Pe)
    import os
    os.makedirs('./results/final/', exist_ok=True)
    os.makedirs("./results/"
                + "F"+str(folding_frac)
                + "P"+str(Pe)
                + "T" + str(int(round(step/step_unit))), exist_ok=True)

    grid = grid_seperation(grid, qx, qy, M, Lx, Ly)

    # Random generator
    rng = np.random.default_rng(seed=seed)


    # Run=================================================================================================
    @njit()
    def body(qx, qy, qtheta, px, py, ax, ay, grid, random_pool):
        for _ in range(savetime):
            # get random variables
            _N = _*3*N
            s_x = random_pool[_N:_N+N]
            s_y = random_pool[_N+N:_N+2*N]
            s_theta = random_pool[_N+2*N:_N+3*N]

            qx, qy, qtheta, px, py, ax, ay = run(qx, qy, qtheta, px, py,
                                                         ax, ay,
                                                         s_x, s_y, s_theta,
                                                         grid, step, Pe, M, Lx, Ly)

            if _ % grid_update_time == 0:
                grid = grid_seperation(grid, qx, qy, M, Lx, Ly)
        return qx, qy, qtheta, px, py, ax, ay


    # Main Loop
    for _e in range(epoch):
        T1 = time()
        random_pool = rng.standard_normal(3*N*savetime, dtype=np.float64)
        qx, qy, qtheta, px, py, ax, ay = body(qx, qy, qtheta,
                                                      px, py,
                                                      ax, ay,
                                                      grid, random_pool)
        T2 = time()
        print("time: ", T2-T1)
        if is_save:
            data = {'epoch': _e,
                    'Pe': Pe,
                    'PackFrac': folding_frac,
                    'N': N,
                    'L': Lx,

                    'qx': qx,
                    'qy': qy,
                    'qtheta': qtheta,
                    'px': px,
                    'py': py,
                    'ax': ax,
                    'ay': ay,
                    }
            save("./results/"
                 + "F"+str(folding_frac)
                 + "P"+str(Pe)
                 + "T" + str(int(round(step/step_unit)))
                 + "/"
                 + str(_e) + ".npz", data)
    save("./results/final/"
         + "F"+str(folding_frac)
         + "P"+str(Pe)
         + "T" + str(int(round(step/step_unit)))
         + ".npz", data)
    return qx, qy

if __name__ == "__main__":
    main()