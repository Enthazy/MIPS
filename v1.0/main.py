from src import *
from time import time

def main():
    import sys
    # Hyper-parameters
    epoch = np.int32(sys.argv[1])  # how many savesfile will be generated
    savetime = np.int32(sys.argv[2])  # iterations in each epoch
    # how many iterations for each savefile
    grid_update_time = int(50)  # how many iterations for each grid update

    N = np.int32(sys.argv[3])  # number of particles
    M = np.int32(sys.argv[4])  # number of grids
    Lx = np.int32(sys.argv[5])  # box size x
    Ly = np.int32(sys.argv[5])  # box size y
    step = np.float32(5e-6)
    Pe = np.int32(sys.argv[6])  # Peclet number
    is_save = True
    is_load = False
    savepoint = 0
    np.random.seed(714)

    # Initialization ===============================================
    def init(N):
        coords = generate_points_with_min_distance(n=N, shape=(Lx * 0.98, Ly * 0.98), min_dist=1)
        qx = np.array(coords[:, 0]).astype(np.float32)
        qy = np.array(coords[:, 1]).astype(np.float32)
        theta = np.random.randn(N).astype(np.float32)

        px = np.zeros(N).astype(np.float32)
        py = np.zeros(N).astype(np.float32)
        return qx, qy, px, py, theta

    qx, qy, px, py, theta = init(N)
    grid = grid_init(M)

    if is_load:
        savepoint = "50_0.684"
        data = load("./results/" + str(savepoint) + ".npz")
        qx = data['qx']
        qy = data['qy']
        px = data['px']
        py = data['py']
        theta = data['theta']
        print("=========Load savepoint successfully=========")

    # Calculate physical quantities
    folding_frac = round(cal_folding(N, Lx, Ly), 3)
    print("folding ratio is: ", folding_frac)
    print("Pe value is: ", Pe)
    import os
    os.makedirs("./results/" + str(Pe) + "_" + str(folding_frac), exist_ok=True)

    grid = grid_seperation(grid, qx, qy, M, Lx, Ly)

    # Random generator
    rng = np.random.default_rng(seed=0)
    # Run

    @njit()
    def body(grid, qx, qy, px, py, theta, random_pool):
        for _ in range(savetime):
            # get random variables
            _N = _*3*N
            s_x = random_pool[_N:_N+N]
            s_y = random_pool[_N+N:_N+2*N]
            s_theta = random_pool[_N+2*N:_N+3*N]

            qx, qy, px, py, theta = run(step, grid, qx, qy, px, py, theta,
                                        s_x, s_y, s_theta,
                                        Pe, N, M, Lx, Ly)
            if _ % grid_update_time == 0:
                grid = grid_seperation(grid, qx, qy, M, Lx, Ly)
        return qx, qy, px, py, theta


    # Main Loop
    for _e in range(epoch):
        T1 = time()
        random_pool = rng.standard_normal(3*N*savetime, dtype=np.float32)
        qx, qy, px, py, theta = body(grid, qx, qy, px, py, theta, random_pool)
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
                    'px': px,
                    'py': py,
                    'theta': theta,
                    }
            save("./results/" + str(Pe) + "_" + str(folding_frac) + "/" + str(_e) + ".npz", data)

    return qx, qy

if __name__ == "__main__":
    main()