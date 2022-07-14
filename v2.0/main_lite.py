from src import *
from time import time

def main():
    # Hyper-parameters
    epoch = int(1e3)  # how many savesfile will be generated
    savetime = int(1e3)  # iterations in each epoch
    # how many iterations for each savefile
    grid_update_time = int(50)  # how many iterations for each grid update

    N = 10000  # number of particles
    M = 20  # number of grids
    Lx = 120  # box size x
    Ly = Lx  # box size y
    step = 5e-6
    Pe = 120  # Peclet number
    W = 0.05   # W number
    is_save = True
    is_load = False
    savepoint = 0
    np.random.seed(714)

    # Initialization ===============================================
    def init(N):
        coords = generate_points_with_min_distance(n=N, shape=(Lx * 0.98, Ly * 0.98), min_dist=1)
        qx = np.array(coords[:, 0]).astype(np.float32)
        qy = np.array(coords[:, 1]).astype(np.float32)
        qtheta = np.random.randn(N).astype(np.float32)

        px = np.zeros(N).astype(np.float32)
        py = np.zeros(N).astype(np.float32)
        ptheta = np.zeros(N).astype(np.float32)
        return qx, qy, qtheta, px, py, ptheta

    qx, qy, qtheta, px, py, ptheta = init(N)
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
    def body(grid, qx, qy, qtheta, px, py, ptheta, random_pool):
        for _ in range(savetime):
            # get random variables
            _N = _*3*N
            s_x = random_pool[_N:_N+N]
            s_y = random_pool[_N+N:_N+2*N]
            s_theta = random_pool[_N+2*N:_N+3*N]

            qx, qy, qtheta, px, py, ptheta = run(step, grid, qx, qy, qtheta, px, py, ptheta,
                                         s_x, s_y, s_theta,
                                         Pe, W, M, Lx, Ly)

            if _ % grid_update_time == 0:
                grid = grid_seperation(grid, qx, qy, M, Lx, Ly)
        return qx, qy, qtheta, px, py, ptheta


    # Main Loop
    for _e in range(epoch):
        T1 = time()
        random_pool = rng.standard_normal(3*N*savetime, dtype=np.float32)
        qx, qy, qtheta, px, py, ptheta = body(grid, qx, qy, qtheta, px, py, ptheta, random_pool)
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
                    'ptheta': ptheta,
                    }
            save("./results/" + str(Pe) + "_" + str(folding_frac) + "/" + str(_e) + ".npz", data)

    return qx, qy

if __name__ == "__main__":
    main()