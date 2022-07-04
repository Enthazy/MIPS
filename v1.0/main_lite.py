from src import *

def main():
    # Hyper-parameters
    epoch = int(1e1)
    savetime = int(1e3)     # iterations in each epoch

    N = 4900  # number of particles
    M = 60    # number of grids
    Lx = 120  # box size x
    Ly = 120  # box size y
    step = 5e-6
    Pe = 120 #Peclet number
    is_save = True
    is_load = False
    savepoint=0
    np.random.seed(714)

    # Initialization ===============================================
    def init(N):
        coords = generate_points_with_min_distance(n=N, shape=(Lx*0.98, Ly*0.98), min_dist=1)
        qx = np.array(coords[:,0]).astype(np.float32)
        qy = np.array(coords[:,1]).astype(np.float32)
        theta = np.random.randn(N).astype(np.float32)

        px = np.zeros(N).astype(np.float32)
        py = np.zeros(N).astype(np.float32)
        return qx, qy, px, py, theta

    qx, qy, px, py, theta = init(N)
    grid = grid_init(M)

    if is_load:
        savepoint = "50_0.684"
        data = load("./results/"+str(savepoint)+".npz")
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
    os.makedirs("./results/"+str(Pe)+"_"+str(folding_frac), exist_ok=True)


    grid = grid_seperation(grid, qx, qy, M, Lx, Ly)
    # Run

    @njit()
    def body(grid, qx, qy, px, py, theta):
        for _ in range(savetime):
            set_seed(_)
            qx, qy, px, py, theta = run(step, grid, qx, qy, px, py, theta, Pe, N, M, Lx, Ly)
            if _%50==0:
                grid = grid_seperation(grid, qx, qy, M, Lx, Ly)
        return qx, qy, px, py, theta

    for _e in range(epoch):
        qx, qy, px, py, theta = body(grid, qx, qy, px, py, theta)

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
            save("./results/"+str(Pe)+"_"+str(folding_frac)+"/"+str(_e)+".npz", data)

    return qx, qy

if __name__== "__main__":
    import sys
    main()
