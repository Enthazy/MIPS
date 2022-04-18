from refine import *


@njit()
def init(N):
    """
    posx, posy: (N,) ndarray, position
    theta: (N,) ndarray, [0,1] represent [0, 2pi], the angle that the particle toward
    """

    posx = np.random.uniform(0, 1, N)
    posy = np.random.uniform(0, 1, N)
    theta = np.random.randn(N)

    vecx = np.zeros(N)
    vecy = np.zeros(N)
    return posx, posy, vecx, vecy, theta


if __name__ == '__main__':
    N = 2  # number of particles
    step = 3e-1
    epoch = 500
    r = 1e-3
    r2 = r ** 2
    r6 = r ** 6  # radius of particle power 6
    gamma = 1  # fraction
    epsilon = 1e-10  # potential depth times 4
    v_p = 0.25  # mean speed
    D_t = 0.25  # transition diffusion
    D_r = 0.25  # rotation diffusion

    posx, posy, vecx, vecy, theta = init(N)
    display(posx, posy)
    for _ in range(epoch):
        print("iteration: ", _)
        run(posx, posy, vecx, vecy, theta,
            N, r, r2, epsilon, step, v_p, D_r, D_t, gamma)



    data = {'px': posx,
            'py': posy,
            'vx': vecx,
            'vy': vecy,
            'theta': theta
            }
    save("state.npz", data)
    display(posx, posy)