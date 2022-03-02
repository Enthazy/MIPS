from numba import jit
from utils import *
from prensenter import *

# np.random.seed(1025)


class Game:
    def __init__(self, updater, presenter, **kwargs):
        """
        posx, posy: (N,) ndarray, position
        theta: (N,) ndarray, [0,1] represent [0, 2pi], the angle that the particle toward
        :param updater:
        :param presenter:
        :param kwargs:
        """
        self.updater = updater
        self.presenter = presenter
        self.p = kwargs

        self.posx = np.array([])
        self.posy = np.array([])
        self.theta = np.array([])

        self.vecx = np.array([])
        self.vecy = np.array([])

        self.set_up(**kwargs)

    def set_up(self, N=100, **kwargs):
        self.posx = np.random.uniform(-1,1,N)
        self.posy = np.random.uniform(-1,1,N)
        self.theta = np.random.randn(N)
        self.velocity_reset(**self.p)

    def velocity_reset(self,N=100, **kwargs):
        self.vecx = np.zeros(N)
        self.vecy = np.zeros(N)

    def pos_updater(self, gamma=1, step=1e-3, **kwargs):
        self.posx += step * self.vecx / gamma
        self.posy += step * self.vecy / gamma
        self.diffusion(**self.p)


    def vec_updater(self, N=100, r=1, gamma=1, v_p=1, D=1, **kwargs):
        for i in range(N):
            # interaction
            for j in range(i + 1, N):
                # check if two particles are contacted, mostly are not contact
                self.interaction(i, j, **kwargs)

            # self mobility
            self.motion(i)

    def interaction(self, i, j, **kwargs):
        if -1 * r < self.posx[i] - self.posx[j] < r:
            if -1 * r < self.posy[i] - self.posy[j] < r:
                x_diff = self.posx[i] - self.posx[j]
                y_diff = self.posy[i] - self.posy[j]
                d2 = x_diff ** 2 + y_diff ** 2
                if d2 < kwargs['r2']:
                    # if contact add the repulsive force
                    vpx, vpy = FJPotential(x_diff, y_diff, np.sqrt(d2), **kwargs)
                    self.vecx[i] += vpx
                    self.vecy[i] += vpy
                    self.vecx[j] -= vpx
                    self.vecy[j] -= vpy

    def motion(self, i, v_p=1,**kwargs):
        self.vecx[i] += v_p * np.cos(self.theta[i])
        self.vecy[i] += v_p * np.sin(self.theta[i])


    def diffusion(self,N=100,D_r=1,D_t=1, step=1e-3, **kwargs):
        self.theta += np.sqrt(2*D_r*step)*np.random.randn(N)
        self.posx += np.sqrt(2*D_t*step)*np.random.randn(N)
        self.posy += np.sqrt(2*D_t*step)*np.random.randn(N)

    def run(self, **kwargs):
        self.vec_updater(**self.p)
        self.pos_updater(**self.p)
        self.velocity_reset(**self.p)

    def display(self):
        # self.presenter.plot(self.posx,self.posy, **self.p)
        self.presenter.plot(self.posx, self.posy)
        pass


if __name__ == '__main__':
    r = 1e-3
    r2 = r ** 2
    r6 = r ** 6

    hyperparameter = {
        'N': 3000,  # number of particles
        'step': 1e-5,
        'epoch': 10,
        'r': r,
        'r2': r2,
        'r6': r6,  # radius of particle power 6
        'gamma': 1,  # fraction
        'epsilon': 1e-10,  # potential depth times 4
        'v_p': 1,  # mean speed
        'D_t': 1,  # transition diffusion
        'D_r': 1,  # rotation diffusion
    }

    P = Presenter()
    game = Game(None, P, **hyperparameter)
    for _ in range(10):
        game.run()
    game.display()
