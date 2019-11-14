from scipy import stats
import numpy as np


#

class SubDistribution(stats.rv_continuous):
    def __init__(self, gauss_weight, point_dist_weight, point_centers, point_weights, mu=0, sigma=1):
        super(SubDistribution, self).__init__()
        self.gauss_weight = gauss_weight
        self.point_dist_weight = point_dist_weight
        self.point_centers = point_centers
        self.point_weights = point_weights

        self.normal_dist = stats.norm(loc=mu, scale=sigma)  # scale is standard deviation

    def _pdf(self, x):
        gauss_pdf = self.normal_dist.pdf(x)
        point_pdf = np.array(
            [self.point_weights[self.point_centers.index(x_i)] if x_i in self.point_centers else 0 for x_i in x])

        return self.gauss_weight * gauss_pdf + self.point_dist_weight * point_pdf

    def _cdf(self, x):
        gauss_cdf = self.normal_dist.cdf(x)
        point_cdf = np.array(
            [sum(w for p_c, w in zip(self.point_centers, self.point_weights) if p_c < x_i) for x_i in x])

        return self.gauss_weight * gauss_cdf + self.point_dist_weight * point_cdf


class MainDistribution():
    def __init__(self, gauss_weight, point_dist_weight, point_centers_matrix, point_weights, mu=0, sigma=1):
        self.gauss_weight = gauss_weight  # (tau1/tau2)/(M+(tau1/tau2))
        self.point_dist_weight = point_dist_weight  # (1/(M+(tau1/tau2))
        self.point_centers_matrix = point_centers_matrix  # theta_ks; size K by num_feats
        self.point_weights = point_weights  # sum_m=1^M phi_m,k; size K
        self.mu = mu
        self.sigma = sigma
        self.dists = []  # list of distributions, assumed to be independent of one another; size of num_feats
        self.set_up_dists()

    def set_up_dists(self):
        # print [self.point_centers_matrix[j][0] for j in range(len(self.point_centers_matrix))]
        # print [self.point_centers_matrix[j][1] for j in range(len(self.point_centers_matrix))]
        self.dists = [SubDistribution(self.gauss_weight, self.point_dist_weight,
                                      [self.point_centers_matrix[j][i] for j in range(len(self.point_centers_matrix))],
                                      self.point_weights, self.mu, self.sigma) for i in
                      range(len(self.point_centers_matrix[0]))]

    def rvs(self, size):
        random_sample = []
        for i in range(size):
            random_sample.append([d.rvs() for d in self.dists])

        return random_sample

    def marginal_pdfs(self, x):
        return [d.pdf(x) for d in self.dists]

    def marginal_cdfs(self, x):
        return [d.cdf(x) for d in self.dists]


def test_single_distribution():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm

    distribution = SubDistribution(.5, .5, [0, 2.39994], [.2, .8], mu=.5)

    x_vals = np.linspace(-3, 3, 100001)
    plt.plot(x_vals, distribution.pdf(x_vals))
    plt.show()

    plt.plot(x_vals, distribution.cdf(x_vals))
    plt.show()

    samples = distribution.rvs(size=1000)

    print "Percent of samples at 0", sum(abs(samples) < 1e-13) / float(len(samples))
    print "Percent of samples at 2", sum(abs(samples - 2.39994) < 1e-13) / float(len(samples))

    plt.hist(samples, bins=np.arange(-4, 4, .25), normed=True)
    plt.plot(x_vals, distribution.pdf(x_vals))
    plt.show()


def test_full_distribution():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm

    distribution = MainDistribution(.5, .5, [[0, .5], [0, -2.0], [-3.0, 3.0]], [.2, .6, .2])

    x_vals = np.linspace(-5, 5, 1001)
    pdfs = distribution.marginal_pdfs(x_vals)
    cdfs = distribution.marginal_cdfs(x_vals)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x_vals, pdfs[0])
    plt.subplot(2, 1, 2)
    plt.plot(x_vals, pdfs[1])
    plt.show()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x_vals, cdfs[0])
    plt.subplot(2, 1, 2)
    plt.plot(x_vals, cdfs[1])
    plt.show()

    samples = distribution.rvs(size=3000)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist([s[0] for s in samples])
    plt.subplot(2, 1, 2)
    plt.hist([s[1] for s in samples])
    plt.show()

    # normal distribution center at x=0 and y=5

    plt.hist2d([s[0] for s in samples], [s[1] for s in samples], bins=40, norm=LogNorm())
    plt.colorbar()
    plt.show()

    joint_pdf = np.atleast_2d(pdfs[0]) * np.atleast_2d(pdfs[0]).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot a basic wireframe.
    # ax.plot_wireframe(np.matlib.repmat(x_vals,1001,1), np.matlib.repmat(np.atleast_2d(x_vals).T,1,1001), joint_pdf, rstride=10, cstride=10)
    surf = ax.plot_surface(np.matlib.repmat(x_vals, 1001, 1), np.matlib.repmat(np.atleast_2d(x_vals).T, 1, 1001),
                           joint_pdf, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    plt.show()


