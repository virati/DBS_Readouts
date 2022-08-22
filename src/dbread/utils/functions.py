def unity(x):
    """
    Basic unity function
    """
    return x


def statics(x):
    """
    Static dynamics
    """
    return 0


def stables(
    x,
    c=0,
):
    """
    Stable dynamics in each dimensions at \vec{c} Γ
    """
    return -(x - c)


def consensus(x):
    """
    Basic all-to-all consensus dynamics
    """
    N = x.shape()[-1]
    L = np.ones(N, N)

    return L @ x
