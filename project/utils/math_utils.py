import numpy as np


def draw_on_ball(n: int, r: float, dim: int = 2) -> np.array:
    """
    Draw random points on a ball
    :param n: Number of points to draw
    :param r: Ray of the ball
    :param dim: Dimension of the latent space
    :return: Array of the drawn points, shape (n,d)
    """
    rng = np.random.default_rng()
    rays = rng.uniform(0, r, n)
    points = rng.uniform(-1, 1, (n, dim))
    samples = (points / np.linalg.norm(points, axis=1)[:, None]) * rays[:, None]
    return samples
