"""
# Q = (1 / (a + b * phi)) * (c + M) / (d = eM)
a, b, c, d, and e are known constants
    a = 0.7476
    b = 0.2458
    c = 2.347
    d = 1.077
    e = 2.274
Q is the production rate of beryllium-10
M is the geomagnetic field strength
phi is the solar modulation, which we want to solve for
By https://onlinelibrary.wiley.com/doi/full/10.1155/2014/345482?msockid=00f9caf6c69469371ab8dbfbc73c68e1:
    This formula relates beryllium-10 production rate with geomagnetic field strength and solar modulation potential
    Solar modulation potential is how much the sun reduces the intensity of cosmic rays reaching the Earth
        More intense cosmic rays cause more clouds, decreasing temperature, so high solar modulation potential means less clouds
        The inverse of solar modulation potential is an effective temperature predictor
"""
a = 0.7476
b = 0.2458
c = 2.347
d = 1.077
e = 2.274


def calculate_solar_modulation(Q, M):
    """
    Calculate the solar modulation potential based on the given formula.
    """
    if Q is None or M is None:
        return None
    phi = (1 / (a + b * ((c + M) / (d + e * M)) / Q)) - a
    return phi
