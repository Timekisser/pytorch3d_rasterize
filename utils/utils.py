import os
import numpy as np

def get_filenames(filelist, filelist_folder):
    with open(os.path.join(filelist_folder, filelist), 'r') as fid:
        lines = fid.readlines()
    filenames = [line.strip() for line in lines]
    return filenames



PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]


def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)



def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]


def generate_views(num_views, offset=(0, 0)):
    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)

    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views  # 弧度转度

    # views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f}
    #          for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    views = [{'azim': y / np.pi * 180, 'elev': p / np.pi * 180, 'radius': r, 'fov': f}
             for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    return views
