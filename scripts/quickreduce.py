# coding: utf-8

""" Quick reduce a 1D spectrum from Modspec at MDM. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import defaultdict
import os

# Third-party
from astropy.io import fits
from astropy import log as logger
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as so
from scipy.stats import scoreatpercentile

def gaussian_constant(pix, A, mu, sigma, B):
    return A/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5 * ((pix - mu)/sigma)**2) + B

def min_func(p, pix, intensity):
    A,mu,sigma,B = p

    if A <= 0. or sigma <= 0. or B <= 0 or mu < pix.min() or mu > pix.max():
        return 1E10

    model = gaussian_constant(pix, A, mu, sigma, B)
    return np.sum((model - intensity)**2)

def main(datafile, flatfile, biasfile, overscan_idx,
         dispersion_stride=1):
    """ """

    SIGMAINIT = 1. # pixels, for fitting Gaussian to row

    # read data from files
    data = fits.getdata(os.path.abspath(os.path.expanduser(datafile)), 0)
    flat = fits.getdata(os.path.abspath(os.path.expanduser(flatfile)), 0)
    bias = fits.getdata(os.path.abspath(os.path.expanduser(biasfile)), 0)

    with np.errstate(divide='ignore'):
        # C = (data - bias) / (flat - bias)
        C = data / flat

    tmp = defaultdict(lambda: None)
    def on_key_press(event, fig):
        if event.key == 'c':
            tmp['dispersion_axis'] = 0
            tmp['spectrum_idx'] = event.xdata
            pl.close(fig)

        elif event.key == 'l' or event.key == 'r':
            tmp['dispersion_axis'] = 1
            tmp['spectrum_idx'] = event.ydata
            pl.close(fig)

    # TODO: make this less hacky
    # vmin = -0.1
    # vmax = 0.2
    vmin = 0.5
    vmax = 1.5

    fig,ax = pl.subplots(1,1)
    ax.imshow(C, cmap='Greys_r', vmin=vmin, vmax=vmax,
              interpolation='nearest', origin='lower')
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event, fig))
    fig.tight_layout()
    pl.show()

    dispersion_axis = tmp['dispersion_axis']
    spectrum_idx = tmp['spectrum_idx']

    if dispersion_axis is None:
        raise ValueError("Did you close the plot without selecting a row "
                         "or column to extract along?")

    C_swap = C.swapaxes(dispersion_axis, 0)

    # slice out overscan region
    overscan = C_swap[:,overscan_idx:]
    C_swap = C_swap[:,:overscan_idx]

    # pixels down dispersion axis
    dispersion_px = np.arange(0, C_swap.shape[0], dispersion_stride, dtype=int)
    amps = np.zeros_like(dispersion_px).astype(float)
    locs = np.zeros_like(dispersion_px).astype(float)

    # pixels across slit
    row_px = np.arange(C_swap.shape[1])

    for j,i in enumerate(dispersion_px):
        row = C_swap[i]
        p0 = (row[int(round(spectrum_idx))],
              spectrum_idx,
              SIGMAINIT,
              np.median(row))

        res = so.minimize(min_func, x0=p0, method='powell',
                          args=(row_px, row))
        amps[j] = res.x[0]
        locs[j] = res.x[1]

        # if i == 1360 or i == 1372:
        #     print(res.x)
        #     pl.clf()
        #     pl.plot(row_px, row, marker=None, drawstyle='steps', lw=1.)
        #     pl.plot(row_px, gaussian_constant(row_px, *res.x), marker=None, drawstyle='steps', lw=1.)
        #     pl.axvline(res.x[1])
        #     pl.axvline(p0[1], linestyle='dashed')
        #     pl.show()

    # fig,ax = pl.subplots(1,1)
    # ax.plot(locs, marker=None, drawstyle='steps')
    # pl.show()

    fig,ax = pl.subplots(1,1)
    ax.plot(dispersion_px, amps, marker=None, drawstyle='steps')
    pl.show()

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    parser.add_argument("--data", dest="datafile", required=True,
                        type=str, help="Data frame.")
    parser.add_argument("--flat", dest="flatfile", required=True,
                        type=str, help="Flatfield image.")
    parser.add_argument("--bias", dest="biasfile", required=True,
                        type=str, help="Bias image.")
    parser.add_argument("--overscan-idx", dest="overscan", required=True,
                        type=int, help="Index of the overscan region.")

    parser.add_argument("--stride", dest="stride", default=1,
                        type=int, help="Skip every X row on the CCD.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(datafile=args.datafile, flatfile=args.flatfile, biasfile=args.biasfile,
         overscan_idx=args.overscan, dispersion_stride=args.stride)
