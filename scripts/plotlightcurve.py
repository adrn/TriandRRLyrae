"""
    Plot the Catalina light curve for a given target name in a catalog file

    Examples:

    python scripts/plotlightcurve.py -f ../spring2016/targets/GASSA13_2016.txt -n GA-RR14

"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import glob
import os
import sys

# Third-party
import astropy.coordinates as coord
from astropy import log as logger
from astropy.io import ascii
import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np
from numpy.lib.recfunctions import stack_arrays

def plot_lc(mjd, mag, mag_err, period, mjd0=None, ylim=None):
    if ylim is None:
        ylim = (np.median(mag) + 1, np.median(mag) - 1)

    if mjd0 is None:
        mjd0 = np.min(mjd)

    fig,axes = pl.subplots(1,2,figsize=(8,5),sharey=True, dpi=128)

    axes[0].errorbar(mjd, mag, mag_err, ls='none', marker='.', ecolor='#aaaaaa', color='k', alpha=0.5)
    axes[0].set_xlabel("MJD [day]")
    axes[0].set_ylabel("mag")
    axes[0].set_ylim(ylim)

    # phase folded
    phase = phase = ((mjd - mjd0) / period) % 1.
    axes[1].errorbar(phase, mag, mag_err, ls='none', marker='.', ecolor='#aaaaaa', color='k', alpha=0.5)
    axes[1].set_xlabel("phase")

    return fig,axes

def main(filename, target_name):
    CSS_LC_file = 'data/CSS_RR_phot.npy'
    CSS_csv_file = 'data/catalina.csv'

    filepath = os.path.abspath(os.path.expanduser(filename))
    tbl = ascii.read(filepath)
    target_row = tbl[tbl['ID'] == target_name]
    if len(target_row) != 1:
        raise ValueError("Invalid name!")
    target_c = coord.SkyCoord(ra=target_row['ra']*u.degree, dec=target_row['dec']*u.degree)

    css = ascii.read(CSS_csv_file)
    css_c = coord.SkyCoord(ra=css['ra']*u.degree, dec=css['dec']*u.degree)

    idx,sep,_ = target_c.match_to_catalog_sky(css_c)
    css_row = css[idx[0]]

    all_data = []
    if not os.path.exists(CSS_LC_file):
        for filename in glob.glob('data/CSS_RR_phot/*.phot'):
            data = np.genfromtxt(filename, delimiter=',', dtype=None,
                                 names=["CSSID", "MJD", "mag", "mag_err", "ra", "dec"])
            all_data.append(data)
        data = stack_arrays(all_data, usemask=False)
        np.save(CSS_LC_file, data)
        del all_data
    else:
        data = np.load(CSS_LC_file)

    for sep_radius in [10, 5, 2.5, 1]*u.arcsecond:
        # match target coordinates to data
        c = coord.ICRS(ra=data['ra']*u.degree, dec=data['dec']*u.degree)
        sep = target_c.separation(c)
        idx = (sep < sep_radius)

        target_data = data[idx]
        if len(np.unique(target_data['CSSID'])) == 1:
            break

    if len(np.unique(target_data['CSSID'])) > 1:
        raise ValueError("More than one CSS target found for this target name!")

    fig,axes = plot_lc(target_data['MJD'], target_data['mag'], target_data['mag_err'],
                       period=css_row['period'], mjd0=css_row['mjd0'])
    pl.tight_layout()
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

    parser.add_argument("-f", "--file", dest="file", required=True,
                        type=str, help="Catalog file.")
    parser.add_argument("-n", "--name", dest="target_name", required=True,
                        type=str, help="Name of the target to plot the light curve for.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(filename=args.file, target_name=args.target_name)

