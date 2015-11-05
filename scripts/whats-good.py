# coding: utf-8

""" Print out a table of stars that would be good to observe now. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
from datetime import datetime
import warnings

# Third-party
from astropy import log as logger
from astropy.io import ascii
import astropy.coordinates as coord
import astropy.time as atime
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from gary.observation.rrlyrae import time_to_phase

kitt_peak = coord.EarthLocation.from_geodetic(lon=-111.5967*u.deg,
                                              lat=31.9583*u.deg,
                                              height=2000.*u.m)

def Vmag_to_exptime(V):
    # estimate exposure times
    s = InterpolatedUnivariateSpline([14.5,16,17.5], np.log10([150,600,2400]), k=1)
    y = 10**s(V)
    return y

def main(filename):
    filepath = os.path.abspath(os.path.expanduser(filename))
    tbl = ascii.read(filepath)
    ntargets = len(tbl)

    # get current time
    time = atime.Time(datetime.utcnow(), scale='utc')

    phases = np.zeros(ntargets)
    for i in range(ntargets):
        t0 = atime.Time(tbl[i]['mjd0'], scale='utc', format='mjd')
        phases[i] = time_to_phase(time, tbl[i]['period']*u.day, t0)

        # HACK: this corrects for the fact that CSS defines phase = 0.5
        #   to be peak brightness...Brani uses phase = 0.
        phases[i] = (phases[i] - 0.5) % 1.

    ix = (phases > 0.05) & (phases < 0.8)
    good_targets = tbl[ix]

    # add a column with estimated exposure times
    good_targets['exptime'] = [Vmag_to_exptime(V) for V in good_targets['VmagAvg']]

    # now add a "goodness" column
    g = coord.Galactic(l=good_targets['l']*u.deg,
                       b=good_targets['b']*u.deg)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        altaz = g.transform_to(coord.AltAz(obstime=time, location=kitt_peak))
    airmass = altaz.secz
    good_targets['airmass'] = airmass

    # filter on airmass
    good_targets = good_targets[good_targets['airmass'] < 1.5]

    print('{0:<16} {1:<10} {2:<10} {3:<10}'.format("ID","V [mag]","airmass","exptime [sec]"))
    print('-'*(16+10+10+10+6))
    fmt_string = '{0:<16} {1:<10.2f} {2:<10.2f} {3:<10.0f}'
    for row in good_targets[['ID2015','VmagAvg','airmass','exptime']]:
        # print(row) #['ID2015','VmagAvg','airmass','exptime'])
        print(fmt_string.format(*row))

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
    # parser.add_argument("-p", dest="plot", action="store_true", default=False,
    #                     help="Plot or not")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(filename=args.file)
