# coding: utf-8

""" Generate a quick wavelength solution for a 1D spectrum from Modspec at MDM. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import defaultdict
import os

# Third-party
from astropy import log as logger
from astropy.io import fits
import astropy.units as u
from matplotlib.widgets import Cursor
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as so

from PyQt4 import QtGui
from PyQt4.QtCore import Qt

def gaussian_constant(pix, A, mu, sigma, B):
    return A/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5 * ((pix - mu)/sigma)**2) + B

def on_key_press(event, fig, textbox, return_dict, pixl, flux):
    ax = fig.axes[0]
    if event.key == 'enter':
        if not str(textbox.text()).strip():
            logger.error("You must enter the wavelength value of an identified line "
                         "before pressing enter over it!")

        pixl_val = event.xdata
        wvln_val = float(textbox.text())

        if wvln_val in return_dict['wavelength']:
            return

        p0 = (flux[int(round(pixl_val))],
              pixl_val,
              1.,
              flux[int(round(pixl_val))-16:int(round(pixl_val))+16].min())
        p,_ = so.curve_fit(gaussian_constant, pixl, flux, p0=p0)
        pixl_val = p[1]

        if p[2] > 5. or p[2] < 0.:
            msg = "Failed to fit line at pixel={}".format(event.xdata)
            logger.error(msg)
            fig.suptitle(msg, color='r', fontsize=12)
            return

        return_dict['wavelength'].append(wvln_val)
        return_dict['pixel'].append(pixl_val)

        ax.axvline(pixl_val, alpha=0.25, c='#2166AC')
        ax.text(pixl_val-1, ax.get_ylim()[1], "{:.1f} $\AA$".format(wvln_val),
                ha='right', va='top', rotation=90)
        fig.suptitle('')
        pl.draw()

def main(wvlnpath, outputpath, oscan_idx, c1=None, c2=None):
    """ """

    data = fits.getdata(os.path.abspath(os.path.expanduser(wvlnpath)))[:,:oscan_idx]
    pixl = np.arange(data.shape[0])
    flux = np.median(data[:,c1:c2], axis=-1)

    _wvln_pix = dict()
    _wvln_pix['wavelength'] = []
    _wvln_pix['pixel'] = []

    # make the plot
    fig,ax = pl.subplots(1,1)
    ax.plot(pixl, flux, marker=None, drawstyle='steps')
    ax.set_xlim(pixl.min(), pixl.max())

    # -------------------------------------------------------
    # magic QT stuff stolen from the interwebs to add a textbox
    root = fig.canvas.manager.window
    panel = QtGui.QWidget()
    hbox = QtGui.QHBoxLayout(panel)
    textbox = QtGui.QLineEdit(parent=panel)
    # textbox.textChanged.connect(update)
    hbox.addWidget(textbox)
    panel.setLayout(hbox)

    dock = QtGui.QDockWidget("control", root)
    root.addDockWidget(Qt.BottomDockWidgetArea, dock)
    dock.setWidget(panel)
    # -------------------------------------------------------

    fig.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event, fig, textbox, _wvln_pix, pixl, flux))
    cursor = Cursor(fig.axes[0], useblit=False, horizOn=False,
                    color='red', linewidth=1)

    fig.tight_layout()
    pl.show()

    # sort by pixel and write to file
    _ix = np.argsort(_wvln_pix['pixel'])
    pix_wvl = zip(np.array(_wvln_pix['pixel'])[_ix],
                  np.array(_wvln_pix['wavelength'])[_ix])
    with open(outputpath,'w') as f:
        txt = ["# pixel wavelength"]
        for row in pix_wvl:
            txt.append("{:.5f} {:.5f}".format(*row))
        f.write("\n".join(txt))

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    parser.add_argument("-o","--output", dest="outputpath", required=True,
                        type=str,
                        help="Path to output file, e.g., "
                             "data/mdm-spring-2016/rough-wavelength.txt")
    parser.add_argument("--comppath", dest="path", required=True,
                        type=str, help="Path to comp spectrum file.")
    parser.add_argument("--oscan-idx", dest="overscan", required=True,
                        type=int, help="Index of the overscan region.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(wvlnpath=args.path, outputpath=args.outputpath, oscan_idx=args.overscan)
