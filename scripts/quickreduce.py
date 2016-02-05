# coding: utf-8

""" Quick reduce a 1D spectrum from Modspec at MDM. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import defaultdict
import os

# Third-party
import ccdproc
from astropy import log as logger
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as pl
from matplotlib.widgets import Cursor
import numpy as np
import scipy.optimize as so
from astropy.modeling import models
from scipy.signal import medfilt2d
import scipy.interpolate as si

# suppress warning about correlated errors
from astropy import nddata
nddata.conf.warn_unsupported_correlated = False

# Properties of Echelle at MDM
gain = 1.3 * u.electron / u.adu
readnoise = 7.9 * u.electron

def gaussian_constant(pix, A, mu, sigma, B):
    return A/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5 * ((pix - mu)/sigma)**2) + B

def oscan_and_trim(image_list, overscan_idx):
    """
    Subtract overscan and trim out the overscan region for each image
    in a list of images. The original list is modified in place.
    """
    for idx, img in enumerate(image_list):
        oscan = ccdproc.subtract_overscan(img, img[:, overscan_idx+8:overscan_idx+72], model=models.Polynomial1D(1), add_keyword=None)
        image_list[idx] = ccdproc.trim_image(oscan[:, :overscan_idx], add_keyword=None)

def auto_scale_imshow(d, nsigma=5, **kwargs):
    _mean = np.mean(d)
    _std = np.std(d)

    fig,ax = pl.subplots(1,1,**kwargs)
    ax.imshow(d, cmap='Greys_r', interpolation='nearest',
              vmax=_mean + 5*_std,
              vmin=_mean - 5*_std)
    return fig

def make_master_bias(images, oscan_idx, plot=False):
    """ Create a master bias file from the image collection """
    bias_list = []
    for hdu, fname in images.hdus(imagetyp='bias', return_fname=True):
        bias_list.append(ccdproc.CCDData(hdu.data*u.adu))
    logger.info("Using {} bias frames.".format(len(bias_list)))

    oscan_and_trim(bias_list, oscan_idx)

    # median combine the bias images
    biases = ccdproc.Combiner(bias_list)
    master_bias = biases.median_combine()

    if plot:
        fig = auto_scale_imshow(master_bias.data, figsize=(10,10))

    return master_bias

def make_master_flat(images, master_bias, oscan_idx, plot=False):
    """ Create a master flatfield from the image collection """
    flat_list = []
    for hdu, fname in images.hdus(imagetyp='flat', return_fname=True):
        flat_list.append(ccdproc.CCDData(hdu.data*u.adu))
    logger.info("Using {} flat frames.".format(len(flat_list)))

    oscan_and_trim(flat_list, oscan_idx)

    # bias correct the flats
    for i,flat in enumerate(flat_list):
        flat_list[i] = ccdproc.subtract_bias(flat, master_bias, add_keyword=None)

    # combine the flats using sigma clipping to reject cosmic rays (?)
    flat_combiner = ccdproc.Combiner(flat_list)
    flat_combiner.sigma_clipping(func=lambda x,axis: np.nanmedian(x,axis))
    master_flat = flat_combiner.median_combine()

    # correct for gain in CCD
    master_flat_e = ccdproc.gain_correct(master_flat, gain=gain, add_keyword=None)

    if plot:
        fig = auto_scale_imshow(master_flat_e.data, figsize=(10,10))

    return master_flat_e

def main(path, datafile, oscan_idx, wavelengthfile=None, ntracebins=32, median_filter_kernel=3, interactive=False):
    """ """

    # all FITS images
    images = ccdproc.ImageFileCollection(path, keywords='*')
    master_bias = make_master_bias(images, oscan_idx)
    master_flat = make_master_flat(images, master_bias, oscan_idx)

    # read the data frame
    data_path = os.path.abspath(os.path.expanduser(datafile))
    obj_img = ccdproc.CCDData.read(data_path, unit=u.adu)
    obj_img = [obj_img] # HACK
    oscan_and_trim(obj_img, oscan_idx) # HACK
    obj_img = obj_img[0] # HACK

    obj_img = ccdproc.subtract_bias(obj_img, master_bias, add_keyword=None)
    obj_img = ccdproc.flat_correct(obj_img, master_flat, add_keyword=None)

    if median_filter_kernel > 1:
        d = medfilt2d(obj_img.data, kernel_size=(median_filter_kernel,median_filter_kernel))
    else:
        d = obj_img.data

    # HACK:
    dispersion_axis = 0

    if interactive:
        tmp = defaultdict(list)
        def on_key_press(event, fig):
            if event.key == 'enter': # left of peak
                if tmp['indices'] is None:
                    tmp['indices'] = []
                tmp['indices'].append(event.xdata)

                if tmp['_lines'] is None:
                    tmp['_lines'] = []
                tmp['_lines'].append(fig.axes[0].axvline(event.xdata, alpha=0.75, color='g'))

                if len(tmp['_lines']) == 2: # user specified both left and righ
                    pl.close(fig)

        # show image of the CCD, let user set approximate position of the spectrum
        #   to extract
        fig = auto_scale_imshow(d, figsize=(10,10))
        cursor = Cursor(fig.axes[0], useblit=False, horizOn=False,
                        color='red', linewidth=1)
        fig.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event, fig))
        fig.tight_layout()
        pl.show()

        if len(tmp['indices']) == 0:
            raise ValueError("Did you close the plot without selecting a "
                             "column or columns to extract?")

        elif len(tmp['indices']) == 1: # center line
            spectrum_idx = tmp['indices'][0]
            row_slc = slice(None, None)

        elif len(tmp['indices']) == 2: # left and right
            spectrum_idx = np.mean(tmp['indices'])
            row_slc = slice(int(np.min(tmp['indices'])), int(np.max(tmp['indices'])))

        else:
            assert True, "should never get here"

    else:
        _idx = np.linspace(0, d.shape[0]-1, min(32,d.shape[0])).astype(int)
        spectrum_idx = np.median(d[_idx].argmax(axis=-1))
        row_slc = slice(None, None)

    # pixels down dispersion axis
    dispersion_px = np.arange(d.shape[0], dtype=float)
    slit_px = np.arange(d.shape[1], dtype=float)

    trace_bin_idx = np.linspace(0, d.shape[0], ntracebins).astype(int)
    flux = np.zeros_like(dispersion_px)
    for i1,i2 in zip(trace_bin_idx[:-1],trace_bin_idx[1:]):
        row = np.median(d[i1:i2,row_slc], axis=0)

        if row_slc.start is None:
            i0 = 0
        else:
            i0 = row_slc.start

        p0 = (row[int(round(spectrum_idx - i0))],
              spectrum_idx - i0,
              2., # initial spread
              np.median(row))
        try:
            p,_ = so.curve_fit(gaussian_constant, slit_px[row_slc], row, p0=p0)
        except RuntimeError:
            logger.debug("Failed to fit Gaussian.")
            continue

        if p[0] < 0.:
            continue

        left = int(round(p[1] - 3*p[2])) + i0
        right = int(round(p[1] + 3*p[2])) + i0
        flux[i1:i2] = d[i1:i2,left:right].sum(axis=1)

    if wavelengthfile is None:
        x = dispersion_px

    else:
        wv = np.genfromtxt(wavelengthfile, names=True)
        # pix2wvl = si.interp1d(wv['pixel'], wv['wavelength'], kind='cubic')
        pix2wvl = si.InterpolatedUnivariateSpline(wv['pixel'], wv['wavelength'], k=3)
        x = pix2wvl(dispersion_px)

    fig,ax = pl.subplots(1,1)
    ax.plot(x, flux, marker=None, drawstyle='steps')
    hdr = fits.getheader(data_path)
    fig.suptitle("{0} ({1:.0f} s)".format(hdr['OBJECT'],hdr['EXPTIME']), fontsize=20)
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
    parser.add_argument("-i", "--interactive", action="store_true", dest="interactive",
                        default=False, help="Interactive mode!")

    parser.add_argument("--data", dest="datafile", required=True,
                        type=str, help="Data frame.")
    parser.add_argument("--wvln", dest="wvlnfile", required=True,
                        type=str, help="Wavelength pixel map file.")
    parser.add_argument("--path", dest="path", required=True,
                        type=str, help="Path to flats and bias files.")
    parser.add_argument("--oscan-idx", dest="overscan", required=True,
                        type=int, help="Index of the overscan region.")
    parser.add_argument("--filter-kernel-size", dest="ksize", default=1,
                        type=int, help="Size of the median filter kernel (in pixels).")
    parser.add_argument("--ntracebins", dest="ntracebins", default=32,
                        type=int, help="Number of bins to use for finding the trace of the "
                                       "spectrum along the dispersion axis.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(path=args.path, datafile=args.datafile, oscan_idx=args.overscan, ntracebins=args.ntracebins,
         interactive=args.interactive, median_filter_kernel=args.ksize,
         wavelengthfile=args.wvlnfile)
