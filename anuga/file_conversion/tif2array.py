import numpy as np


def tif2array(filename, verbose=False,):

    import rasterio

    with rasterio.open(filename) as raster:
        ncols = raster.width
        nrows = raster.height
        transform = raster.transform
        x_origin = transform.c
        x_res = transform.a
        y_origin = transform.f
        y_res = transform.e
        NODATA_value = raster.nodata
        Z = raster.read(1)

    if NODATA_value is not None:
        Z = np.where(Z == NODATA_value, np.nan, Z)

    if y_res < 0:
        x = np.linspace(x_origin, x_origin + (ncols - 1) * x_res, ncols)
        y = np.linspace(y_origin + (nrows - 1) * y_res, y_origin, nrows)
        Z = np.flip(Z, axis=0)
        Z = Z.transpose()
    elif y_res >= 0:
        x = np.linspace(x_origin, x_origin + (ncols - 1) * x_res, ncols)
        y = np.linspace(y_origin, y_origin + (nrows - 1) * y_res, nrows)
        Z = Z.transpose()


    return x, y, Z


