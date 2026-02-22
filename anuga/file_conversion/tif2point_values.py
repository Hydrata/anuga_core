__author__ = 'Allen Zhi Li'
__date__= '2020/06/08'

# Adapted by Stephen Roberts 2023

from pprint import pprint

def tif2point_values(filename, zone=None, south=True, points=None, verbose=False):

    import numpy as np
    import rasterio
    from pyproj import CRS

    with rasterio.open(filename) as raster:
        ncols = raster.width
        nrows = raster.height
        tif_epsg = str(raster.crs.to_epsg())
        NODATA_value = raster.nodata
        Z = raster.read(1)
        affine_transform = raster.transform

    # treat nan with 0 for now
    if NODATA_value is not None:
        Z = np.where(Z == NODATA_value, 0, Z)
    maxRows, maxCols = Z.shape

    # CRS for input points assumed UTM defined by zone and whether south or not
    points_utm = CRS.from_dict({'proj': 'utm', 'zone': zone, 'south': south})

    if tif_epsg == '4326':
        # tif file is lat long projection ie 'EPSG:4326'
        tif_georeference = CRS.from_epsg(4326)

        from pyproj import Transformer
        transformer = Transformer.from_crs(points_utm, tif_georeference)
        points_lat, points_lon = transformer.transform(points[:,0], points[:,1])

        ilocs = np.array(~affine_transform * (points_lon, points_lat))

    elif (tif_epsg == str(32600 + int(zone))) and not south:
        # no need for transformation
        ilocs = np.array(~affine_transform * (points[:,0], points[:,1]))

    elif (tif_epsg == str(32700 + int(zone))) and south:
        # no need for transformation
        ilocs = np.array(~affine_transform * (points[:,0], points[:,1]))

    elif (tif_epsg == str(7800 + int(zone)))  and south:
        # no need for transformation
        ilocs = np.array(~affine_transform * (points[:,0], points[:,1]))

    else:
        msg = 'zone and hemisphere of tif not the same as zone and hemisphere of points'
        raise Exception(msg)

    icols = ilocs[0,:].astype(int); irows = ilocs[1,:].astype(int)

    if (icols<maxCols).all() and (irows<maxRows).all():
        return Z[irows, icols]
    elif (icols-3<maxCols).all() and (irows<maxRows).all():
        mask = (icols>=maxCols)
        icols[mask] = maxCols-1
        return Z[irows,icols]
    elif (icols<maxCols).all() and (irows-3<maxRows).all():
        mask = (irows>=maxRows)
        irows[mask] = maxRows-1
        return Z[irows,icols]
    else:
        msg = 'points outside the extent of the source tif file, please crop tif file with a larger extent'
        raise ValueError(msg)
