import numpy as np
from osgeo import gdal, gdal_array


def raster_calculate(expression):
    return


def raster_to_vector(raster):
    return


def reclassify_by_value(raster):
    return


def getRasterArray(path: str | gdal.Dataset):
    arr = None
    if (isinstance(path, gdal.Dataset)):
        arr = path.GetRasterBand(1).ReadAsArray()
    elif (isinstance(path, str)):
        with gdal.Open(path) as ds:
            arr = ds.GetRasterBand(1).ReadAsArray()
    else:
        raise "Invalid input"

    return arr


def uniqueValuesReport(path: str | gdal.Dataset | np.ndarray, name: str):
    arr = None
    output = None

    if (isinstance(path, np.ndarray)):
        arr = path

    elif (isinstance(path, str)):
        with gdal.Open(path) as ds:
            arr = ds.GetRasterBand(1).ReadAsArray()

    elif (isinstance(path, gdal.Dataset)):
        arr = path.GetRasterBand(1).ReadAsArray()

    else:
        raise "Invalid input data"

    values, counts = np.unique(arr, return_counts=True)
    output = {x: y for x, y in zip(values, counts)}
    output['name'] = name
    return output
