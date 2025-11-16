from osgeo import gdal
import numpy as np


def raster_calculate(expression):
    return


def raster_to_vector(raster):
    return


def reclassify_by_value(raster):
    return

def uniqueValuesReport(path:str, name:str):
    output = None
    with gdal.Open(path) as ds:
        arr  = ds.GetRasterBand(1).GetBandAsArray()
        values, counts = np.unique(arr, return_counts=True)
        output = {x: y for x, y in zip(values, counts)}
    
    output['name'] = name
    return output
