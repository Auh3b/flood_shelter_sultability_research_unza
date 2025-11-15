import copy

import geopandas as gpd
import h3
import shapely as sh


def geomToH3shape(geom):
    return h3.geo_to_h3shape(geom)


def geomToCells(geom, res):
    h3s = geomToH3shape(geom)
    return rawFeatureToCells(h3s, res)


def cellsToPolygon(cells: list[str]):
    shapes = [h3.cells_to_geo([cell]) for cell in cells]
    polys = [sh.polygons(shape['coordinates']) for shape in shapes]
    return sh.multipolygons(polys)


def rawFeatureToCells(h3shape, res):
    return h3.h3shape_to_cells(h3shape, res)


def addCellsToGDF(gdf: gpd.GeoDataFrame, res: int | list[int]):
    output_gdf = copy.deepcopy(gdf)
    if (type(res) == int):
        output_gdf['cells'] = output_gdf.apply(
            lambda x: geomToCells(x.geometry, res), axis=1)
        output_gdf['cell_features'] = output_gdf.apply(
            lambda x: cellsToPolygon(x['cells']), axis=1)
    if (type(res) == list):
        for r in res:
            name = f'cells_{r}'
            output_gdf[name] = output_gdf.apply(
                lambda x: geomToCells(x.geometry, res), axis=1)
            output_gdf[f'cell_{r}_features'] = output_gdf.apply(
                lambda x: cellsToPolygon(x[name]), axis=1)

    return output_gdf
