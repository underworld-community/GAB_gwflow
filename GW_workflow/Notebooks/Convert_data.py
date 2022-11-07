#!/usr/bin/env python
# coding: utf-8
# %%

# %%


def transform_coordinates(x, y, epsg_in, epsg_out):
    """
    Transform between any coordinate system.
    **Requires `pyproj`** - install using pip.
    Args:
        x : float / 1D array
            x coordinates (may be in degrees or metres/eastings)
        y : float / 1D array
            y coordinates (may be in degrees or metres/northings)
        epsg_in : int
            CRS of x and y coordinates
        epsg_out : int
            CRS of output
    Returns:
        x_out : float / list of floats
            x coordinates projected in `epsg_out`
        y_out : float / list of floats
            y coordinates projected in `epsg_out`
    """
    import pyproj

    proj_in = pyproj.CRS("EPSG:" + str(epsg_in))
    proj_out = pyproj.CRS("EPSG:" + str(epsg_out))
    transformer = pyproj.Transformer.from_crs(proj_in, proj_out, always_xy=True)
    return transformer.transform(x,y)

