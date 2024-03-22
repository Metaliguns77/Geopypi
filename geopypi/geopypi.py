"""Main module."""

import ipyleaflet
from ipyleaflet import basemaps

class Map(ipyleaflet.Map):

    def __init__(self, center=[20,0], zoom= 2, **kwargs):
        super().__init__(center=center, zoom=zoom, **kwargs)
        self.add_control(ipyleaflet.LayersControl())

def add_tile_layer(map, url, name, active=False, **kwargs):
    layer = ipyleaflet.TileLayer(url=url, name=name, active=active, **kwargs)
    map.add(layer)

def add_basemap(self, name):

    if isinstance(name, str):
        basemap = eval(f"basmeaps.{name}")
        self 