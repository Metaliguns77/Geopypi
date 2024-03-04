"""Main module."""

import ipyleaflet

class Map(ipyleaflet.Map):
    def __init__(self, center= [20,0], **kwargs):
        super().__init__(center=center, **kwargs)
        self.add_control(ipyleaflet.LayersControl())
