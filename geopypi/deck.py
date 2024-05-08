class Map(pdk.Deck):
    """The Map class inherits pydeck.Deck.

    Returns:
        object: pydeck.Deck object.
    """

    def __init__(self, center=(20, 0), zoom=1.2, **kwargs):
        """Initialize a Map object.

        Args:
            center (tuple, optional): Center of the map in the format of (lat, lon). Defaults to (20, 0).
            zoom (int, optional): The map zoom level. Defaults to 1.2.
        """
        if "initial_view_state" not in kwargs:
            kwargs["initial_view_state"] = pdk.ViewState(
                latitude=center[0], longitude=center[1], zoom=zoom
            )

        if "map_style" not in kwargs:
            kwargs["map_style"] = "light"

        super().__init__(**kwargs)

    def add_layer(self, layer, layer_name: Optional[str] = None, **kwargs):
        """Add a layer to the map.

        Args:
            layer (pydeck.Layer): A pydeck Layer object.
        """

        try:
            if isinstance(layer, str) and layer.startswith("http"):
                pdk.settings.custom_libraries = [
                    {
                        "libraryName": "MyTileLayerLibrary",
                        "resourceUri": "https://cdn.jsdelivr.net/gh/giswqs/pydeck_myTileLayer@master/dist/bundle.js",
                    }
                ]
                layer = pdk.Layer("MyTileLayer", layer, id=layer_name)

            self.layers.append(layer)
        except Exception as e:
            raise Exception(e)
    def add_gdf(
        self,
        gdf,
        layer_name: Optional[str] = None,
        random_color_column: Optional[str] = None,
        **kwargs
    ):
        """Adds a GeoPandas GeoDataFrame to the map.

        Args:
            gdf (GeoPandas.GeoDataFrame): The GeoPandas GeoDataFrame to add to the map.
            layer_name (str, optional): The layer name to be used. Defaults to None.
            random_color_column (str, optional): The column name to use for random color. Defaults to None.

        Raises:
            TypeError: gdf must be a GeoPandas GeoDataFrame.
        """

        try:
            import geopandas as gpd

            if not isinstance(gdf, gpd.GeoDataFrame):
                raise TypeError("gdf must be a GeoPandas GeoDataFrame.")

            if layer_name is None:
                layer_name = "layer_" + random_string(3)

            if "pickable" not in kwargs:
                kwargs["pickable"] = True
            if "opacity" not in kwargs:
                kwargs["opacity"] = 0.5
            if "stroked" not in kwargs:
                kwargs["stroked"] = True
            if "filled" not in kwargs:
                kwargs["filled"] = True
            if "extruded" not in kwargs:
                kwargs["extruded"] = False
            if "wireframe" not in kwargs:
                kwargs["wireframe"] = True
            if "get_line_color" not in kwargs:
                kwargs["get_line_color"] = [0, 0, 0]
            if "get_line_width" not in kwargs:
                kwargs["get_line_width"] = 2
            if "line_width_min_pixels" not in kwargs:
                kwargs["line_width_min_pixels"] = 1

            if random_color_column is not None:
                if random_color_column not in gdf.columns.values.tolist():
                    raise ValueError(
                        "The random_color_column provided does not exist in the vector file."
                    )
                color_lookup = pdk.data_utils.assign_random_colors(
                    gdf[random_color_column]
                )
                gdf["color"] = gdf.apply(
                    lambda row: color_lookup.get(row[random_color_column]), axis=1
                )
                kwargs["get_fill_color"] = "color"

            layer = pdk.Layer(
                "GeoJsonLayer",
                gdf,
                id=layer_name,
                **kwargs,
            )
            self.add_layer(layer)

        except Exception as e:
            raise Exception(e)
    