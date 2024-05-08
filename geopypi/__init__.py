"""Top-level package for Geopypi."""

__author__ = """KH Shakibul Islam"""
__email__ = "kshakibu@vols.utk.edu"
__version__ = "0.0.7"


from .geopypi import Map
import os
from .report import Report

def view_vector(
    vector,
    zoom_to_layer=True,
    pickable=True,
    color_column=None,
    color_scheme="Quantiles",
    color_map=None,
    color_k=5,
    color_args={},
    open_args={},
    map_args={},
    **kwargs,
):
    """Visualize a vector dataset on the map.

    Args:
            vector (Union[str, GeoDataFrame]): The file path or URL to the vector data, or a GeoDataFrame.
            zoom_to_layer (bool, optional): Flag to zoom to the added layer. Defaults to True.
            pickable (bool, optional): Flag to enable picking on the added layer. Defaults to True.
            color_column (Optional[str], optional): The column to be used for color encoding. Defaults to None.
            color_map (Optional[Union[str, Dict]], optional): The color map to use for color encoding. It can be a string or a dictionary. Defaults to None.
            color_scheme (Optional[str], optional): The color scheme to use for color encoding. Defaults to "Quantiles".
                Name of a choropleth classification scheme (requires mapclassify).
                A mapclassify.MapClassifier object will be used
                under the hood. Supported are all schemes provided by mapclassify (e.g.
                'BoxPlot', 'EqualInterval', 'FisherJenks', 'FisherJenksSampled',
                'HeadTailBreaks', 'JenksCaspall', 'JenksCaspallForced',
                'JenksCaspallSampled', 'MaxP', 'MaximumBreaks',
                'NaturalBreaks', 'Quantiles', 'Percentiles', 'StdMean',
                'UserDefined'). Arguments can be passed in classification_kwds.
            color_k (Optional[int], optional): The number of classes to use for color encoding. Defaults to 5.
            color_args (dict, optional): Additional keyword arguments that will be passed to assign_continuous_colors(). Defaults to {}.
            open_args (dict, optional): Additional keyword arguments that will be passed to geopandas.read_file(). Defaults to {}.
            map_args (dict, optional): Additional keyword arguments that will be passed to lonboard.Map. Defaults to {}.
            **kwargs: Additional keyword arguments that will be passed to lonboard.Layer.from_geopandas()

    Returns:
        lonboard.Map: A lonboard Map object.
    """
    from .deckgl import Map

    m = Map(**map_args)
    m.add_vector(
        vector,
        zoom_to_layer,
        pickable,
        color_column,
        color_scheme,
        color_map,
        color_k,
        color_args,
        open_args,
        **kwargs,
    )
    return m

