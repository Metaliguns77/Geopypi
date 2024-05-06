"""The common module contains common functions and classes used by the other modules.
"""
import os
import zipfile

from typing import Union, List, Dict, Optional, Tuple

def hello_world():
    """Prints "Hello World!" to the console.
    """
    print("Hello World!")

def random_number():
    """Returns a random number between 1 and 100.
    """
    import random
    return random.randint(1, 100)

def github_raw_url(url):
    """Get the raw URL for a GitHub file.

    Args:
        url (str): The GitHub URL.
    Returns:
        str: The raw URL.
    """
    if isinstance(url, str) and url.startswith("https://github.com/") and "blob" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace(
            "blob/", ""
        )
    return url
def download_file(
    url=None,
    output=None,
    quiet=False,
    proxy=None,
    speed=None,
    use_cookies=True,
    verify=True,
    id=None,
    fuzzy=False,
    resume=False,
    unzip=True,
    overwrite=False,
    subfolder=False,
):
    """Download a file from URL, including Google Drive shared URL.

    Args:
        url (str, optional): Google Drive URL is also supported. Defaults to None.
        output (str, optional): Output filename. Default is basename of URL.
        quiet (bool, optional): Suppress terminal output. Default is False.
        proxy (str, optional): Proxy. Defaults to None.
        speed (float, optional): Download byte size per second (e.g., 256KB/s = 256 * 1024). Defaults to None.
        use_cookies (bool, optional): Flag to use cookies. Defaults to True.
        verify (bool | str, optional): Either a bool, in which case it controls whether the server's TLS certificate is verified, or a string,
            in which case it must be a path to a CA bundle to use. Default is True.. Defaults to True.
        id (str, optional): Google Drive's file ID. Defaults to None.
        fuzzy (bool, optional): Fuzzy extraction of Google Drive's file Id. Defaults to False.
        resume (bool, optional): Resume the download from existing tmp file if possible. Defaults to False.
        unzip (bool, optional): Unzip the file. Defaults to True.
        overwrite (bool, optional): Overwrite the file if it already exists. Defaults to False.
        subfolder (bool, optional): Create a subfolder with the same name as the file. Defaults to False.

    Returns:
        str: The output file path.
    """
    try:
        import gdown
    except ImportError:
        print(
            "The gdown package is required for this function. Use `pip install gdown` to install it."
        )
        return

    if output is None:
        if isinstance(url, str) and url.startswith("http"):
            output = os.path.basename(url)

    out_dir = os.path.abspath(os.path.dirname(output))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if isinstance(url, str):
        if os.path.exists(os.path.abspath(output)) and (not overwrite):
            print(
                f"{output} already exists. Skip downloading. Set overwrite=True to overwrite."
            )
            return os.path.abspath(output)
        else:
            url = github_raw_url(url)

    if "https://drive.google.com/file/d/" in url:
        fuzzy = True

    output = gdown.download(
        url, output, quiet, proxy, speed, use_cookies, verify, id, fuzzy, resume
    )

    if unzip:
        if output.endswith(".zip"):
            with zipfile.ZipFile(output, "r") as zip_ref:
                if not quiet:
                    print("Extracting files...")
                if subfolder:
                    basename = os.path.splitext(os.path.basename(output))[0]

                    output = os.path.join(out_dir, basename)
                    if not os.path.exists(output):
                        os.makedirs(output)
                    zip_ref.extractall(output)
                else:
                    zip_ref.extractall(os.path.dirname(output))
        elif output.endswith(".tar.gz") or output.endswith(".tar"):
            if output.endswith(".tar.gz"):
                mode = "r:gz"
            else:
                mode = "r"

            with tarfile.open(output, mode) as tar_ref:
                if not quiet:
                    print("Extracting files...")
                if subfolder:
                    basename = os.path.splitext(os.path.basename(output))[0]
                    output = os.path.join(out_dir, basename)
                    if not os.path.exists(output):
                        os.makedirs(output)
                    tar_ref.extractall(output)
                else:
                    tar_ref.extractall(os.path.dirname(output))

    return os.path.abspath(output)


def download_files(
    urls,
    out_dir=None,
    filenames=None,
    quiet=False,
    proxy=None,
    speed=None,
    use_cookies=True,
    verify=True,
    id=None,
    fuzzy=False,
    resume=False,
    unzip=True,
    overwrite=False,
    subfolder=False,
    multi_part=False,
):
    """Download files from URLs, including Google Drive shared URL.

    Args:
        urls (list): The list of urls to download. Google Drive URL is also supported.
        out_dir (str, optional): The output directory. Defaults to None.
        filenames (list, optional): Output filename. Default is basename of URL.
        quiet (bool, optional): Suppress terminal output. Default is False.
        proxy (str, optional): Proxy. Defaults to None.
        speed (float, optional): Download byte size per second (e.g., 256KB/s = 256 * 1024). Defaults to None.
        use_cookies (bool, optional): Flag to use cookies. Defaults to True.
        verify (bool | str, optional): Either a bool, in which case it controls whether the server's TLS certificate is verified, or a string, in which case it must be a path to a CA bundle to use. Default is True.. Defaults to True.
        id (str, optional): Google Drive's file ID. Defaults to None.
        fuzzy (bool, optional): Fuzzy extraction of Google Drive's file Id. Defaults to False.
        resume (bool, optional): Resume the download from existing tmp file if possible. Defaults to False.
        unzip (bool, optional): Unzip the file. Defaults to True.
        overwrite (bool, optional): Overwrite the file if it already exists. Defaults to False.
        subfolder (bool, optional): Create a subfolder with the same name as the file. Defaults to False.
        multi_part (bool, optional): If the file is a multi-part file. Defaults to False.

    Examples:

        files = ["sam_hq_vit_tiny.zip", "sam_hq_vit_tiny.z01", "sam_hq_vit_tiny.z02", "sam_hq_vit_tiny.z03"]
        base_url = "https://github.com/opengeos/datasets/releases/download/models/"
        urls = [base_url + f for f in files]
        leafmap.download_files(urls, out_dir="models", multi_part=True)
    """

    if out_dir is None:
        out_dir = os.getcwd()

    if filenames is None:
        filenames = [None] * len(urls)

    filepaths = []
    for url, output in zip(urls, filenames):
        if output is None:
            filename = os.path.join(out_dir, os.path.basename(url))
        else:
            filename = os.path.join(out_dir, output)

        filepaths.append(filename)
        if multi_part:
            unzip = False

        download_file(
            url,
            filename,
            quiet,
            proxy,
            speed,
            use_cookies,
            verify,
            id,
            fuzzy,
            resume,
            unzip,
            overwrite,
            subfolder,
        )

    if multi_part:
        archive = os.path.splitext(filename)[0] + ".zip"
        out_dir = os.path.dirname(filename)
        extract_archive(archive, out_dir)

        for file in filepaths:
            os.remove(file)

def read_parquet(
    source: str,
    geometry: Optional[str] = None,
    columns: Optional[Union[str, list]] = None,
    exclude: Optional[Union[str, list]] = None,
    db: Optional[str] = None,
    table_name: Optional[str] = None,
    sql: Optional[str] = None,
    limit: Optional[int] = None,
    src_crs: Optional[str] = None,
    dst_crs: Optional[str] = None,
    return_type: str = "gdf",
    **kwargs,
):
    """
    Read Parquet data from a source and return a GeoDataFrame or DataFrame.

    Args:
        source (str): The path to the Parquet file or directory containing Parquet files.
        geometry (str, optional): The name of the geometry column. Defaults to None.
        columns (str or list, optional): The columns to select. Defaults to None (select all columns).
        exclude (str or list, optional): The columns to exclude from the selection. Defaults to None.
        db (str, optional): The DuckDB database path or alias. Defaults to None.
        table_name (str, optional): The name of the table in the DuckDB database. Defaults to None.
        sql (str, optional): The SQL query to execute. Defaults to None.
        limit (int, optional): The maximum number of rows to return. Defaults to None (return all rows).
        src_crs (str, optional): The source CRS (Coordinate Reference System) of the geometries. Defaults to None.
        dst_crs (str, optional): The target CRS to reproject the geometries. Defaults to None.
        return_type (str, optional): The type of object to return:
            - 'gdf': GeoDataFrame (default)
            - 'df': DataFrame
            - 'numpy': NumPy array
            - 'arrow': Arrow Table
            - 'polars': Polars DataFrame
        **kwargs: Additional keyword arguments that are passed to the DuckDB connection.

    Returns:
        Union[gpd.GeoDataFrame, pd.DataFrame, np.ndarray]: The loaded data.

    Raises:
        ValueError: If the columns or exclude arguments are not of the correct type.

    """
    import duckdb

    if isinstance(db, str):
        con = duckdb.connect(db)
    else:
        con = duckdb.connect()

    con.install_extension("httpfs")
    con.load_extension("httpfs")

    con.install_extension("spatial")
    con.load_extension("spatial")

    if columns is None:
        columns = "*"
    elif isinstance(columns, list):
        columns = ", ".join(columns)
    elif not isinstance(columns, str):
        raise ValueError("columns must be a list or a string.")

    if exclude is not None:
        if isinstance(exclude, list):
            exclude = ", ".join(exclude)
        elif not isinstance(exclude, str):
            raise ValueError("exclude_columns must be a list or a string.")
        columns = f"{columns} EXCLUDE {exclude}"

    if return_type in ["df", "numpy", "arrow", "polars"]:
        if sql is None:
            sql = f"SELECT {columns} FROM '{source}'"
        if limit is not None:
            sql += f" LIMIT {limit}"

        if return_type == "df":
            result = con.sql(sql, **kwargs).df()
        elif return_type == "numpy":
            result = con.sql(sql, **kwargs).fetchnumpy()
        elif return_type == "arrow":
            result = con.sql(sql, **kwargs).arrow()
        elif return_type == "polars":
            result = con.sql(sql, **kwargs).pl()

        if table_name is not None:
            con.sql(f"CREATE OR REPLACE TABLE {table_name} AS FROM result", **kwargs)

    elif return_type == "gdf":
        if geometry is None:
            geometry = "geometry"
        if sql is None:
            # if src_crs is not None and dst_crs is not None:
            #     geom_sql = f"ST_AsText(ST_Transform(ST_GeomFromWKB({geometry}), '{src_crs}', '{dst_crs}', true)) AS {geometry}"
            # else:
            geom_sql = f"ST_AsText(ST_GeomFromWKB({geometry})) AS {geometry}"
            sql = f"SELECT {columns} EXCLUDE {geometry}, {geom_sql} FROM '{source}'"
        if limit is not None:
            sql += f" LIMIT {limit}"

        df = con.sql(sql, **kwargs).df()
        if table_name is not None:
            con.sql(f"CREATE OR REPLACE TABLE {table_name} AS FROM df", **kwargs)
        result = df_to_gdf(df, geometry=geometry, src_crs=src_crs, dst_crs=dst_crs)

    con.close()
    return result

def df_to_gdf(df, geometry="geometry", src_crs="EPSG:4326", dst_crs=None, **kwargs):
    """
    Converts a pandas DataFrame to a GeoPandas GeoDataFrame.

    Args:
        df (pandas.DataFrame): The pandas DataFrame to convert.
        geometry (str): The name of the geometry column in the DataFrame.
        src_crs (str): The coordinate reference system (CRS) of the GeoDataFrame. Default is "EPSG:4326".
        dst_crs (str): The target CRS of the GeoDataFrame. Default is None

    Returns:
        geopandas.GeoDataFrame: The converted GeoPandas GeoDataFrame.
    """
    import geopandas as gpd
    from shapely import wkt

    # Convert the geometry column to Shapely geometry objects
    df[geometry] = df[geometry].apply(lambda x: wkt.loads(x))

    # Convert the pandas DataFrame to a GeoPandas GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=src_crs, **kwargs)
    if dst_crs is not None and dst_crs != src_crs:
        gdf = gdf.to_crs(dst_crs)

    return gdf

def open_image_from_url(url: str):
    """Loads an image from the specified URL.

    Args:
        url (str): URL of the image.

    Returns:
        object: Image object.
    """
    from PIL import Image

    from io import BytesIO

    # from urllib.parse import urlparse

    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(e)

    def pillow_to_base64(image: Image.Image) -> str:
        """
        Convert a PIL image to a base64-encoded string.

        Parameters
        ----------
        image: PIL.Image.Image
            The image to be converted.

        Returns
        -------
        str
            The base64-encoded string.
        """
        in_mem_file = io.BytesIO()
        image.save(in_mem_file, format="JPEG", subsampling=0, quality=100)
        img_bytes = in_mem_file.getvalue()  # bytes
        image_str = base64.b64encode(img_bytes).decode("utf-8")
        base64_src = f"data:image/jpg;base64,{image_str}"
        return base64_src
    
    def read_image_as_pil(
        image: Union[Image.Image, str, np.ndarray], exif_fix: bool = False
    ):
        """
        Loads an image as PIL.Image.Image.
        Args:
            image : Can be image path or url (str), numpy image (np.ndarray) or PIL.Image
        """
        # https://stackoverflow.com/questions/56174099/how-to-load-images-larger-than-max-image-pixels-with-pil
        Image.MAX_IMAGE_PIXELS = None

        if isinstance(image, Image.Image):
            image_pil = image.convert("RGB")
        elif isinstance(image, str):
            # read image if str image path is provided
            try:
                image_pil = Image.open(
                    requests.get(image, stream=True).raw
                    if str(image).startswith("http")
                    else image
                ).convert("RGB")
                if exif_fix:
                    image_pil = exif_transpose(image_pil)
            except:  # handle large/tiff image reading
                try:
                    import skimage.io
                except ImportError:
                    raise ImportError(
                        "Please run 'pip install -U scikit-image imagecodecs' for large image handling."
                    )
                image_sk = skimage.io.imread(image).astype(np.uint8)
                if len(image_sk.shape) == 2:  # b&w
                    image_pil = Image.fromarray(image_sk, mode="1").convert("RGB")
                elif image_sk.shape[2] == 4:  # rgba
                    image_pil = Image.fromarray(image_sk, mode="RGBA").convert("RGB")
                elif image_sk.shape[2] == 3:  # rgb
                    image_pil = Image.fromarray(image_sk, mode="RGB")
                else:
                    raise TypeError(
                        f"image with shape: {image_sk.shape[3]} is not supported."
                    )
        elif isinstance(image, np.ndarray):
            if image.shape[0] < 5:  # image in CHW
                image = image[:, :, ::-1]
            image_pil = Image.fromarray(image).convert("RGB")
        else:
            raise TypeError("read image with 'pillow' using 'Image.open()'")

        return image_pil
    
def classify(
    data,
    column,
    cmap=None,
    colors=None,
    labels=None,
    scheme="Quantiles",
    k=5,
    legend_kwds=None,
    classification_kwds=None,
):
    """Classify a dataframe column using a variety of classification schemes.

    Args:
        data (str | pd.DataFrame | gpd.GeoDataFrame): The data to classify. It can be a filepath to a vector dataset, a pandas dataframe, or a geopandas geodataframe.
        column (str): The column to classify.
        cmap (str, optional): The name of a colormap recognized by matplotlib. Defaults to None.
        colors (list, optional): A list of colors to use for the classification. Defaults to None.
        labels (list, optional): A list of labels to use for the legend. Defaults to None.
        scheme (str, optional): Name of a choropleth classification scheme (requires mapclassify).
            Name of a choropleth classification scheme (requires mapclassify).
            A mapclassify.MapClassifier object will be used
            under the hood. Supported are all schemes provided by mapclassify (e.g.
            'BoxPlot', 'EqualInterval', 'FisherJenks', 'FisherJenksSampled',
            'HeadTailBreaks', 'JenksCaspall', 'JenksCaspallForced',
            'JenksCaspallSampled', 'MaxP', 'MaximumBreaks',
            'NaturalBreaks', 'Quantiles', 'Percentiles', 'StdMean',
            'UserDefined'). Arguments can be passed in classification_kwds.
        k (int, optional): Number of classes (ignored if scheme is None or if column is categorical). Default to 5.
        legend_kwds (dict, optional): Keyword arguments to pass to :func:`matplotlib.pyplot.legend` or `matplotlib.pyplot.colorbar`. Defaults to None.
            Keyword arguments to pass to :func:`matplotlib.pyplot.legend` or
            Additional accepted keywords when `scheme` is specified:
            fmt : string
                A formatting specification for the bin edges of the classes in the
                legend. For example, to have no decimals: ``{"fmt": "{:.0f}"}``.
            labels : list-like
                A list of legend labels to override the auto-generated labblels.
                Needs to have the same number of elements as the number of
                classes (`k`).
            interval : boolean (default False)
                An option to control brackets from mapclassify legend.
                If True, open/closed interval brackets are shown in the legend.
        classification_kwds (dict, optional): Keyword arguments to pass to mapclassify. Defaults to None.

    Returns:
        pd.DataFrame, dict: A pandas dataframe with the classification applied and a legend dictionary.
    """

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    try:
        import mapclassify
    except ImportError:
        raise ImportError(
            "mapclassify is required for this function. Install with `pip install mapclassify`."
        )

    if (
        isinstance(data, gpd.GeoDataFrame)
        or isinstance(data, pd.DataFrame)
        or isinstance(data, pd.Series)
    ):
        df = data
    else:
        try:
            df = gpd.read_file(data)
        except Exception:
            raise TypeError(
                "Data must be a GeoDataFrame or a path to a file that can be read by geopandas.read_file()."
            )

    if df.empty:
        warnings.warn(
            "The GeoDataFrame you are attempting to plot is "
            "empty. Nothing has been displayed.",
            UserWarning,
        )
        return

    columns = df.columns.values.tolist()
    if column not in columns:
        raise ValueError(
            f"{column} is not a column in the GeoDataFrame. It must be one of {columns}."
        )

    # Convert categorical data to numeric
    init_column = None
    value_list = None
    if np.issubdtype(df[column].dtype, np.object0):
        value_list = df[column].unique().tolist()
        value_list.sort()
        df["category"] = df[column].replace(value_list, range(0, len(value_list)))
        init_column = column
        column = "category"
        k = len(value_list)

    if legend_kwds is not None:
        legend_kwds = legend_kwds.copy()

    # To accept pd.Series and np.arrays as column
    if isinstance(column, (np.ndarray, pd.Series)):
        if column.shape[0] != df.shape[0]:
            raise ValueError(
                "The dataframe and given column have different number of rows."
            )
        else:
            values = column

            # Make sure index of a Series matches index of df
            if isinstance(values, pd.Series):
                values = values.reindex(df.index)
    else:
        values = df[column]

    values = df[column]
    nan_idx = np.asarray(pd.isna(values), dtype="bool")

    if cmap is None:
        cmap = "Blues"
    cmap = plt.cm.get_cmap(cmap, k)
    if colors is None:
        colors = [mpl.colors.rgb2hex(cmap(i))[1:] for i in range(cmap.N)]
        colors = ["#" + i for i in colors]
    elif isinstance(colors, list):
        colors = [check_color(i) for i in colors]
    elif isinstance(colors, str):
        colors = [check_color(colors)] * k

    allowed_schemes = [
        "BoxPlot",
        "EqualInterval",
        "FisherJenks",
        "FisherJenksSampled",
        "HeadTailBreaks",
        "JenksCaspall",
        "JenksCaspallForced",
        "JenksCaspallSampled",
        "MaxP",
        "MaximumBreaks",
        "NaturalBreaks",
        "Quantiles",
        "Percentiles",
        "StdMean",
        "UserDefined",
    ]

    if scheme.lower() not in [s.lower() for s in allowed_schemes]:
        raise ValueError(
            f"{scheme} is not a valid scheme. It must be one of {allowed_schemes}."
        )

    if classification_kwds is None:
        classification_kwds = {}
    if "k" not in classification_kwds:
        classification_kwds["k"] = k

    binning = mapclassify.classify(
        np.asarray(values[~nan_idx]), scheme, **classification_kwds
    )
    df["category"] = binning.yb
    df["color"] = [colors[i] for i in df["category"]]

    if legend_kwds is None:
        legend_kwds = {}

    if "interval" not in legend_kwds:
        legend_kwds["interval"] = True

    if "fmt" not in legend_kwds:
        if np.issubdtype(df[column].dtype, np.floating):
            legend_kwds["fmt"] = "{:.2f}"
        else:
            legend_kwds["fmt"] = "{:.0f}"

    if labels is None:
        # set categorical to True for creating the legend
        if legend_kwds is not None and "labels" in legend_kwds:
            if len(legend_kwds["labels"]) != binning.k:
                raise ValueError(
                    "Number of labels must match number of bins, "
                    "received {} labels for {} bins".format(
                        len(legend_kwds["labels"]), binning.k
                    )
                )
            else:
                labels = list(legend_kwds.pop("labels"))
        else:
            # fmt = "{:.2f}"
            if legend_kwds is not None and "fmt" in legend_kwds:
                fmt = legend_kwds.pop("fmt")

            labels = binning.get_legend_classes(fmt)
            if legend_kwds is not None:
                show_interval = legend_kwds.pop("interval", False)
            else:
                show_interval = False
            if not show_interval:
                labels = [c[1:-1] for c in labels]

        if init_column is not None:
            labels = value_list
    elif isinstance(labels, list):
        if len(labels) != len(colors):
            raise ValueError("The number of labels must match the number of colors.")
    else:
        raise ValueError("labels must be a list or None.")

    legend_dict = dict(zip(labels, colors))
    df["category"] = df["category"] + 1
    return df, legend_dict

def gdf_to_geojson(
    gdf, out_geojson=None, epsg=None, tuple_to_list=False, encoding="utf-8"
):
    """Converts a GeoDataFame to GeoJSON.

    Args:
        gdf (GeoDataFrame): A GeoPandas GeoDataFrame.
        out_geojson (str, optional): File path to he output GeoJSON. Defaults to None.
        epsg (str, optional): An EPSG string, e.g., "4326". Defaults to None.
        tuple_to_list (bool, optional): Whether to convert tuples to lists. Defaults to False.
        encoding (str, optional): The encoding to use for the GeoJSON. Defaults to "utf-8".

    Raises:
        TypeError: When the output file extension is incorrect.
        Exception: When the conversion fails.

    Returns:
        dict: When the out_json is None returns a dict.
    """
    check_package(name="geopandas", URL="https://geopandas.org")

    def listit(t):
        return list(map(listit, t)) if isinstance(t, (list, tuple)) else t

    try:
        if epsg is not None:
            if gdf.crs is not None and gdf.crs.to_epsg() != epsg:
                gdf = gdf.to_crs(epsg=epsg)
        geojson = gdf.__geo_interface__

        if tuple_to_list:
            for feature in geojson["features"]:
                feature["geometry"]["coordinates"] = listit(
                    feature["geometry"]["coordinates"]
                )

        if out_geojson is None:
            return geojson
        else:
            ext = os.path.splitext(out_geojson)[1]
            if ext.lower() not in [".json", ".geojson"]:
                raise TypeError(
                    "The output file extension must be either .json or .geojson"
                )
            out_dir = os.path.dirname(out_geojson)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            gdf.to_file(out_geojson, driver="GeoJSON", encoding=encoding)
    except Exception as e:
        raise Exception(e)
    
def check_package(name: str, URL: Optional[str] = ""):
    try:
        __import__(name.lower())
    except Exception:
        raise ImportError(
            f"{name} is not installed. Please install it before proceeding. {URL}"
        )

def get_geometry_type(in_geojson: Union[str, Dict]) -> str:
    """Get the geometry type of a GeoJSON file.

    Args:
        in_geojson (str | dict): The path to the GeoJSON file or a GeoJSON dictionary.

    Returns:
        str: The geometry type. Can be one of "Point", "LineString", "Polygon", "MultiPoint",
            "MultiLineString", "MultiPolygon", "GeometryCollection", or "Unknown".
    """

    import geojson

    try:
        if isinstance(in_geojson, str):  # If input is a file path
            with open(in_geojson, "r") as geojson_file:
                geojson_data = geojson.load(geojson_file)
        elif isinstance(in_geojson, dict):  # If input is a GeoJSON dictionary
            geojson_data = in_geojson
        else:
            return "Invalid input type. Expected file path or dictionary."

        if "type" in geojson_data:
            if geojson_data["type"] == "FeatureCollection":
                features = geojson_data.get("features", [])
                if features:
                    first_feature = features[0]
                    geometry = first_feature.get("geometry")
                    if geometry and "type" in geometry:
                        return geometry["type"]
                    else:
                        return "No geometry type found in the first feature."
                else:
                    return "No features found in the FeatureCollection."
            elif geojson_data["type"] == "Feature":
                geometry = geojson_data.get("geometry")
                if geometry and "type" in geometry:
                    return geometry["type"]
                else:
                    return "No geometry type found in the Feature."
            else:
                return "Unsupported GeoJSON type."
        else:
            return "No 'type' field found in the GeoJSON data."
    except Exception as e:
        raise e
