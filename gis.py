# functions for griddata & Geographic Information System operations

import rasterio
import numpy as np


def load_raster(path):
    """
    Laadt een ASC/GeoTIFF rasterbestand.
    Geeft (array, transform) terug.
    """
    with rasterio.open(path) as src:
        return src.read(1), src.transform


def load_shapefile(path):
    """
    Laadt een shapefile met nederzettingsgegevens via geopandas.
    Correspondeert met NetLogo's gis:load-dataset + gis:feature-list-of.
    Geeft een lijst van dicts terug met x, y en properties per site.
    """
    try:
        import geopandas as gpd

        gdf = gpd.read_file(path)

        sites = []
        for _, row in gdf.iterrows():
            geom = row.geometry

            # Punt-geometrie: x en y direct uitlezen
            if geom.geom_type == "Point":
                x, y = geom.x, geom.y
            elif geom.geom_type == "MultiPoint":
                x, y = geom.geoms[0].x, geom.geoms[0].y
            else:
                # Fallback: centroïde gebruiken
                x, y = geom.centroid.x, geom.centroid.y

            # Kolomnamen case-insensitief zoeken
            cols = {c.lower(): c for c in gdf.columns}

            def get_prop(key, default):
                col = cols.get(key.lower())
                return row[col] if col and row[col] is not None else default

            site = {
                "x":     x,
                "y":     y,
                "Site":  get_prop("Site",  f"site_{len(sites)}"),
                "Type":  get_prop("Type",  "village"),
                "Start": get_prop("Start", "IA"),
            }
            sites.append(site)

        print(f"Shapefile geladen: {len(sites)} sites gevonden.")
        return sites

    except ImportError:
        raise ImportError("geopandas is vereist voor shapefile-loading: pip install geopandas")
    except Exception as e:
        raise RuntimeError(f"Kan shapefile niet laden: {e}")