import os

import geopandas as gpd


def create_amazon_boundary() -> None:
    """
    Create a GeoJSON file of the Amazon boundary.
    """
    print("Creating Amazon boundary...")
    shp_path = "data/raw/Limites2024/LimBiogeografico.shp"
    if not os.path.exists(shp_path):
        raise FileNotFoundError(
            f"Shapefile not found at {shp_path}. Please read the Appendix of the paper for more details."
        )
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS information!")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    out_path = "data/amazon_boundary.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"Amazon boundary saved to {out_path}")


if __name__ == "__main__":
    create_amazon_boundary()
