"""Test titiler.pgstac Mosaic endpoints."""

import io
from datetime import datetime
from unittest.mock import patch

import pytest
import rasterio
from rasterio.crs import CRS

from .conftest import mock_rasterio_open, parse_img


@pytest.fixture
def search_no_bbox(app):
    """Search Query without BBOX."""
    query = {"collections": ["noaa-emergency-response"]}
    response = app.post("/searches/register", json=query)
    assert response.status_code == 200
    resp = response.json()
    assert resp["links"]
    assert [link["rel"] for link in resp["links"]] == [
        "metadata",
        "tilejson",
        "map",
        "wmts",
    ]
    return resp["id"]


@pytest.fixture
def search_bbox(app):
    """Search Query with BBOX."""
    query = {
        "collections": ["noaa-emergency-response"],
        "bbox": [-85.535, 36.137, -85.465, 36.179],
    }
    response = app.post("/searches/register", json=query)
    assert response.status_code == 200

    resp = response.json()
    assert resp["links"]
    assert [link["rel"] for link in resp["links"]] == [
        "metadata",
        "tilejson",
        "map",
        "wmts",
    ]
    return resp["id"]


def test_info(app, search_no_bbox):
    """Should return metadata about a search query."""
    response = app.get(f"/searches/{search_no_bbox}/info")
    assert response.status_code == 200
    resp = response.json()
    assert resp["search"]
    assert resp["links"]
    search = resp["search"]
    assert search["search"] == {
        "collections": ["noaa-emergency-response"],
        "filter-lang": "cql2-json",
    }
    assert search["metadata"] == {"type": "mosaic"}

    response = app.get("/searches/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/info")
    assert response.status_code == 404
    resp = response.json()
    assert resp["detail"] == "SearchId `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` not found"


def test_assets_for_point(app, search_no_bbox, search_bbox):
    """Get assets for a Point."""
    response = app.get(f"/searches/{search_no_bbox}/point/-85.6358,36.1624/assets")
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 1
    assert list(resp[0]) == ["id", "bbox", "assets", "collection"]
    assert resp[0]["id"] == "20200307aC0853900w361030"

    # make sure we can find assets when having both bbox and geometry
    response = app.get(f"/searches/{search_bbox}/point/-85.5,36.1624/assets")
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 2

    # with coord-crs
    response = app.get(
        f"/searches/{search_bbox}/point/-9517816.46282489,4322990.432036275/assets",
        params={"coord_crs": "epsg:3857"},
    )
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 2

    # no assets found outside the mosaic bbox
    response = app.get(f"/searches/{search_bbox}/point/-85.6358,36.1624/assets")
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 0

    # searchId not found
    response = app.get(
        "/searches/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/point/-85.5,36.1624/assets"
    )
    assert response.status_code == 404
    resp = response.json()
    assert resp["detail"] == "SearchId `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` not found"


def test_assets_for_tile(app, search_no_bbox, search_bbox):
    """Get assets for a Tile."""
    response = app.get(
        f"/searches/{search_no_bbox}/tiles/WebMercatorQuad/15/8589/12849/assets"
    )
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 1
    assert list(resp[0]) == ["id", "bbox", "assets", "collection"]
    assert resp[0]["id"] == "20200307aC0853900w361030"

    # make sure we can find assets when having both bbox and geometry
    response = app.get(
        f"/searches/{search_bbox}/tiles/WebMercatorQuad/15/8601/12849/assets"
    )
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 2

    # With WGS1994Quad TMS
    response = app.get(f"/searches/{search_bbox}/tiles/WGS1984Quad/14/8601/4901/assets")
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 4

    # no assets found outside the query bbox
    response = app.get(
        f"/searches/{search_bbox}/tiles/WebMercatorQuad/15/8589/12849/assets"
    )
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 0

    # searchId not found
    response = app.get(
        "/searches/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/tiles/WebMercatorQuad/15/8589/12849/assets"
    )
    assert response.status_code == 404
    resp = response.json()
    assert resp["detail"] == "SearchId `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` not found"


def test_tilejson(app, search_no_bbox, search_bbox):
    """Create TileJSON."""
    response = app.get(f"/searches/{search_no_bbox}/WebMercatorQuad/tilejson.json")
    assert response.status_code == 400

    response = app.get(
        f"/searches/{search_no_bbox}/WebMercatorQuad/tilejson.json?assets=cog"
    )
    assert response.headers["content-type"] == "application/json"
    assert response.status_code == 200
    resp = response.json()
    assert resp["name"] == search_no_bbox
    assert resp["minzoom"] == 0
    assert resp["maxzoom"] == 24
    assert round(resp["bounds"][0]) == -180
    assert "?assets=cog" in resp["tiles"][0]

    response = app.get(
        f"/searches/{search_no_bbox}/WebMercatorQuad/tilejson.json?assets=cog&scan_limit=100&items_limit=1&time_limit=2&exitwhenfull=False&skipcovered=False"
    )
    assert response.headers["content-type"] == "application/json"
    assert response.status_code == 200
    resp = response.json()
    assert (
        "?assets=cog&scan_limit=100&items_limit=1&time_limit=2&exitwhenfull=False&skipcovered=False"
        in resp["tiles"][0]
    )

    response = app.get(
        f"/searches/{search_no_bbox}/WebMercatorQuad/tilejson.json?expression=cog"
    )
    assert response.status_code == 200
    resp = response.json()
    assert "?expression=cog" in resp["tiles"][0]

    response = app.get(
        f"/searches/{search_no_bbox}/WorldCRS84Quad/tilejson.json?assets=cog"
    )
    assert response.status_code == 200
    resp = response.json()
    assert resp["minzoom"] == 0
    assert resp["maxzoom"] == 23
    for xc, yc in zip(resp["bounds"], [-180.0, -90.0, 180.0, 90.0]):
        assert round(xc, 5) == round(yc, 5)
    assert "?assets=cog" in resp["tiles"][0]

    response = app.get(
        f"/searches/{search_no_bbox}/WebMercatorQuad/tilejson.json?assets=cog&tile_format=png"
    )
    assert response.status_code == 200
    resp = response.json()
    assert ".png?assets=cog" in resp["tiles"][0]

    response = app.get(
        f"/searches/{search_bbox}/WebMercatorQuad/tilejson.json?assets=cog"
    )
    assert response.headers["content-type"] == "application/json"
    assert response.status_code == 200
    resp = response.json()
    assert resp["name"] == search_bbox
    assert resp["minzoom"] == 0
    assert resp["maxzoom"] == 24
    assert resp["bounds"] == [-85.535, 36.137, -85.465, 36.179]
    assert "?assets=cog" in resp["tiles"][0]

    # searchId not found
    response = app.get(
        "/searches/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/WebMercatorQuad/tilejson.json?assets=cog"
    )
    assert response.status_code == 404
    resp = response.json()
    assert resp["detail"] == "SearchId `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` not found"


@patch("rio_tiler.io.rasterio.rasterio")
def test_tiles(rio, app, search_no_bbox, search_bbox):
    """Create tiles."""
    rio.open = mock_rasterio_open

    z, x, y = 15, 8589, 12849

    # missing assets
    response = app.get(f"/searches/{search_no_bbox}/tiles/WebMercatorQuad/{z}/{x}/{y}")
    assert response.status_code == 400

    response = app.get(
        f"/searches/{search_no_bbox}/tiles/WebMercatorQuad/{z}/{x}/{y}?assets=cog"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    meta = parse_img(response.content)
    assert meta["width"] == 256
    assert meta["height"] == 256

    response = app.get(
        f"/searches/{search_no_bbox}/tiles/WebMercatorQuad/{z}/{x}/{y}?assets=cog&buffer=0.5"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    meta = parse_img(response.content)
    assert meta["width"] == 257
    assert meta["height"] == 257

    response = app.get(
        f"/searches/{search_no_bbox}/tiles/WebMercatorQuad/{z}/{x}/{y}.png?assets=cog"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    meta = parse_img(response.content)
    assert meta["width"] == 256
    assert meta["height"] == 256

    # tile is outside mosaic bbox, it should return 404 (NoAssetFoundError)
    response = app.get(
        f"/searches/{search_bbox}/tiles/WebMercatorQuad/{z}/{x}/{y}?assets=cog"
    )
    assert response.status_code in [404, 204]

    response = app.get(
        f"/searches/{search_no_bbox}/tiles/WebMercatorQuad/{z}/{x}/{y}.tif?assets=cog"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff; application=geotiff"
    meta = parse_img(response.content)
    assert meta["crs"] == CRS.from_epsg(3857)
    assert meta["width"] == 256
    assert meta["height"] == 256

    response = app.get(
        f"/searches/{search_no_bbox}/tiles/WorldCRS84Quad/18/137421/78424.tif?assets=cog"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff; application=geotiff"
    meta = parse_img(response.content)
    assert meta["crs"] == CRS.from_epsg(4326)
    assert meta["width"] == 256
    assert meta["height"] == 256

    # searchId not found
    response = app.get(
        "/searches/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/tiles/WebMercatorQuad/0/0/0?assets=cog"
    )
    assert response.status_code == 404
    resp = response.json()
    assert resp["detail"] == "SearchId `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` not found"


def test_wmts(app, search_no_bbox):
    """Create wmts document."""
    # missing assets
    response = app.get(
        f"/searches/{search_no_bbox}/WebMercatorQuad/WMTSCapabilities.xml"
    )
    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Could not find any valid layers in metadata or construct one from Query Parameters."
    )

    response = app.get(
        f"/searches/{search_no_bbox}/WebMercatorQuad/WMTSCapabilities.xml?assets=cog"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/xml"

    # Validate it's a good WMTS
    with rasterio.open(io.BytesIO(response.content)) as src:
        assert src.crs == "epsg:3857"
        assert src.profile["driver"] == "WMTS"

    response = app.get(
        f"/searches/{search_no_bbox}/WorldCRS84Quad/WMTSCapabilities.xml?assets=cog"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/xml"

    # Validate it's a good WMTS
    with rasterio.open(io.BytesIO(response.content)) as src:
        assert src.crs == "OGC:CRS84"
        assert src.profile["driver"] == "WMTS"


@patch("rio_tiler.io.rasterio.rasterio")
def test_cql2(rio, app):
    """Test with cql2."""
    rio.open = mock_rasterio_open

    query = {
        "filter": {
            "op": "=",
            "args": [{"property": "collection"}, "noaa-emergency-response"],
        }
    }
    response = app.post("/searches/register", json=query)
    assert response.status_code == 200
    resp = response.json()
    assert resp["id"]
    assert resp["links"]

    cql2_id = resp["id"]

    response = app.get(f"/searches/{cql2_id}/info")
    assert response.status_code == 200
    resp = response.json()
    assert resp["search"]
    assert resp["links"]
    search = resp["search"]
    assert search["search"] == {
        "filter": {
            "op": "=",
            "args": [{"property": "collection"}, "noaa-emergency-response"],
        },
        "filter-lang": "cql2-json",
    }
    assert search["metadata"] == {"type": "mosaic"}

    response = app.get(f"/searches/{cql2_id}/point/-85.6358,36.1624/assets")
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 1
    assert list(resp[0]) == ["id", "bbox", "assets", "collection"]
    assert resp[0]["id"] == "20200307aC0853900w361030"

    response = app.get(
        f"/searches/{cql2_id}/tiles/WebMercatorQuad/15/8589/12849/assets"
    )
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 1
    assert list(resp[0]) == ["id", "bbox", "assets", "collection"]
    assert resp[0]["id"] == "20200307aC0853900w361030"

    response = app.get(f"/searches/{cql2_id}/WebMercatorQuad/tilejson.json?assets=cog")
    assert response.headers["content-type"] == "application/json"
    assert response.status_code == 200
    resp = response.json()
    assert resp["name"] == cql2_id
    assert resp["minzoom"] == 0
    assert resp["maxzoom"] == 24
    assert round(resp["bounds"][0]) == -180
    # Make sure we return a tilejson with the `/{search_id}/tiles/{tms}` format
    assert (
        f"/searches/{cql2_id}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}?assets=cog"
        in resp["tiles"][0]
    )

    z, x, y = 15, 8589, 12849
    response = app.get(
        f"/searches/{cql2_id}/tiles/WebMercatorQuad/{z}/{x}/{y}?assets=cog"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    meta = parse_img(response.content)
    assert meta["width"] == 256
    assert meta["height"] == 256


@patch("rio_tiler.io.rasterio.rasterio")
def test_cql2_with_geometry(rio, app):
    """Test with cql2 with geometry filter."""
    rio.open = mock_rasterio_open
    # Filter with geometry
    query = {
        "filter": {
            "op": "and",
            "args": [
                {
                    "op": "=",
                    "args": [{"property": "collection"}, "noaa-emergency-response"],
                },
                {
                    "op": "s_intersects",
                    "args": [
                        {"property": "geometry"},
                        {
                            "coordinates": [
                                [
                                    [-85.535, 36.137],
                                    [-85.535, 36.179],
                                    [-85.465, 36.179],
                                    [-85.465, 36.137],
                                    [-85.535, 36.137],
                                ]
                            ],
                            "type": "Polygon",
                        },
                    ],
                },
            ],
        }
    }
    response = app.post("/searches/register", json=query)
    assert response.status_code == 200
    resp = response.json()
    assert resp["id"]
    assert resp["links"]

    cql2_id = resp["id"]

    response = app.get(f"/searches/{cql2_id}/info")
    assert response.status_code == 200
    resp = response.json()
    assert resp["search"]
    assert resp["links"]
    search = resp["search"]
    assert search["metadata"] == {"type": "mosaic"}

    # make sure we can find assets when having both geometry filter and geometry
    response = app.get(
        f"/searches/{cql2_id}/tiles/WebMercatorQuad/15/8601/12849/assets"
    )
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 2

    # point is outside the geometry filter
    response = app.get(f"/searches/{cql2_id}/point/-85.6358,36.1624/assets")
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 0

    # make sure we can find assets when having both geometry filter and geometry
    response = app.get(
        f"/searches/{cql2_id}/tiles/WebMercatorQuad/15/8601/12849/assets"
    )
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 2

    # tile is outside the geometry filter
    response = app.get(
        f"/searches/{cql2_id}/tiles/WebMercatorQuad/15/8589/12849/assets"
    )
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 0

    # tile is outside the geometry filter
    z, x, y = 15, 8589, 12849
    response = app.get(
        f"/searches/{cql2_id}/tiles/WebMercatorQuad/{z}/{x}/{y}?assets=cog"
    )
    assert response.status_code in [404, 204]


def test_query_with_metadata(app):
    """Test with cql2."""
    query = {
        "filter": {
            "op": "=",
            "args": [{"property": "collection"}, "noaa-emergency-response"],
        },
        "metadata": {"name": "mymosaic", "minzoom": 1, "maxzoom": 2},
    }

    response = app.post("/searches/register", json=query)
    assert response.status_code == 200
    resp = response.json()
    assert resp["id"]
    assert resp["links"]

    mosaic_id = resp["id"]

    response = app.get(f"/searches/{mosaic_id}/info")
    assert response.status_code == 200
    resp = response.json()
    assert resp["search"]
    assert resp["links"]
    search = resp["search"]
    assert search["search"] == {
        "filter": {
            "args": [{"property": "collection"}, "noaa-emergency-response"],
            "op": "=",
        },
        "filter-lang": "cql2-json",
    }
    assert search["metadata"] == {
        "type": "mosaic",
        "name": "mymosaic",
        "minzoom": 1,
        "maxzoom": 2,
    }

    response = app.get(
        f"/searches/{mosaic_id}/WebMercatorQuad/tilejson.json?assets=cog"
    )
    assert response.status_code == 200
    resp = response.json()
    assert resp["minzoom"] == 1
    assert resp["maxzoom"] == 2

    # Check that `defaults` created `tilejson` URL in links
    query = {
        "filter": {
            "op": "=",
            "args": [{"property": "collection"}, "noaa-emergency-response"],
        },
        "metadata": {
            "name": "mymosaic",
            "minzoom": 1,
            "maxzoom": 2,
            "defaults": {
                "one_band": {
                    "assets": ["cog"],
                    "asset_bidx": ["cog|1"],
                },
                "three_bands": {
                    "assets": ["cog"],
                    "asset_bidx": ["cog|1,2,3"],
                },
                "rescale": {
                    "assets": ["cog"],
                    "asset_bidx": ["cog|1"],
                    "rescale": [
                        [-1, 1],
                    ],
                },
                "colormap": {
                    "assets": ["cog"],
                    "asset_bidx": ["cog|1"],
                    "colormap": {"1": [0, 0, 0, 255], "1000": [255, 255, 255, 255]},
                },
                # missing `assets`
                "bad_layer": {
                    "asset_bidx": ["cog|1,2,3"],
                },
            },
        },
    }

    with pytest.warns(UserWarning):
        response = app.post("/searches/register", json=query)
    assert response.status_code == 200
    resp = response.json()
    assert resp["id"]
    assert len(resp["links"]) == 8  # info, tilejson, map, wmts, tilejson layers
    link = resp["links"][-2]

    mosaic_id_metadata = resp["id"]

    assert link["title"] == "TileJSON link for `one_band` layer (Template URL)."
    assert "asset_bidx=cog%7C1" in link["href"]
    assert "assets=cog" in link["href"]

    # Test WMTS
    # 1. missing assets and no metadata layers
    response = app.get(f"/searches/{mosaic_id}/WebMercatorQuad/WMTSCapabilities.xml")
    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Could not find any valid layers in metadata or construct one from Query Parameters."
    )

    # 2. assets and no metadata layers
    response = app.get(
        f"/searches/{mosaic_id}/WebMercatorQuad/WMTSCapabilities.xml",
        params={"assets": "cog"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/xml"

    with rasterio.open(io.BytesIO(response.content)) as src:
        assert src.crs == "epsg:3857"
        assert src.profile["driver"] == "WMTS"
        assert not src.subdatasets

    # 3. no assets and metadata layers
    with pytest.warns(UserWarning):
        response = app.get(
            f"/searches/{mosaic_id_metadata}/WebMercatorQuad/WMTSCapabilities.xml"
        )
    assert response.status_code == 200

    assert response.headers["content-type"] == "application/xml"

    with rasterio.open(io.BytesIO(response.content)) as src:
        assert src.profile["driver"] == "WMTS"
        assert len(src.subdatasets) == 4

    # 4. assets and metadata layers
    with pytest.warns(UserWarning):
        response = app.get(
            f"/searches/{mosaic_id_metadata}/WebMercatorQuad/WMTSCapabilities.xml",
            params={"assets": "cog"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/xml"

    with rasterio.open(io.BytesIO(response.content)) as src:
        assert src.profile["driver"] == "WMTS"
        assert len(src.subdatasets) == 5

    with pytest.warns(UserWarning):
        response = app.get(f"/searches/{mosaic_id_metadata}/info")
    assert response.status_code == 200
    resp = response.json()
    assert resp["search"]["hash"] == mosaic_id_metadata
    assert len(resp["links"]) == 12  # self, tilejson (5), map (5), wmts (1)

    assert resp["links"][1]["title"] == "TileJSON link (Template URL)."
    assert (
        resp["links"][2]["title"] == "TileJSON link for `rescale` layer (Template URL)."
    )
    assert (
        resp["links"][3]["title"]
        == "TileJSON link for `colormap` layer (Template URL)."
    )

    assert "asset_bidx=cog%7C1" in resp["links"][2]["href"]
    assert "assets=cog" in resp["links"][2]["href"]


@patch("rio_tiler.io.rasterio.rasterio")
def test_statistics(rio, app, search_no_bbox, search_bbox):
    """Get Stats."""
    rio.open = mock_rasterio_open

    feat = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-85.64065933227539, 36.16587374136926],
                            [-85.64546585083008, 36.161716102717804],
                            [-85.64443588256836, 36.158043338486344],
                            [-85.64083099365234, 36.157904740240866],
                            [-85.63679695129393, 36.15901351934466],
                            [-85.6358528137207, 36.161577510965],
                            [-85.63568115234375, 36.16441859292501],
                            [-85.63902854919434, 36.16511152412467],
                            [-85.64065933227539, 36.16587374136926],
                        ]
                    ],
                },
            }
        ],
    }

    response = app.post(
        f"/searches/{search_no_bbox}/statistics", json=feat, params={"max_size": 1024}
    )
    assert response.status_code == 400

    response = app.post(
        f"/searches/{search_no_bbox}/statistics",
        json=feat,
        params={"assets": "cog", "max_size": 1024},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/geo+json"
    assert response.json()["features"][0]["properties"]["statistics"]["cog_b1"]

    response = app.post(
        f"/searches/{search_no_bbox}/statistics",
        json=feat["features"][0],
        params={"assets": "cog", "max_size": 1024},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/geo+json"
    assert response.json()["properties"]["statistics"]["cog_b1"]

    # searchId not found
    response = app.post(
        "/searches/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/statistics",
        json=feat,
        params={"assets": "cog", "max_size": 1024},
    )
    assert response.status_code == 404
    resp = response.json()
    assert resp["detail"] == "SearchId `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` not found"

    # with Algorithm
    response = app.post(
        f"/searches/{search_no_bbox}/statistics",
        json=feat,
        params={
            "assets": "cog",
            "max_size": 1024,
            "algorithm": "normalizedIndex",
            "asset_bidx": "cog|1,2",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/geo+json"
    resp = response.json()
    stats = resp["features"][0]["properties"]["statistics"]
    assert "(cog_b2 - cog_b1) / (cog_b2 + cog_b1)" in stats


def test_mosaic_list(app):
    """Test list mosaic."""
    response = app.get("/searches/list")
    assert response.status_code == 200
    resp = response.json()
    assert ["searches", "links", "context"] == list(resp)
    assert len(resp["searches"]) > 0
    assert len(resp["links"]) >= 1

    response = app.get("/searches/list?limit=1")
    assert response.status_code == 200
    resp = response.json()
    assert ["searches", "links", "context"] == list(resp)
    assert len(resp["searches"]) == 1
    assert len(resp["links"]) >= 2

    response = app.get("/searches/list?limit=1&offset=1")
    assert response.status_code == 200
    resp = response.json()
    assert ["searches", "links", "context"] == list(resp)
    assert len(resp["searches"]) == 1
    assert len(resp["links"]) >= 3

    query = {
        "filter": {
            "op": "=",
            "args": [{"property": "collection"}, "noaa-emergency-response"],
        },
        "metadata": {"data": "noaa", "num": 2},
    }
    response = app.post("/searches/register", json=query)

    query = {
        "filter": {
            "op": "=",
            "args": [{"property": "collection"}, "noaa-emergency-response"],
        },
        "metadata": {"data": "noaa", "num": 1},
    }
    response = app.post("/searches/register", json=query)

    query = {
        "filter": {
            "op": "=",
            "args": [{"property": "collection"}, "noaa-emergency-response"],
        },
        "metadata": {"data": "noaa", "num": 3},
    }
    response = app.post("/searches/register", json=query)

    response = app.get("/searches/list?data=noaa")
    assert response.status_code == 200
    resp = response.json()
    assert ["searches", "links", "context"] == list(resp)
    assert resp["context"] == {"returned": 3, "limit": 10, "matched": 3}
    assert len(resp["searches"]) == 3
    assert [s["search"]["metadata"]["num"] for s in resp["searches"]] == [2, 1, 3]

    response = app.get("/searches/list?data=noaa&sortby=%2Bnum")
    assert response.status_code == 200
    resp = response.json()
    assert [s["search"]["metadata"]["num"] for s in resp["searches"]] == [1, 2, 3]

    response = app.get("/searches/list?data=noaa&sortby=num")
    assert response.status_code == 200
    resp = response.json()
    assert [s["search"]["metadata"]["num"] for s in resp["searches"]] == [1, 2, 3]

    response = app.get("/searches/list?data=noaa&sortby=-num")
    assert response.status_code == 200
    resp = response.json()
    assert [s["search"]["metadata"]["num"] for s in resp["searches"]] == [3, 2, 1]

    response = app.get("/searches/list?sortby=lastused")
    assert response.status_code == 200
    resp = response.json()
    assert resp["context"]
    dates = [
        datetime.strptime(s["search"]["lastused"], "%Y-%m-%dT%H:%M:%S.%fZ")
        for s in resp["searches"]
    ]
    assert dates[0] < dates[-1]

    response = app.get("/searches/list?sortby=-lastused")
    assert response.status_code == 200
    resp = response.json()
    assert resp["context"]
    dates = [
        datetime.strptime(s["search"]["lastused"], "%Y-%m-%dT%H:%M:%S.%fZ")
        for s in resp["searches"]
    ]
    assert dates[0] > dates[-1]


def test_map(app, search_bbox):
    """test /map endpoint."""
    response = app.get(f"/searches/{search_bbox}/WebMercatorQuad/map.html")
    assert response.status_code == 400

    response = app.get(
        f"/searches/{search_bbox}/WebMercatorQuad/map.html", params={"assets": "cog"}
    )
    assert response.status_code == 200


@patch("rio_tiler.io.rasterio.rasterio")
def test_feature(rio, app, search_no_bbox):
    """Get feature image."""
    rio.open = mock_rasterio_open

    feat = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-85.64065933227539, 36.16587374136926],
                    [-85.64546585083008, 36.161716102717804],
                    [-85.64443588256836, 36.158043338486344],
                    [-85.64083099365234, 36.157904740240866],
                    [-85.63679695129393, 36.15901351934466],
                    [-85.6358528137207, 36.161577510965],
                    [-85.63568115234375, 36.16441859292501],
                    [-85.63902854919434, 36.16511152412467],
                    [-85.64065933227539, 36.16587374136926],
                ]
            ],
        },
    }

    response = app.post(
        f"/searches/{search_no_bbox}/feature", json=feat, params={"max_size": 1024}
    )
    assert response.status_code == 400

    response = app.post(
        f"/searches/{search_no_bbox}/feature",
        json=feat,
        params={"assets": "cog", "max_size": 1024},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    meta = parse_img(response.content)
    assert meta["width"] == 725
    assert meta["height"] == 591
    assert meta["count"] == 4

    response = app.post(
        f"/searches/{search_no_bbox}/feature.jpeg",
        json=feat,
        params={"assets": "cog", "max_size": 1024},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    meta = parse_img(response.content)
    assert meta["width"] == 725
    assert meta["height"] == 591
    assert meta["count"] == 3

    response = app.post(
        f"/searches/{search_no_bbox}/feature/300x400.jpeg",
        json=feat,
        params={"assets": "cog", "max_size": 1024},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    meta = parse_img(response.content)
    assert meta["width"] == 300
    assert meta["height"] == 400
    assert meta["count"] == 3

    response = app.post(
        f"/searches/{search_no_bbox}/feature/300x400.tif",
        json=feat,
        params={"assets": "cog", "max_size": 1024},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff; application=geotiff"
    meta = parse_img(response.content)
    assert meta["crs"] == "epsg:4326"

    response = app.post(
        f"/searches/{search_no_bbox}/feature/300x400.tif",
        json=feat,
        params={"assets": "cog", "max_size": 1024, "dst_crs": "epsg:3857"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff; application=geotiff"
    meta = parse_img(response.content)
    assert meta["crs"] == "epsg:3857"


@patch("rio_tiler.io.rasterio.rasterio")
def test_bbox(rio, app, search_no_bbox):
    """Get bbox image."""
    rio.open = mock_rasterio_open

    bbox = [
        -85.64546585083008,
        36.157904740240866,
        -85.63568115234375,
        36.16587374136926,
    ]
    str_bbox = ",".join(map(str, bbox))
    response = app.get(
        f"/searches/{search_no_bbox}/bbox/{str_bbox}.png", params={"max_size": 1024}
    )
    assert response.status_code == 400

    response = app.get(
        f"/searches/{search_no_bbox}/bbox/{str_bbox}.png",
        params={"assets": "cog", "max_size": 1024},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    meta = parse_img(response.content)
    assert meta["width"] == 725
    assert meta["height"] == 591
    assert meta["count"] == 4

    response = app.get(
        f"/searches/{search_no_bbox}/bbox/{str_bbox}.jpeg",
        params={"assets": "cog", "max_size": 1024},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    meta = parse_img(response.content)
    assert meta["width"] == 725
    assert meta["height"] == 591
    assert meta["count"] == 3

    response = app.get(
        f"/searches/{search_no_bbox}/bbox/{str_bbox}/300x400.jpeg",
        params={"assets": "cog", "max_size": 1024},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    meta = parse_img(response.content)
    assert meta["width"] == 300
    assert meta["height"] == 400
    assert meta["count"] == 3

    response = app.get(
        f"/searches/{search_no_bbox}/bbox/{str_bbox}.tif",
        params={"assets": "cog", "max_size": 1024},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff; application=geotiff"
    meta = parse_img(response.content)
    assert meta["crs"] == "epsg:4326"

    response = app.get(
        f"/searches/{search_no_bbox}/bbox/{str_bbox}.tif",
        params={"assets": "cog", "max_size": 1024, "dst_crs": "epsg:3857"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff; application=geotiff"
    meta = parse_img(response.content)
    assert meta["crs"] == "epsg:3857"


def test_query_point_searches(app, search_no_bbox, search_bbox):
    """Test getting values for a Point."""
    response = app.get(
        f"/searches/{search_no_bbox}/point/-85.5,36.1624", params={"assets": "cog"}
    )

    assert response.status_code == 200
    resp = response.json()

    values = resp["values"]
    assert len(values) == 2
    assert values[0][0] == "noaa-emergency-response/20200307aC0853130w361030"
    assert values[0][2] == ["cog_b1", "cog_b2", "cog_b3"]

    # with coord-crs
    response = app.get(
        f"/searches/{search_no_bbox}/point/-9517816.46282489,4322990.432036275",
        params={"assets": "cog", "coord_crs": "epsg:3857"},
    )
    assert response.status_code == 200
    resp = response.json()
    assert len(resp["values"]) == 2

    # SearchId not found
    response = app.get(
        "/searches/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/point/-85.5,36.1624",
        params={"assets": "cog"},
    )
    assert response.status_code == 404
    resp = response.json()
    assert resp["detail"] == "SearchId `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` not found"

    # outside of searchid bbox
    response = app.get(
        f"/searches/{search_bbox}/point/-86.0,35.0", params={"assets": "cog"}
    )

    assert response.status_code == 204  # (no content)


def test_cache_middleware_settings(app, search_no_bbox):
    """Make sure some endpoints do not have cache-control headers."""
    response = app.get("/searches/list")
    assert response.status_code == 200
    assert not response.headers.get("Cache-Control")

    response = app.get(
        f"/searches/{search_no_bbox}/point/-85.5,36.1624", params={"assets": "cog"}
    )
    assert response.status_code == 200
    assert response.headers.get("Cache-Control")


def test_search_ids_parameter(app):
    """Check that ids parameter work."""
    query = {
        "collections": ["noaa-emergency-response"],
    }
    response = app.post("/searches/register", json=query)
    assert response.status_code == 200
    resp = response.json()
    search_id = resp["id"]

    response = app.get(f"/searches/{search_id}/point/-85.5,36.1624/assets")
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) >= 1

    query = {
        "collections": ["noaa-emergency-response"],
        "ids": ["20200307aC0853130w361030"],
    }
    response = app.post("/searches/register", json=query)
    assert response.status_code == 200
    resp = response.json()
    search_id = resp["id"]

    response = app.get(f"/searches/{search_id}/point/-85.5,36.1624/assets")
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 1
    assert resp[0]["id"] == "20200307aC0853130w361030"

    query = {
        "collections": ["noaa-emergency-response"],
        "ids": ["20200307aC0853130w361030"],
        "filter-lang": "cql2-json",
    }
    response = app.post("/searches/register", json=query)
    assert response.status_code == 200
    resp = response.json()
    search_id = resp["id"]

    response = app.get(f"/searches/{search_id}/point/-85.5,36.1624/assets")
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 1
    assert resp[0]["id"] == "20200307aC0853130w361030"

    query = {
        "filter": {
            "op": "and",
            "args": [
                {
                    "op": "=",
                    "args": [{"property": "collection"}, "noaa-emergency-response"],
                },
                {
                    "op": "=",
                    "args": [{"property": "id"}, "20200307aC0853130w361030"],
                },
            ],
        },
        "filter-lang": "cql2-json",
    }
    response = app.post("/searches/register", json=query)
    assert response.status_code == 200
    resp = response.json()
    search_id = resp["id"]

    response = app.get(f"/searches/{search_id}/point/-85.5,36.1624/assets")
    assert response.status_code == 200
    resp = response.json()
    assert len(resp) == 1
    assert resp[0]["id"] == "20200307aC0853130w361030"
