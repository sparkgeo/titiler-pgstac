"""titiler.pgstac extensions."""

import warnings
from dataclasses import dataclass
from typing import List, Tuple
from urllib.parse import urlencode

from cogeo_mosaic.errors import MosaicNotFoundError
from fastapi import Depends
from psycopg.rows import class_row
from starlette.requests import Request
from starlette.routing import NoMatchFound

from titiler.core.factory import FactoryExtension
from titiler.core.utils import check_query_params
from titiler.pgstac import model
from titiler.pgstac.factory import MosaicTilerFactory


@dataclass
class searchInfoExtension(FactoryExtension):
    """Add /info endpoint"""

    def register(self, factory: MosaicTilerFactory):  # noqa: C901
        """Register endpoint to the tiler factory."""

        @factory.router.get(
            "/info",
            responses={200: {"description": "Get Search query metadata."}},
            response_model=model.Info,
            response_model_exclude_none=True,
            operation_id=f"{factory.operation_prefix}getInfo",
        )
        def info(  # noqa: C901
            request: Request, search_id=Depends(factory.path_dependency)
        ):
            """Get Search query metadata."""
            with request.app.state.dbpool.connection() as conn:
                with conn.cursor(row_factory=class_row(model.Search)) as cursor:
                    cursor.execute(
                        "SELECT * FROM searches WHERE hash=%s;",
                        (search_id,),
                    )
                    search_info = cursor.fetchone()

            if not search_info:
                raise MosaicNotFoundError(f"SearchId `{search_id}` not found")

            links: List[model.Link] = [
                model.Link(
                    rel="self",
                    title="Mosaic metadata",
                    href=factory.url_for(request, "info"),
                ),
            ]

            layers: List[Tuple[str, str]] = []
            if renders := search_info.metadata.defaults_params:
                # List of dependencies a `/tile` URL should validate
                # Note: Those dependencies should only require Query() inputs
                tile_dependencies = [
                    factory.layer_dependency,
                    factory.dataset_dependency,
                    factory.pixel_selection_dependency,
                    factory.process_dependency,
                    factory.colormap_dependency,
                    factory.render_dependency,
                    factory.assets_accessor_dependency,
                    factory.reader_dependency,
                    factory.backend_dependency,
                ]

                for name, values in renders.items():
                    if check_query_params(tile_dependencies, values):
                        layers.append((name, urlencode(values, doseq=True)))
                    else:
                        warnings.warn(
                            f"Cannot construct URL for layer `{name}`",
                            UserWarning,
                            stacklevel=2,
                        )

            try:
                tilejson_endpoint = factory.url_for(
                    request, "tilejson", tileMatrixSetId="{tileMatrixSetId}"
                )
                links.append(
                    model.Link(
                        title="TileJSON link (Template URL).",
                        rel="tilejson",
                        href=tilejson_endpoint,
                        templated=True,
                    ),
                )
                for layer, qs in layers:
                    links.append(
                        model.Link(
                            title=f"TileJSON link for `{layer}` layer (Template URL).",
                            rel="tilejson",
                            href=tilejson_endpoint + f"?{qs}",
                            templated=True,
                        ),
                    )

            except NoMatchFound:
                pass

            try:
                map_viewer_endpoint = factory.url_for(
                    request, "map_viewer", tileMatrixSetId="{tileMatrixSetId}"
                )
                links.append(
                    model.Link(
                        rel="map",
                        title="Map viewer link (Template URL).",
                        href=map_viewer_endpoint,
                        templated=True,
                    )
                )
                for layer, qs in layers:
                    links.append(
                        model.Link(
                            title=f"Map viewer link for `{layer}` layer (Template URL).",
                            rel="map",
                            href=map_viewer_endpoint + f"?{qs}",
                            templated=True,
                        ),
                    )

            except NoMatchFound:
                pass

            try:
                wmts_endpoint = factory.url_for(
                    request, "wmts", tileMatrixSetId="{tileMatrixSetId}"
                )
                links.append(
                    model.Link(
                        rel="wmts",
                        title="WMTS link (Template URL)",
                        href=wmts_endpoint,
                        templated=True,
                    )
                )

            except NoMatchFound:
                pass

            return model.Info(search=search_info, links=links)
