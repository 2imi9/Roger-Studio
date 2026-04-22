import logging

import httpx
from fastapi import APIRouter, Query

from app.models.schemas import EnvDataResult

logger = logging.getLogger(__name__)

router = APIRouter()

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


@router.get("/env-data", response_model=EnvDataResult)
async def get_env_data(
    west: float = Query(...),
    south: float = Query(...),
    east: float = Query(...),
    north: float = Query(...),
    variables: str = Query("wind,temperature,solar,humidity"),
) -> EnvDataResult:
    """Fetch current weather + solar data from Open-Meteo for a bbox centroid.

    Non-200s and network errors are captured in ``result.error`` rather
    than silently returning null fields with a 200 OK. That lets the
    frontend distinguish "upstream down" from "this location has no data"
    — previously both surfaced as nulls and the UI had no way to show
    "ERA5/Open-Meteo unavailable, try again later".
    """
    lat = (north + south) / 2
    lon = (east + west) / 2
    var_list = [v.strip() for v in variables.split(",")]

    result = EnvDataResult()

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Fetch current weather from Open-Meteo
        params: dict = {
            "latitude": lat,
            "longitude": lon,
            "current": [],
        }

        if "wind" in var_list:
            params["current"].extend(["wind_speed_10m", "wind_direction_10m"])
        if "temperature" in var_list:
            params["current"].append("temperature_2m")
        if "humidity" in var_list:
            params["current"].append("relative_humidity_2m")
        if "solar" in var_list:
            params["current"].append("direct_radiation")

        if params["current"]:
            params["current"] = ",".join(params["current"])
            try:
                resp = await client.get(OPEN_METEO_FORECAST, params=params)
                if resp.status_code == 200:
                    data = resp.json().get("current", {})

                    if "wind" in var_list:
                        result.wind = {
                            "speed": data.get("wind_speed_10m", 0),
                            "direction": data.get("wind_direction_10m", 0),
                        }
                    if "temperature" in var_list:
                        result.temperature = data.get("temperature_2m")
                    if "humidity" in var_list:
                        result.humidity = data.get("relative_humidity_2m")
                    if "solar" in var_list:
                        result.solar_irradiance = data.get("direct_radiation")
                else:
                    # Open-Meteo returned a non-200 (rate-limited, bad
                    # params, maintenance). Capture the status so the UI
                    # can differentiate "service returned an error" from
                    # "service was unreachable".
                    detail = resp.text[:200] if resp.text else ""
                    result.error = (
                        f"open_meteo_http_{resp.status_code}"
                        + (f": {detail}" if detail else "")
                    )
                    logger.warning(
                        "Open-Meteo returned HTTP %d for (%.3f, %.3f): %s",
                        resp.status_code, lat, lon, detail,
                    )
            except httpx.TimeoutException as e:
                result.error = f"open_meteo_timeout: {e}"[:200]
                logger.warning("Open-Meteo timeout for (%.3f, %.3f): %s", lat, lon, e)
            except httpx.RequestError as e:
                # Network error, DNS failure, connection refused, etc.
                result.error = f"open_meteo_unreachable: {type(e).__name__}: {e}"[:200]
                logger.warning(
                    "Open-Meteo unreachable for (%.3f, %.3f): %s",
                    lat, lon, e,
                )
            except ValueError as e:
                # JSON decode failure (very rare — Open-Meteo always
                # returns JSON, but paranoia is cheap here).
                result.error = f"open_meteo_parse_error: {e}"[:200]
                logger.warning("Open-Meteo JSON decode error: %s", e)

    return result
