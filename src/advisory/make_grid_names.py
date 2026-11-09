import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

df = pd.read_parquet("data/sample/mock_forecasts.parquet")
cells = df[["grid_id", "lat", "lon"]].drop_duplicates("grid_id")

geo = Nominatim(user_agent="agricast-hackathon-2026")
reverse = RateLimiter(geo.reverse, min_delay_seconds=1.1)
PLACE_KEYS = ["hamlet", "village", "suburb", "neighbourhood", "town", "city", "municipality"]

def name_for(lat, lon):
    loc = reverse((lat, lon), zoom=14, language="en")
    if not loc:
        return None
    a = loc.raw.get("address", {})
    place = next((a[k] for k in PLACE_KEYS if k in a), None)
    area = a.get("county") or a.get("state_district") or a.get("state")
    return ", ".join(p for p in (place, area) if p) or None

cells["place_name"] = [name_for(r.lat, r.lon) or r.grid_id for r in cells.itertuples()]

dup = cells.duplicated("place_name", keep=False)
cells.loc[dup, "place_name"] += " (" + cells.loc[dup, "grid_id"] + ")"
cells[["grid_id", "place_name"]].to_csv("data/sample/grid_names.csv", index=False)
