// Coverage polygons for OlmoEarth project-labelled datasets. Kept in sync
// with `PROJECT_REGIONS` in backend/app/services/olmoearth_datasets.py. Used
// by the Map tab's "OlmoEarth Data Layers" section: when a dataset is
// cached AND toggled on, its coverage polygon renders as a shaded overlay
// so you can see on the map where the dataset applies.

export interface DatasetCoverage {
  name: string;
  color: string;
  geometry: GeoJSON.Polygon;
}

export const DATASET_COVERAGE: Record<string, DatasetCoverage> = {
  "allenai/olmoearth_projects_mangrove": {
    name: "Mangrove (tropical belt)",
    color: "#0891b2", // cyan — mangrove / water association
    geometry: {
      type: "Polygon",
      coordinates: [[
        [-180, -30],
        [180, -30],
        [180, 30],
        [-180, 30],
        [-180, -30],
      ]],
    },
  },
  "allenai/olmoearth_projects_awf": {
    name: "AWF southern Kenya",
    color: "#ca8a04", // savanna yellow
    geometry: {
      type: "Polygon",
      coordinates: [[
        [33.5, -5.0],
        [42.0, -5.0],
        [42.0, -1.0],
        [33.5, -1.0],
        [33.5, -5.0],
      ]],
    },
  },
};
