# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is "The Adventures of Indy" - an archaeological AI system for discovering lost civilizations in the Amazon rainforest using satellite imagery, LiDAR data, and machine learning. The project was submitted for OpenAI's City of Z competition and uses AI agents to analyze geospatial data for archaeological site detection.

## Development Commands

### Setup
```bash
uv sync && mkdir tmp && uv run earthengine authenticate
```

### Running Evaluations
```bash
# Run LiDAR evaluation
uv run -m src.evals.lidar.run_eval --exp_num 0 --exp_type eval

# Run Terra Preta evaluation  
uv run -m src.evals.terra_preta.run_eval --exp_num 1

# Generate evaluation statistics
uv run -m src.evals.lidar.stats --exp_num 0 --exp_type eval
```

### Data Processing Scripts
```bash
# Process Amazon boundary data
uv run -m src.scripts.amazon_boundary

# Process confirmed archaeological sites
uv run -m src.scripts.confirmed

# Process LiDAR tiles
uv run -m src.scripts.lidar_tiles.cms
uv run -m src.scripts.lidar_tiles.cms_download
uv run -m src.scripts.lidar_tiles.paracou_nouragues
uv run -m src.scripts.lidar_tiles.process_zscore
uv run -m src.scripts.lidar_tiles.prepare_eval_data
```

### Code Quality
```bash
# Lint code
uv run ruff check .

# Format code
uv run ruff format .
```

## Architecture

The codebase is organized into several key components:

### Core Configuration (`src/config.py`)
- Centralizes all API credentials and settings using Pydantic
- Manages Google Earth Engine, OpenAI, Mapbox, and Cloudflare R2 authentication
- Provides singleton `settings` object for global access

### Evaluation Framework (`src/evals/`)
Two main evaluation types:
- **LiDAR Analysis** (`src/evals/lidar/`): Analyzes LiDAR elevation data to detect earthworks and archaeological features
- **Terra Preta Detection** (`src/evals/terra_preta/`): Uses satellite imagery and vegetation indices to identify Amazonian Dark Earth sites

Both evaluations use OpenAI's o3 model with custom tools and detailed prompts for archaeological analysis.

### Data Processing Pipeline (`src/scripts/`)
- **Boundary Processing**: Converts RAISG shapefiles to GeoJSON format
- **Confirmed Sites**: Processes archaeological datasets from multiple academic sources
- **LiDAR Processing**: Downloads, processes, and analyzes massive LiDAR datasets using Modal for parallelization

### Utilities (`src/utils.py`)
- Geospatial filtering functions for Amazon boundary
- Image encoding utilities for AI model inputs
- Haversine distance calculations for geographic analysis

## Key Dependencies

- **Geospatial**: `geopandas`, `fiona`, `shapely`, `rasterio`, `gdal`
- **AI/ML**: `openai`, `scikit-learn`, `numpy`
- **Cloud Services**: `boto3` (R2 storage), `earthengine-api` (Google Earth Engine)
- **Data Processing**: `pandas`, `laspy` (LiDAR), `pdal`
- **Visualization**: `folium`, `matplotlib`, `plotly`
- **Parallelization**: `modal` for distributed processing

## Data Structure

- `data/`: Contains processed archaeological and geospatial datasets
- `data/confirmed/`: Verified archaeological sites from academic sources
- `data/evals/`: Evaluation datasets and results
- `data/lidar_tiles/`: LiDAR tile metadata and boundaries
- `tmp/`: Temporary processing outputs organized by experiment ID

## Environment Variables Required

```bash
OPENAI_API_KEY=your_api_key
MAPBOX_API_KEY=your_api_key  
GEE_PROJECT_ID=your_project_id
R2_ACCESS_KEY_ID=your_access_key_id
R2_SECRET_ACCESS_KEY=your_secret_access_key
R2_URL=your_url
R2_BUCKET_NAME=city-of-z
```