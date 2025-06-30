# Indy: Evals and AI Agents to search the Amazon for lost civilizations

![Indy](./assets/indy.png)

[This is my submission for OpenAI's City of Z competition](https://openai.com/openai-to-z-challenge/)

If you wish to read a comprehensive writeup, [go here](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/the-adventures-of-indy)

This repo contains the code and instructions to replicate the data sources I used and run the evals.

If you get stuck at any step, please send me an email (sarhaangulati737@gmail.com) and I'll get back to you asap!
If you want to build all the data sources yourself from scratch, you can do so by following the instructions below. However, if you want to just run the evals, please send me an email (sarhaangulati737@gmail.com) and I'll send you credentials for my R2 bucket. Its roughly ~400GB of data so its not feasible to share it here.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Run `uv sync` to install the dependencies.
3. Create a folder named `tmp` in the root directory.
4. Create a `.env` file in the root directory and add the following:

```bash
OPENAI_API_KEY=your_api_key # for openai
MAPBOX_API_KEY=your_api_key # for mapbox
GEE_PROJECT_ID=your_project_id # for google earth engine

# for lidar data storage!
R2_ACCESS_KEY_ID=your_access_key_id
R2_SECRET_ACCESS_KEY=your_secret_access_key
R2_URL=your_url
R2_BUCKET_NAME=city-of-z
```

1. in `src/scripts/lidar_tiles/process_zscore.py`, set the R2_URL and R2_BUCKET_NAME to your own!
2. Make sure to properly authenticate GEE `uv run earthengine authenticate`
3. Setup a bucket in R2 with the name: `city-of-z` and create API tokens that have Write permissions.

```bash
uv sync && mkdir tmp && uv run earthengine authenticate
```

## Running the evals

### Terra preta evals

Warning: this will take 2-3 hours since GEE rate limits us a lot!

```bash
uv run -m src.evals.terra_preta.run_eval --exp-num 1
```

Once done, this will save the results in `tmp/{exp_num}/`

To get stats (you can safely ignore the warnings):

```bash
uv run -m src.evals.terra_preta.stats --exp-num 1
```

### Lidar evals

After R2 has been setup, first lets prepare the data:

```bash
uv run -m src.scripts.lidar_tiles.prepare_eval_data
```

This creates json files in `data/evals/` that contain the signed urls for the lidar tiles.

We did 3 big runs:

1. exp_num: 0, exp_type: eval: the reported eval numbers. To replicate this please run the following:

```bash
uv run -m src.evals.lidar.run_eval --exp_num 3 --exp_type eval
```

Now, if you wish to get the stats for these runs, please run the following:

```bash
uv run -m src.evals.lidar.stats --exp_num 3 --exp_type eval
```

2. exp_num: 1, exp_type: test: a test run with 2000 tiles
3. exp_num: 2, exp_type: test: a test run with all the tiles, these are the numbers we report in the paper! (prompts and tools did not change!). To replicate this please run the following:

```bash
uv run -m src.evals.lidar.run_eval --exp_num 4 --exp_type test
```

Now, please run this to get the stats for our run, a geojson file with the results stored in `data/evals/lidar_eval_output_test_4.geojson` and a report json file with the results stored in `data/evals/report_lidar_eval_output_test_4.json`. We submit both these files to the competition.

```bash
uv run -m src.evals.lidar.stats --exp_num 4 --exp_type test
```

## Replicating the data sources

All our json, geojson, and csv files are stored in the `data` folder. Let's delete it and recreate it!

```bash
rm -rf data && mkdir data && mkdir data/raw
```

### Create the Amazon boundary

We use the maps produced by [RAISG](https://www.raisg.org/en/about/)

1. `https://www.raisg.org/en/maps/` has a map of the Amazon boundary under the name `RAISG Limits` (left column). Fill out the form, download the shapefile, unzip it, and put it in `data/raw`. You should have this: `data/raw/Limites2024/`
2. In this folder, we have the option of using either the `LimRAISG.shp` or `LimBiogeografico.shp`. We use `LimBiogeografico.shp`, as it is commonly used for Scientific and ecological research.
3. We ran this process on May 28th, 2025 and used the file retrieved on that date.

`uv run -m src.scripts.amazon_boundary`

This will create a file called `data/amazon_boundary.geojson`.

### Confirmed Site Datasets

Now let's build the datasets that will tell us about the archaeological sites that have been confirmed to exist. We need to know what has already been discovered so we don't waste our time and can have a higher degree of confidence in our results.

We have a total of 9871 confirmed archaeologically significant sites within our Amazon boundary.

First, follow the instructions below to create the datasets.

1. [Geometry by Design: Contribution of Lidar to the Understanding of Settlement Patterns of the Mound Villages in SW Amazonia](https://journal.caa-international.org/articles/45/files/submission/proof/45-1-988-1-10-20200428.pdf)

- Dataset of mound sites in the Amazon.
- PDF has information on a lot of mound sites in UTM19, we asked chatgpt to convert it to a csv!
- saved as `data/raw/iriarte.csv`
- dataset_id: `iriarte`
- type: `mound_site`

2. [Geolocation of unpublished archaeological sites in the Peruvian Amazon](https://www.nature.com/articles/s41597-021-01067-7)

- Comes with an excel sheet of locations! https://springernature.figshare.com/articles/dataset/Archaeological_sites_in_the_Department_of_Loreto_Peruvian_Amazon/13547684?backTo=%2Fcollections%2FGeolocation_of_unpublished_archaeological_sites_in_the_Peruvian_Amazon%2F5262530&file=25997585
- Extracted as csv, cleaned it up so that headers only take 1 row!
- Saved as `data/raw/coomes.csv`
- dataset_id: `coomes`
- type: `other`
  -- As the research paper explicitly states: "Limited information is provided in the database of site characteristics, archaeological tradition or phase, or material culture which would be a valuable supplementary addition through future work."

3. [James Q. Jacobs - ArchaeoBlog](https://www.jqjacobs.net)

- Download `amazon_geoglyphs.kml` from `https://www.jqjacobs.net/blog/`.
- Then I manually converted the kml to geojson. Includes everything in geoglyphs, mound sites (no potenital mound villages or rondonia), and mato grasso folders (except line and outline stuff)
- Saved it as: `data/confirmed/james_q_jacobs.geojson`
- Downloaded on 2025-06-24
- dataset_id: `james_q_jacobs`
- type: one of: `potential_geoglyph`, `confirmed_geoglyph`, `lidar_located_earthwork`, `zanja`, `mound_village`, `mound_sites`

4. [IPHAN Database](http://portal.iphan.gov.br/pagina/detalhes/1701/)

- Brazil's National Historic and Artistic Heritage Institute (IPHAN) has a database of confirmed archaeological sites. We use `Georeferenced Archaeological Sites - Shapefiles` from the [IPHAN](http://portal.iphan.gov.br/pagina/detalhes/1701/) website (shoutout [Dr. WinklerPrins](https://antoinette.winklerprins.us/) for pointing me to this)!
- Download the shapefile, unzip it, and put it in `data/raw`. You should have this: `data/raw/sitios/`
- dataset_id: `iphan`

Unfortunately, the dataset is not labelled and nor do I speak Portuguese. To help figure out what exactly is in any given confirmed site, I decided to use `o3` to classify each site into one of the following categories:

- `high_probability_terra_preta`
- `potential_terra_preta`
- `high_probability_geoglyphs`
- `potential_geoglyphs`
- `high_probability_earthworks`
- `potential_earthworks`
- `other`

(Takes a few minutes, runs 500 sites at a time, make sure your openai api key has appropriate rate limits. Also, this will not be perfectly deterministic, so you may get different results. Expect ~50 `high_probability_terra_preta` sites) Creates a new geojson file called `data/confirmed/iphan_classified.geojson`.

5. [Casarabe culture sites](https://www.nature.com/articles/s41586-022-04780-4#MOESM1)

- They provide a Supplementary Table that contains the coordinates of sites (`Supplementary Table 1 | List of Casarabe culture sites.`).
- I downloaded the docx file, gave it to chatgpt, and asked it to convert it to a csv.
- Saved it as `data/raw/prumers.csv`
- dataset_id: `prumers`
- type: `lidar_located_earthwork`
  -- The lidar study shows stepped platform mounds up to 22 m tall, concentric ramparts and kilometres-long raised canals/causeways—all classic earthwork types. These constructions hosted dwellings, plazas and reservoirs, indicating everyday and hydraulic functions, not just symbolic imagery. The authors of the paper themselves use the vocabulary of “platforms,” “causeways,” “moats” and “earthworks” and never call them geoglyphs.

6. [Pre-Columbian earth-builders settled along the entire southern rim of the Amazon](https://www.nature.com/articles/s41467-018-03510-7)

- Supplementary Information accompanies this paper at `https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-03510-7/MediaObjects/41467_2018_3510_MOESM1_ESM.pdf`
- Supplementary Table 4. Archaeological sites identified in the Upper Tapajós Basin.
- Downloaded as pdf, and parsed as csv with chatgpt. Careful, the longitudes and latitudes are flipped, chatgpt was helpful!
- Saved as `data/raw/souza.csv`
- dataset_id: `souza`
- type: `lidar_located_earthwork`

7. [Predicting the geographic distribution of ancient Amazonian archaeological sites with machine learning](https://peerj.com/articles/15137/)

- Under Data Availability: `https://zenodo.org/records/7651334`
- Unzip it, extract the submit.csv
- Saved it as `data/raw/walker.csv`
- dataset_id: `walker`
- type: `earthwork` or `ade` (there is also type other but we discard it!)

8. [Data from: More than 10,000 pre-Columbian earthworks are still hidden throughout Amazonia](https://zenodo.org/records/10214943)

- Download the zip file, extract it, and save the csv as `data/raw/carolina/`
- dataset_id: `carolina`
- we are only interested in the `Earthworks.rds` file, so make sure it is in the `data/raw/carolina/Earthworks.rds`
- type: `earthwork`

9. [Pre-LBA RADAMBRASIL Project Data](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=941)

- 1,153 government-validated soil profiles with zero terra preta contamination across the Amazon basin. Latossolo (Oxisols) soils have very low fertility, poor water retention, and minimal organic matter - opposite of terra preta characteristics. Most common Amazon soil type (~40% of basin) with predictable drought stress response for reliable comparisons.
- We use this as "control" for the terra preta evals.
- Download the data from: `https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=941`, unzip it, within the `data` subfolder get the `Soil_Profiles_Amazonia.csv` file
- Saved it as `data/raw/radambrasil.csv`
- dataset_id: `radambrasil`
- type: `terra_preta_control`

10. [A “Dirty” Footprint: Soil macrofauna biodiversity and fertility in Amazonian Dark Earths and adjacent soils](https://doi.org/10.5061/dryad.3tx95x6cc)

- 225 confirmed ADE sites, downloaded excel sheet and took columns where Soil was ADE in Coordinates Table. Donwloaded a csv and then converted to geojson. 45 unique geographic locations, it had 5 identical copies of each measurement at every location (same coordinates AND same elevation values for all 5 copies).
- Saved as `data/confirmed/wilian.geojson`
- dataset_id: `wilian`
- type: `terra_preta`

### LIDAR Tiles

Since we are dealing with gigabytes of data, I decided to use [Modal](https://modal.com/) to massively parallelize processing and downloading. It is free to use (their credits are very generous and more than enough).

Now, lets download and process our LIDAR datasets. We have a total of 4789 tiles!

1. [LiDAR Surveys over Selected Forest Research Sites, Brazilian Amazon, 2008-2018](https://daac.ornl.gov/CMS/guides/LiDAR_Forest_Inventory_Brazil.html)

- Retrieve the csv `cms_brazil_lidar_tile_inventory.csv` from their Data Files and put it in `data/raw/`
- `dataset_id`: `cms`
- Please order the complete dataset from the website, get the necessary link and put it in `src/scripts/lidar_tiles/cms_download.py` as the `BASE_URL` variable.

2. We downloaded 11 more datasets manually. You can find information about them in `data/lidar_tiles/others_metadata.json`

- We download them and put them all in r2! under keys: `dataset_id/filename/filename`
- There can be subfolders but that doesnt matter since the way we create our geojson is by reading all the las and laz files in the r2 bucket for this! For example: `Keller_Batistella_Gorgens/JAR_A01_ID27_L1_C01.laz/JAR_A01_ID27_L1_C01.laz`
- We then manually processed all these newly added keys in our R2 to determine their bounds. We created a geojson out of this and saved the file in `data/lidar_tiles/others.geojson`

3. French Guiana LIDAR from Nov 2019

- [Aerial LiDAR data from French Guiana, Paracou, November 2019](https://catalogue.ceda.ac.uk/uuid/1d554ff41c104491ac3661c6f6f52aab/)
- [Aerial LiDAR data from French Guiana, Nouragues, November 2019](https://catalogue.ceda.ac.uk/uuid/7bdc5bfc06264802be34f918597150e8/)

To get metadata information on how to download this:
On their webpages, press download, select the folder with the data, view bulk download info, and click on the json listing. Copy paste this into `paracou_metadata.json` in `data/lidar_tile/paracou_metadata,json` and `data/lidar_tiles/nouragues_metadata.json`

### Processing the datasets

1. To process the confirmed site datasets, run the following command:

```bash
uv run -m src.scripts.confirmed
```

2. To process the lidar tiles, run the following command:

```bash
uv run -m src.scripts.lidar_tiles.cms
```

```bash
uv run -m src.scripts.lidar_tiles.cms_download
```

```bash
uv run -m src.scripts.lidar_tiles.paracou_nouragues
```

```bash
uv run -m src.scripts.lidar_tiles.process_zscore
```

```bash
uv run -m src.scripts.lidar_tiles.prepare_eval_data
```
