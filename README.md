

# SPC Student Insights Dashboard

This repository contains a Streamlit dashboard and ETL utilities to explore student enrollment, demographics, and program trends for (SPC).

This README explains how to set up your environment (Windows), run the ETL to produce the curated dataset, and launch the dashboard locally.

---

## Prerequisites

- Python 3.10+ (this project was developed with Python 3.12.3; earlier 3.10+ should work)
- Git (optional)
- Recommended: use a virtual environment

Files you will use:
- `App.py` — Streamlit dashboard application
- `etl/etl_cbmc1.py` — ETL script to merge CBM + student files and write curated Parquet/CSV
- `data/` — place your raw input files under `data/cbmc1/` and `data/stu220/` or update the CLI accordingly
- `data/curated/` — curated outputs (parquet/csv) are written here
- `requirements.txt` — Python dependencies

---

## Setup (Windows PowerShell)

Open PowerShell and run the following commands in the repository root.

1. Create & activate a virtual environment

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

If activation is blocked, run as administrator once to allow the policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then re-run the activation command.

2. Install dependencies

```powershell
pip install -r requirements.txt
```

Note: If you already have a working environment you can skip creating the venv and install into your existing environment. The `requirements.txt` contains the major packages used by the ETL and dashboard (streamlit, pandas, plotly, numpy, pyarrow, openpyxl, xlrd, duckdb, etc.).

---

## Running the ETL (optional)

If you already have a curated Parquet file at `data/curated/data_merged.parquet` you can skip this step and run the dashboard directly. Otherwise, the ETL script will read files from `data/cbmc1/` and `data/stu220/` (or any paths you provide) and write curated outputs.

Example usage (PowerShell):

```powershell
python etl/etl_cbmc1.py \\
	--cbmc_inputs data/cbmc1/"SPR24 XST_CBM0C1_SPC .xlsx" \\
	--stu_inputs data/stu220/"STU0220_Spring 2024_4.11.24.xlsx" \\
	--out_parquet data/curated/data_merged.parquet \\
	--out_csv data/curated/data_merged.csv \\
	--duckdb warehouse/warehouse.duckdb \\
	--duckdb-table cbmc1_merged
```

Notes:
- `--cbmc_inputs` and `--stu_inputs` accept one or more files or directories; directories are scanned for `.csv`, `.xls`, and `.xlsx` files.
- The script will attempt to read Excel files and CSVs robustly and will warn when required columns are missing.
- The ETL writes both Parquet and CSV outputs; Parquet is preferred for speed.

---

## Running the dashboard

From the repository root, run:

PowerShell:
```powershell
.\\.venv\\Scripts\\Activate.ps1  # if not already activated
streamlit run App.py
```

Command Prompt (cmd.exe):
```cmd
.\\.venv\\Scripts\\activate.bat
streamlit run App.py
```

After launching, Streamlit will print a local URL (usually `http://localhost:8501`) you can open in your browser.

The dashboard will attempt to load the curated Parquet at `data/curated/data_merged.parquet` by default. If your curated file is located elsewhere, open `App.py` and update the `load_and_process_data(...)` call to point to the correct file path.

---

## Troubleshooting

- "ModuleNotFoundError": Ensure you're in the virtual environment and `pip install -r requirements.txt` completed successfully.
- Excel reads fail: install `openpyxl` and `xlrd` (already included in `requirements.txt`). Large Excel files can be slow—consider saving as CSV.
- Parquet issues: `pyarrow` or `fastparquet` are required depending on how you read/write Parquet. `pyarrow` is listed in `requirements.txt`.
- Streamlit UI not updating: stop and re-run `streamlit run App.py`. Use `Ctrl+C` in the terminal to stop the server.



## Development tips

- Edit `App.py` to change default file paths or tweak UI text. The file contains helpful comments and caching decorators for performance (`@st.cache_data`).
- The ETL (`etl/etl_cbmc1.py`) is written to be robust to missing columns and varying file types. If you need a custom column mapping, update `COLUMN_MAPPING` in the ETL script.


---

## Optional: Run with Docker

If you'd prefer to run the dashboard in Docker (no local Python install required), a simple `Dockerfile` and `docker-compose.yml` are included.

Build and run with Docker (from repository root):

```powershell
docker build -t spc-dashboard:latest .
docker run -p 8501:8501 -v ${PWD}:/app spc-dashboard:latest
```

Or using Docker Compose:

```powershell
docker-compose up --build
```

Notes:
- The Docker image installs packages from `requirements.txt`. Large dependencies (e.g., pandas, pyarrow) will increase image size and build time.
- The `docker-compose.yml` mounts the repo into the container so code changes are reflected without rebuilding. For production, remove the bind-mount and bake files into the image.



