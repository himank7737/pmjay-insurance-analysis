# Data Instructions

## Source
All data comes from India's **National Family Health Survey (NFHS)**, conducted by IIPS Mumbai
under the DHS Program. The data is **free but requires registration**.

## Download Steps

1. Go to https://dhsprogram.com/data/new-user-registration.cfm
2. Register for free access (approval is usually instant)
3. Request access to:
   - **India 2015-16 (NFHS-4)** — Standard DHS
   - **India 2019-21 (NFHS-5)** — Standard DHS
4. Download the **Household Recode** files in **Stata (.DTA)** format:
   - NFHS-4: `IAHR74DT.zip` → extract `IAHR74FL.DTA` (~3.5 GB)
   - NFHS-5: `IAHR7EDT.zip` → extract `IAHR7EFL.DTA` (~4.8 GB)

## File Placement
Place the `.DTA` files anywhere on your machine, then update the paths
at the top of `scripts/phase1_build_combined.py`:

```python
NFHS4_PATH = r"C:\path\to\IAHR74FL.DTA"
NFHS5_PATH = r"C:\path\to\IAHR7EFL.DTA"
OUTPUT_PATH = r"C:\path\to\nfhs_combined.parquet"
```

## What Gets Generated
Running the pipeline produces two parquet files:
- `nfhs_combined.parquet` — raw merged dataset (~1.24M rows)
- `nfhs_analysis.parquet` — clean analysis-ready dataset (21 columns)

These are excluded from the repo via `.gitignore` because they are large
and can be fully regenerated from the raw DHS files.

## Variable Reference
Key variables used from the Household Recode:

| Variable     | NFHS-4  | NFHS-5  | Description               |
|--------------|---------|---------|---------------------------|
| Any insured  | SH54    | SH71    | Primary outcome variable  |
| RSBY/PM-JAY  | SH55D   | SH72D   | Scheme-specific flag      |
| BPL card     | SH58    | SH75    | Below poverty line card   |
| State        | HV024   | HV024   | State code (1–36)         |
| Urban/rural  | HV025   | HV025   | 1=Urban, 2=Rural          |
| Wealth index | HV270   | HV270   | 1=Poorest to 5=Richest    |
| Religion     | SH34    | SH47    | Household head religion   |
| Caste type   | SH36    | SH49    | SC/ST/OBC/General         |
| HH size      | HV009   | HV009   | Number of members         |
| HH weight    | SHV005  | SHWEIGHT| State household weight    |

Source: NFHS-4 Individual Recode Documentation (IAIR74FL.pdf)
        NFHS-5 Individual Recode Documentation (IAIR7EFL.pdf)
