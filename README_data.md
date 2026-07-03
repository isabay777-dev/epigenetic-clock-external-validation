# DNA Methylation Data Provenance

This directory contains the first reduced data deliverable for Article #3. The
data are real GEO-derived DNA methylation beta values, restricted to CpGs used
by the Hannum-71 and Horvath-353 epigenetic clocks.

## Reproduce

Run from the project root:

```bash
python3 fetch_methylation_data.py
```

Use `--force` to discard and rebuild existing reduced CSV outputs. The script
uses only Python standard-library modules plus the system `curl` command. Large
GEO matrices are downloaded one at a time into `/tmp`, parsed locally for the
requested CpG rows, and deleted immediately after parsing.

## Sources

- GSE40279 series matrix:
  `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/matrix/GSE40279_series_matrix.txt.gz`
- GSE87571 metadata series matrix:
  `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/GSE87571/matrix/GSE87571_series_matrix.txt.gz`
- GSE87571 beta matrices:
  `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/GSE87571/suppl/GSE87571_matrix1of2.txt.gz`
  and
  `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/GSE87571/suppl/GSE87571_matrix2of2.txt.gz`
- Clock CpG lists:
  `https://raw.githubusercontent.com/bio-learn/biolearn/v0.9.1/biolearn/data/Hannum.csv`
  and
  `https://raw.githubusercontent.com/bio-learn/biolearn/v0.9.1/biolearn/data/Horvath1.csv`

Retrieval date for the generated files: 2026-07-01.

## Outputs

- `data/gse40279_clock_cpgs.csv`: 656 samples, 418 matched clock CpG columns.
- `data/gse87571_clock_cpgs.csv`: 732 raw samples, 418 matched clock CpG columns; 729 have nonmissing chronological age and are used by `make_clock.py`.
- `data/facts.md`: audit facts, matched CpG counts, URLs, and missing-data notes.

Each CSV is sample-level wide format:

`dataset, sample_geo_accession, sample_title, age, sex, tissue, <CpG beta columns...>`

## Current Counts

- Hannum clock source list: 71 unique CpGs.
- Horvath clock source list: 353 unique CpGs.
- Union of requested clock CpGs: 418 unique CpGs, because 6 CpGs overlap between clocks.
- GSE40279 matched 71/71 Hannum CpGs and 353/353 Horvath CpGs.
- GSE87571 matched 71/71 Hannum CpGs and 353/353 Horvath CpGs.

No beta values were imputed. GSE40279 has 0 missing beta cells in the reduced
clock-CpG matrix. GSE87571 has 10 explicit `NA` beta cells; see `data/facts.md`
for the affected sample/CpG pairs.
