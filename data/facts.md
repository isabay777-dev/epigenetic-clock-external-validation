# Methylation Data Fetch Facts

Retrieval date: 2026-07-01

## Clock CpG Sources

- Hannum clock source: https://raw.githubusercontent.com/bio-learn/biolearn/v0.9.1/biolearn/data/Hannum.csv (71 unique CpGs)
- Horvath clock source: https://raw.githubusercontent.com/bio-learn/biolearn/v0.9.1/biolearn/data/Horvath1.csv (353 unique CpGs)
- Union of requested clock CpGs: 418; overlap between clock lists: 6

## GEO Downloads

### GSE40279

- Series metadata URL: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/matrix/GSE40279_series_matrix.txt.gz
- Beta matrix URL(s):
  - https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/matrix/GSE40279_series_matrix.txt.gz
- Samples in GEO metadata: 656
- Samples written to reduced CSV: 656
- Samples with at least one retained beta value: 656
- Samples with age metadata: 656
- Samples with sex metadata: 656
- Hannum CpGs matched: 71 / 71
- Horvath CpGs matched: 353 / 353
- Requested CpG union matched: 418 / 418
- Missing beta cells in reduced CSV: 0
- Matrix rows scanned: not rescanned; summary derived from existing reduced CSV
- Reduced CSV: data/gse40279_clock_cpgs.csv (2829299 bytes)
- Notes:
  - Existing reduced CSV was reused because --force was not supplied.

### GSE87571

- Series metadata URL: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/GSE87571/matrix/GSE87571_series_matrix.txt.gz
- Beta matrix URL(s):
  - https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/GSE87571/suppl/GSE87571_matrix1of2.txt.gz
  - https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/GSE87571/suppl/GSE87571_matrix2of2.txt.gz
- Samples in GEO metadata: 732
- Samples written to reduced CSV: 732
- Samples with at least one retained beta value: 732
- Samples with nonmissing age in reduced CSV: 729
- Samples with nonmissing sex in reduced CSV: 730
- Hannum CpGs matched: 71 / 71
- Horvath CpGs matched: 353 / 353
- Requested CpG union matched: 418 / 418
- Missing beta cells in reduced CSV: 10
- Matrix rows scanned: not rescanned; summary derived from existing reduced CSV
- Reduced CSV: data/gse87571_clock_cpgs.csv (5618044 bytes)
- Notes:
  - Existing reduced CSV was reused because --force was not supplied.
  - The GEO series matrix is metadata-only (`Sample_data_row_count` = 0); beta values were read from the two GEO supplementary matrix files listed above. Bare `Xn` columns were retained; paired `Xn.1` columns were ignored because they are not beta-value columns.
  - The reduced CSV contains 732 raw rows; 3 rows have missing chronological age and are excluded by `make_clock.py`, leaving 729 analyzed GSE87571 samples.
- Missing beta entries:
  - GSM2334496: cg11025793
  - GSM2334595: cg11025793
  - GSM2334605: cg10501210
  - GSM2334605: cg17099569
  - GSM2334605: cg00431549
  - GSM2334605: cg11025793
  - GSM2334615: cg17099569
  - GSM2334615: cg11025793
  - GSM2334615: cg14409958
  - GSM2334626: cg11025793

## Missing-Data Handling

No beta values are imputed or simulated. Empty, `NA`, `NaN`, and `NULL` beta entries are retained as empty/missing values in the reduced CSVs and counted above.

Full raw GEO matrix files are downloaded one at a time to `/tmp`, parsed for the requested CpG rows, and deleted immediately after parsing. They are not retained in this project.
