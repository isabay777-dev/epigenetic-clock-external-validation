#!/usr/bin/env python3
"""Fetch clock CpG beta values from GEO methylation datasets.

This script downloads one GEO gzipped matrix at a time to a temporary file,
streams that local gzip, and retains only CpGs in the Hannum-71 and
Horvath-353 clock marker sets. Full raw matrices are deleted after parsing.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DATA_DIR = Path("data")
RETRIEVAL_DATE = date.today().isoformat()

CLOCK_SOURCES = {
    "hannum": "https://raw.githubusercontent.com/bio-learn/biolearn/v0.9.1/biolearn/data/Hannum.csv",
    "horvath": "https://raw.githubusercontent.com/bio-learn/biolearn/v0.9.1/biolearn/data/Horvath1.csv",
}

DATASETS = {
    "GSE40279": {
        "series_matrix_url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/matrix/GSE40279_series_matrix.txt.gz",
        "matrix_urls": [
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/matrix/GSE40279_series_matrix.txt.gz"
        ],
        "output": DATA_DIR / "gse40279_clock_cpgs.csv",
        "sample_id_kind": "gsm",
    },
    "GSE87571": {
        "series_matrix_url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/GSE87571/matrix/GSE87571_series_matrix.txt.gz",
        "matrix_urls": [
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/GSE87571/suppl/GSE87571_matrix1of2.txt.gz",
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/GSE87571/suppl/GSE87571_matrix2of2.txt.gz",
        ],
        "output": DATA_DIR / "gse87571_clock_cpgs.csv",
        "sample_id_kind": "title_x",
    },
}


def download_text(url: str, timeout: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def fetch_clock_cpgs() -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    clocks: Dict[str, List[str]] = {}
    for clock_name, url in CLOCK_SOURCES.items():
        text = download_text(url)
        reader = csv.DictReader(io.StringIO(text))
        cpgs = [row["CpGmarker"].strip() for row in reader if row.get("CpGmarker")]
        if len(cpgs) != len(set(cpgs)):
            raise RuntimeError(f"Duplicate CpG IDs found in {clock_name} source: {url}")
        clocks[clock_name] = cpgs
    return clocks, CLOCK_SOURCES.copy()


def ordered_union(*lists: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for values in lists:
        for value in values:
            if value not in seen:
                seen.add(value)
                out.append(value)
    return out


def csv_fields(line: str) -> List[str]:
    return next(csv.reader([line], delimiter="\t"))


def open_gzip_url(url: str):
    response = urllib.request.urlopen(url, timeout=120)
    return response, gzip.GzipFile(fileobj=response)


def download_matrix_to_temp(url: str) -> Path:
    """Download a gzipped matrix to /tmp with retries; caller deletes it."""
    handle = tempfile.NamedTemporaryFile(
        prefix="geo_matrix_", suffix=".txt.gz", dir="/tmp", delete=False
    )
    tmp_path = Path(handle.name)
    handle.close()
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "60",
        "-o",
        str(tmp_path),
        url,
    ]
    print(f"Downloading temporary matrix: {url}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path


def parse_series_metadata(url: str) -> Tuple[List[dict], Dict[str, str]]:
    """Parse GEO series/sample metadata before the matrix table."""
    sample_lines: Dict[str, List[List[str]]] = defaultdict(list)
    series_fields: Dict[str, str] = {}

    response, gz = open_gzip_url(url)
    try:
        for raw in gz:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Series_"):
                fields = csv_fields(line)
                key = fields[0].lstrip("!")
                series_fields.setdefault(key, " | ".join(fields[1:]))
            elif line.startswith("!Sample_"):
                fields = csv_fields(line)
                key = fields[0]
                sample_lines[key].append(fields[1:])
    finally:
        gz.close()
        response.close()

    accessions = single_sample_line(sample_lines, "!Sample_geo_accession")
    titles = single_sample_line(sample_lines, "!Sample_title")
    if len(accessions) != len(titles):
        raise RuntimeError(f"Metadata length mismatch: {len(accessions)} accessions vs {len(titles)} titles")

    characteristics = sample_lines.get("!Sample_characteristics_ch1", [])
    char_by_sample: List[dict] = [dict() for _ in accessions]
    for char_line in characteristics:
        if len(char_line) != len(accessions):
            raise RuntimeError("Sample_characteristics_ch1 length does not match sample count")
        for i, raw_value in enumerate(char_line):
            if ":" not in raw_value:
                continue
            key, value = raw_value.split(":", 1)
            char_by_sample[i][key.strip().lower()] = value.strip()

    records = []
    for i, accession in enumerate(accessions):
        title = titles[i]
        chars = char_by_sample[i]
        age = chars.get("age") or chars.get("age (y)") or parse_age_from_title(title)
        sex = chars.get("gender") or chars.get("sex") or ""
        records.append(
            {
                "sample_geo_accession": accession,
                "sample_title": title,
                "matrix_sample_id": parse_matrix_sample_id(title),
                "age": age,
                "sex": normalize_sex(sex),
                "tissue": chars.get("tissue", ""),
            }
        )
    return records, series_fields


def single_sample_line(sample_lines: Dict[str, List[List[str]]], key: str) -> List[str]:
    lines = sample_lines.get(key, [])
    if len(lines) != 1:
        raise RuntimeError(f"Expected exactly one {key} line, found {len(lines)}")
    return lines[0]


def parse_age_from_title(title: str) -> str:
    match = re.search(r"\bage\s+(\d+(?:\.\d+)?)\s*y?\b", title, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def parse_matrix_sample_id(title: str) -> str:
    match = re.match(r"(X\d+)\b", title)
    return match.group(1) if match else ""


def normalize_sex(value: str) -> str:
    value = value.strip()
    lower = value.lower()
    if lower in {"f", "female"}:
        return "female"
    if lower in {"m", "male"}:
        return "male"
    return value


def stream_matrix_rows(
    urls: Sequence[str],
    target_cpgs: Sequence[str],
    sample_id_kind: str,
    title_to_gsm: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]], Dict[str, int]]:
    """Return beta values as sample -> CpG -> value, plus CpG source URLs."""
    target_set = set(target_cpgs)
    sample_values: Dict[str, Dict[str, str]] = defaultdict(dict)
    cpg_sources: Dict[str, List[str]] = defaultdict(list)
    stats = {"matrix_rows_scanned": 0, "target_rows_found": 0}

    for url in urls:
        print(f"Preparing matrix: {url}", flush=True)
        tmp_path = download_matrix_to_temp(url)
        gz = gzip.open(tmp_path, "rb")
        header: List[str] | None = None
        beta_column_indexes: List[int] = []
        beta_sample_ids: List[str] = []
        in_series_table = False
        last_progress = time.time()
        try:
            for raw in gz:
                stripped = raw.rstrip(b"\n")
                if not stripped:
                    continue

                if stripped.startswith(b"!series_matrix_table_begin"):
                    in_series_table = True
                    continue
                if stripped.startswith(b"!series_matrix_table_end"):
                    break

                if header is None:
                    if stripped.startswith(b"!"):
                        continue
                    header_line = stripped.decode("utf-8", errors="replace")
                    header = [strip_quotes(field) for field in csv_fields(header_line)]
                    beta_column_indexes, beta_sample_ids = select_beta_columns(
                        header, sample_id_kind, title_to_gsm
                    )
                    continue

                if sample_id_kind == "gsm" and not in_series_table:
                    continue

                tab_index = stripped.find(b"\t")
                if tab_index == -1:
                    continue
                row_id = strip_quotes(stripped[:tab_index].decode("utf-8", errors="replace"))
                stats["matrix_rows_scanned"] += 1
                if row_id not in target_set:
                    now = time.time()
                    if now - last_progress > 30:
                        print(
                            f"  scanned {stats['matrix_rows_scanned']:,} matrix rows; "
                            f"found {stats['target_rows_found']} target rows",
                            flush=True,
                        )
                        last_progress = now
                    continue

                row = [strip_quotes(field) for field in csv_fields(stripped.decode("utf-8", errors="replace"))]
                for col_index, sample_id in zip(beta_column_indexes, beta_sample_ids):
                    if col_index < len(row):
                        sample_values[sample_id][row_id] = row[col_index]
                cpg_sources[row_id].append(url)
                stats["target_rows_found"] += 1
                print(f"  found target CpG {row_id}", flush=True)
        finally:
            gz.close()
            try:
                tmp_path.unlink()
                print(f"Deleted temporary matrix: {tmp_path}", flush=True)
            except OSError as exc:
                print(f"WARNING: could not delete temporary matrix {tmp_path}: {exc}", file=sys.stderr)

    return sample_values, cpg_sources, stats


def strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def select_beta_columns(
    header: Sequence[str], sample_id_kind: str, title_to_gsm: Dict[str, str]
) -> Tuple[List[int], List[str]]:
    if not header or header[0] != "ID_REF":
        raise RuntimeError(f"Unexpected matrix header: first field is {header[0] if header else '<empty>'}")

    indexes: List[int] = []
    sample_ids: List[str] = []
    for i, column in enumerate(header[1:], start=1):
        if sample_id_kind == "gsm":
            if column.startswith("GSM"):
                indexes.append(i)
                sample_ids.append(column)
        elif sample_id_kind == "title_x":
            # GSE87571 supplementary matrices include paired Xn and Xn.1 columns.
            # The bare Xn columns contain beta values; Xn.1 columns are not retained.
            if re.fullmatch(r"X\d+", column):
                indexes.append(i)
                try:
                    sample_ids.append(title_to_gsm[column])
                except KeyError as exc:
                    raise RuntimeError(f"Matrix column {column} is absent from sample metadata") from exc
        else:
            raise ValueError(f"Unknown sample_id_kind: {sample_id_kind}")

    if not indexes:
        raise RuntimeError("No beta-value sample columns found in matrix header")
    return indexes, sample_ids


def is_missing(value: str) -> bool:
    return value == "" or value.upper() in {"NA", "NAN", "NULL"}


def write_dataset_csv(
    output_path: Path,
    dataset: str,
    metadata: Sequence[dict],
    sample_values: Dict[str, Dict[str, str]],
    ordered_cpgs: Sequence[str],
) -> Dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "sample_geo_accession", "sample_title", "age", "sex", "tissue"] + list(ordered_cpgs)
    missing_cells = 0
    written_samples = 0
    samples_with_any_beta = 0

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in metadata:
            gsm = record["sample_geo_accession"]
            values = sample_values.get(gsm, {})
            if values:
                samples_with_any_beta += 1
            row = {
                "dataset": dataset,
                "sample_geo_accession": gsm,
                "sample_title": record["sample_title"],
                "age": record["age"],
                "sex": record["sex"],
                "tissue": record["tissue"],
            }
            for cpg in ordered_cpgs:
                value = values.get(cpg, "")
                if is_missing(value):
                    missing_cells += 1
                row[cpg] = value
            writer.writerow(row)
            written_samples += 1

    return {
        "written_samples": written_samples,
        "samples_with_any_beta": samples_with_any_beta,
        "missing_beta_cells": missing_cells,
    }


def write_facts(
    facts_path: Path,
    summaries: Dict[str, dict],
    clock_sources: Dict[str, str],
    clocks: Dict[str, List[str]],
) -> None:
    facts_path.parent.mkdir(parents=True, exist_ok=True)
    total_clock_union = len(ordered_union(clocks["hannum"], clocks["horvath"]))
    clock_overlap = len(set(clocks["hannum"]) & set(clocks["horvath"]))
    lines = [
        "# Methylation Data Fetch Facts",
        "",
        f"Retrieval date: {RETRIEVAL_DATE}",
        "",
        "## Clock CpG Sources",
        "",
        f"- Hannum clock source: {clock_sources['hannum']} ({len(clocks['hannum'])} unique CpGs)",
        f"- Horvath clock source: {clock_sources['horvath']} ({len(clocks['horvath'])} unique CpGs)",
        f"- Union of requested clock CpGs: {total_clock_union}; overlap between clock lists: {clock_overlap}",
        "",
        "## GEO Downloads",
        "",
    ]

    for dataset, summary in summaries.items():
        missing_hannum = summary["hannum_missing"]
        missing_horvath = summary["horvath_missing"]
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"- Series metadata URL: {summary['series_matrix_url']}",
                "- Beta matrix URL(s):",
            ]
        )
        lines.extend([f"  - {url}" for url in summary["matrix_urls"]])
        lines.extend(
            [
                f"- Samples in GEO metadata: {summary['metadata_samples']}",
                f"- Samples written to reduced CSV: {summary['written_samples']}",
                f"- Samples with at least one retained beta value: {summary['samples_with_any_beta']}",
                f"- Samples with age metadata: {summary['samples_with_age']}",
                f"- Samples with sex metadata: {summary['samples_with_sex']}",
                f"- Hannum CpGs matched: {summary['hannum_matched']} / {len(clocks['hannum'])}",
                f"- Horvath CpGs matched: {summary['horvath_matched']} / {len(clocks['horvath'])}",
                f"- Requested CpG union matched: {summary['union_matched']} / {summary['union_requested']}",
                f"- Missing beta cells in reduced CSV: {summary['missing_beta_cells']}",
                f"- Matrix rows scanned: {summary['matrix_rows_scanned']}",
                f"- Reduced CSV: {summary['output_path']} ({summary['output_size_bytes']} bytes)",
            ]
        )
        if missing_hannum:
            lines.append(f"- Hannum CpGs not found: {', '.join(missing_hannum)}")
        if missing_horvath:
            lines.append(f"- Horvath CpGs not found: {', '.join(missing_horvath)}")
        if summary.get("notes"):
            lines.extend(["- Notes:"] + [f"  - {note}" for note in summary["notes"]])
        if summary.get("missing_beta_entries"):
            lines.append("- Missing beta entries:")
            lines.extend(
                [
                    f"  - {sample}: {cpg}"
                    for sample, cpg in summary["missing_beta_entries"]
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## Missing-Data Handling",
            "",
            "No beta values are imputed or simulated. Empty, `NA`, `NaN`, and `NULL` beta entries are retained as empty/missing values in the reduced CSVs and counted above.",
            "",
            "Full raw GEO matrix files are downloaded one at a time to `/tmp`, parsed for the requested CpG rows, and deleted immediately after parsing. They are not retained in this project.",
            "",
        ]
    )
    facts_path.write_text("\n".join(lines), encoding="utf-8")


def summarize_existing_csv(
    dataset: str,
    cfg: dict,
    clocks: Dict[str, List[str]],
    target_cpgs: Sequence[str],
) -> dict:
    output_path = cfg["output"]
    with output_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Existing output has no header: {output_path}")
        cpg_columns = [field for field in reader.fieldnames if field in set(target_cpgs)]
        matched_union = set(cpg_columns)
        missing_beta_cells = 0
        missing_beta_entries: List[Tuple[str, str]] = []
        written_samples = 0
        samples_with_any_beta = 0
        samples_with_age = 0
        samples_with_sex = 0
        for row in reader:
            written_samples += 1
            samples_with_age += bool(row.get("age", ""))
            samples_with_sex += bool(row.get("sex", ""))
            has_any = False
            for cpg in cpg_columns:
                value = row.get(cpg, "")
                if is_missing(value):
                    missing_beta_cells += 1
                    missing_beta_entries.append((row.get("sample_geo_accession", ""), cpg))
                else:
                    has_any = True
            samples_with_any_beta += has_any

    hannum_set = set(clocks["hannum"])
    horvath_set = set(clocks["horvath"])
    return {
        "series_matrix_url": cfg["series_matrix_url"],
        "matrix_urls": cfg["matrix_urls"],
        "metadata_samples": written_samples,
        "written_samples": written_samples,
        "samples_with_any_beta": samples_with_any_beta,
        "samples_with_age": samples_with_age,
        "samples_with_sex": samples_with_sex,
        "hannum_matched": len(matched_union & hannum_set),
        "horvath_matched": len(matched_union & horvath_set),
        "hannum_missing": [cpg for cpg in clocks["hannum"] if cpg not in matched_union],
        "horvath_missing": [cpg for cpg in clocks["horvath"] if cpg not in matched_union],
        "union_matched": len(matched_union),
        "union_requested": len(target_cpgs),
        "missing_beta_cells": missing_beta_cells,
        "matrix_rows_scanned": "not rescanned; summary derived from existing reduced CSV",
        "output_path": str(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "notes": existing_dataset_notes(dataset),
        "missing_beta_entries": missing_beta_entries,
    }


def existing_dataset_notes(dataset: str) -> List[str]:
    notes = ["Existing reduced CSV was reused because --force was not supplied."]
    if dataset == "GSE87571":
        notes.append(
            "The GEO series matrix is metadata-only (`Sample_data_row_count` = 0); beta values were read from the two GEO supplementary matrix files listed above. Bare `Xn` columns were retained; paired `Xn.1` columns were ignored because they are not beta-value columns."
        )
    return notes


def existing_outputs_complete(paths: Iterable[Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Hannum/Horvath clock CpG beta values from GSE40279 and GSE87571."
    )
    parser.add_argument("--force", action="store_true", help="Re-fetch and overwrite existing reduced CSV outputs.")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    facts_path = DATA_DIR / "facts.md"

    clocks, clock_sources = fetch_clock_cpgs()
    target_cpgs = ordered_union(clocks["hannum"], clocks["horvath"])
    union_requested = len(target_cpgs)
    summaries: Dict[str, dict] = {}

    for dataset, cfg in DATASETS.items():
        output_path = cfg["output"]
        if output_path.exists() and not args.force:
            print(f"Reusing existing {dataset} output: {output_path}", flush=True)
            summaries[dataset] = summarize_existing_csv(dataset, cfg, clocks, target_cpgs)
            continue

        print(f"Parsing metadata for {dataset}", flush=True)
        metadata, _series_fields = parse_series_metadata(cfg["series_matrix_url"])
        title_to_gsm = {
            record["matrix_sample_id"]: record["sample_geo_accession"]
            for record in metadata
            if record["matrix_sample_id"]
        }

        sample_values, cpg_sources, matrix_stats = stream_matrix_rows(
            cfg["matrix_urls"], target_cpgs, cfg["sample_id_kind"], title_to_gsm
        )
        matched_union = set(cpg_sources)
        ordered_matched_cpgs = [cpg for cpg in target_cpgs if cpg in matched_union]
        write_stats = write_dataset_csv(output_path, dataset, metadata, sample_values, ordered_matched_cpgs)

        hannum_set = set(clocks["hannum"])
        horvath_set = set(clocks["horvath"])
        notes = []
        if dataset == "GSE87571":
            notes.append(
                "The GEO series matrix is metadata-only (`Sample_data_row_count` = 0); beta values were read from the two GEO supplementary matrix files listed above. Bare `Xn` columns were retained; paired `Xn.1` columns were ignored because they are not beta-value columns."
            )

        summaries[dataset] = {
            "series_matrix_url": cfg["series_matrix_url"],
            "matrix_urls": cfg["matrix_urls"],
            "metadata_samples": len(metadata),
            "written_samples": write_stats["written_samples"],
            "samples_with_any_beta": write_stats["samples_with_any_beta"],
            "samples_with_age": sum(bool(record["age"]) for record in metadata),
            "samples_with_sex": sum(bool(record["sex"]) for record in metadata),
            "hannum_matched": len(matched_union & hannum_set),
            "horvath_matched": len(matched_union & horvath_set),
            "hannum_missing": [cpg for cpg in clocks["hannum"] if cpg not in matched_union],
            "horvath_missing": [cpg for cpg in clocks["horvath"] if cpg not in matched_union],
            "union_matched": len(matched_union),
            "union_requested": union_requested,
            "missing_beta_cells": write_stats["missing_beta_cells"],
            "matrix_rows_scanned": matrix_stats["matrix_rows_scanned"],
            "output_path": str(output_path),
            "output_size_bytes": output_path.stat().st_size,
            "notes": notes,
        }

    if len(summaries) != len(DATASETS):
        # Preserve existing facts if a partial non-forced run skipped a dataset.
        print("Not rewriting facts.md because this run did not process all datasets.", file=sys.stderr)
        return 1

    write_facts(facts_path, summaries, clock_sources, clocks)
    print(f"Wrote {facts_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nERROR:", exc, file=sys.stderr)
        print(textwrap.dedent(
            """
            No synthetic or substitute data were generated. Inspect the error above,
            then re-run with --force after fixing the download or parsing issue if needed.
            """
        ).strip(), file=sys.stderr)
        raise
