from __future__ import annotations

import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OM_CSV = REPO_ROOT / "data" / "raw" / "operation-and-maintenance-activity.csv"
DEFAULT_OM_DICT = REPO_ROOT / "data" / "raw" / "operation-and-maintenance-activity-data-dictionary.csv"

DISPLAY_COLUMNS = (
    "event_number",
    "company_name",
    "pipeline_name",
    "pipeline_outside_diameter_mm",
    "pipeline_length_m",
    "commodity_carried",
    "integrity_dig",
    "dig_count",
    "commencement_date",
    "province_territory",
    "nearest_populated_centre",
    "circumstances",
)

NPS_OD_MM: dict[float, str] = {
    60.3: '2"',
    168.3: '6"',
    219.1: '8"',
    273.1: '10"',
    323.9: '12"',
    406.4: '16"',
    508.0: '20"',
    609.6: '24"',
    762.0: '30"',
    863.6: '34"',
    914.4: '36"',
    1066.8: '42"',
    1219.2: '48"',
}


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    number = _parse_float(value)
    if number is None:
        return None
    return int(number)


@dataclass(frozen=True)
class OmRecord:
    event_number: str
    company_name: str
    pipeline_name: str
    pipeline_outside_diameter_mm: float | None
    pipeline_length_m: float | None
    commodity_carried: str
    integrity_dig: str
    dig_count: int | None
    commencement_date: str
    completion_date: str
    province_territory: str
    nearest_populated_centre: str
    circumstances: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> OmRecord:
        normalized = {_normalize_key(k): (v or "").strip() for k, v in row.items()}
        return cls(
            event_number=normalized.get("event_number", ""),
            company_name=normalized.get("company_name", ""),
            pipeline_name=normalized.get("pipeline_name", ""),
            pipeline_outside_diameter_mm=_parse_float(normalized.get("pipeline_outside_diameter")),
            pipeline_length_m=_parse_float(normalized.get("pipeline_length")),
            commodity_carried=normalized.get("commodity_carried", ""),
            integrity_dig=normalized.get("integrity_dig", ""),
            dig_count=_parse_int(normalized.get("dig_count")),
            commencement_date=normalized.get("commencement_date", ""),
            completion_date=normalized.get("completion_date", ""),
            province_territory=normalized.get("province_territory", ""),
            nearest_populated_centre=normalized.get("nearest_populated_centre", ""),
            circumstances=normalized.get("circumstances", ""),
        )

    def to_display_row(self) -> tuple[str, ...]:
        od = self.pipeline_outside_diameter_mm
        od_text = f"{od:.1f}" if od is not None else ""
        nps = od_mm_to_nps_label(od) if od is not None else ""
        if nps and od_text:
            od_text = f"{od_text} ({nps})"
        length = self.pipeline_length_m
        length_text = f"{length:.1f}" if length is not None else ""
        dig = self.dig_count
        dig_text = str(dig) if dig is not None else ""
        circ = self.circumstances
        if len(circ) > 80:
            circ = circ[:77] + "..."
        return (
            self.event_number,
            self.company_name,
            self.pipeline_name,
            od_text,
            length_text,
            self.commodity_carried,
            self.integrity_dig,
            dig_text,
            self.commencement_date,
            self.province_territory,
            self.nearest_populated_centre,
            circ,
        )


def default_paths() -> tuple[Path, Path]:
    return DEFAULT_OM_CSV, DEFAULT_OM_DICT


def load_data_dictionary(path: Path | None = None) -> list[tuple[str, str]]:
    dict_path = path or DEFAULT_OM_DICT
    rows: list[tuple[str, str]] = []
    with dict_path.open(newline="", encoding="latin-1") as handle:
        for row in csv.DictReader(handle):
            term = (row.get("Term ") or row.get("Term") or "").strip()
            desc = (row.get("Description") or "").strip()
            if term:
                rows.append((term, desc))
    return rows


def load_om_activity(path: Path | None = None) -> list[OmRecord]:
    csv_path = path or DEFAULT_OM_CSV
    records: list[OmRecord] = []
    with csv_path.open(newline="", encoding="latin-1") as handle:
        for row in csv.DictReader(handle):
            records.append(OmRecord.from_row(row))
    return records


def filter_records(
    records: Iterable[OmRecord],
    *,
    integrity_only: bool = False,
    has_od: bool = False,
    province: str | None = None,
    company: str | None = None,
    search: str | None = None,
) -> list[OmRecord]:
    out: list[OmRecord] = []
    province_key = province.strip().lower() if province else None
    company_key = company.strip().lower() if company else None
    search_key = search.strip().lower() if search else None

    for record in records:
        if integrity_only and record.integrity_dig.upper() != "YES":
            continue
        if has_od and record.pipeline_outside_diameter_mm is None:
            continue
        if province_key and province_key not in record.province_territory.lower():
            continue
        if company_key and company_key not in record.company_name.lower():
            continue
        if search_key:
            haystack = " ".join(
                [
                    record.event_number,
                    record.company_name,
                    record.pipeline_name,
                    record.nearest_populated_centre,
                    record.circumstances,
                ]
            ).lower()
            if search_key not in haystack:
                continue
        out.append(record)
    return out


def od_mm_to_nps_label(od_mm: float | None) -> str:
    if od_mm is None:
        return ""
    nearest = min(NPS_OD_MM, key=lambda mm: abs(mm - od_mm))
    if abs(nearest - od_mm) <= 2.0:
        return f"NPS {NPS_OD_MM[nearest]}"
    nps = od_mm / 25.4
    return f"~NPS {nps:.1f}\""


def summarize(records: Iterable[OmRecord]) -> dict[str, Any]:
    items = list(records)
    total = len(items)
    if total == 0:
        return {"count": 0}

    integrity = sum(1 for r in items if r.integrity_dig.upper() == "YES")
    with_od = [r.pipeline_outside_diameter_mm for r in items if r.pipeline_outside_diameter_mm is not None]
    with_length = [r.pipeline_length_m for r in items if r.pipeline_length_m is not None]
    dig_counts = [r.dig_count for r in items if r.dig_count is not None]

    od_hist: dict[float, int] = {}
    for od in with_od:
        nearest = min(NPS_OD_MM, key=lambda mm: abs(mm - od))
        bucket = nearest if abs(nearest - od) <= 2.0 else round(od, 1)
        od_hist[bucket] = od_hist.get(bucket, 0) + 1

    provinces: dict[str, int] = {}
    for record in items:
        key = record.province_territory or "(blank)"
        provinces[key] = provinces.get(key, 0) + 1

    commodities: dict[str, int] = {}
    for record in items:
        if record.commodity_carried:
            commodities[record.commodity_carried] = commodities.get(record.commodity_carried, 0) + 1

    summary: dict[str, Any] = {
        "count": total,
        "integrity_dig_yes": integrity,
        "with_outside_diameter": len(with_od),
        "with_pipeline_length": len(with_length),
        "with_dig_count": len(dig_counts),
    }
    if with_od:
        summary["od_mm_median"] = statistics.median(with_od)
        summary["od_mm_mean"] = statistics.mean(with_od)
        summary["od_mm_min"] = min(with_od)
        summary["od_mm_max"] = max(with_od)
        summary["od_mm_top"] = sorted(od_hist.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    if with_length:
        summary["length_m_median"] = statistics.median(with_length)
        summary["length_m_mean"] = statistics.mean(with_length)
    if dig_counts:
        summary["dig_count_median"] = statistics.median(dig_counts)
        summary["dig_count_mean"] = statistics.mean(dig_counts)
    summary["provinces_top"] = sorted(provinces.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    summary["commodities_top"] = sorted(commodities.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    if summary.get("count", 0) == 0:
        return "No records match the current filters."

    lines = [
        f"Rows: {summary['count']}",
        f"Integrity dig = Yes: {summary.get('integrity_dig_yes', 0)}",
        f"With outside diameter: {summary.get('with_outside_diameter', 0)}",
        f"With dig count: {summary.get('with_dig_count', 0)}",
    ]
    if "od_mm_median" in summary:
        lines.append(
            "OD (mm): "
            f"median={summary['od_mm_median']:.1f}, "
            f"mean={summary['od_mm_mean']:.1f}, "
            f"range=[{summary['od_mm_min']:.1f}, {summary['od_mm_max']:.1f}]"
        )
        top_od = summary.get("od_mm_top") or []
        if top_od:
            od_parts = [f"{od_mm:g} mm ({count})" for od_mm, count in top_od]
            lines.append("Top OD values: " + ", ".join(od_parts))
    if "length_m_median" in summary:
        lines.append(
            f"Activity length (m): median={summary['length_m_median']:.1f}, "
            f"mean={summary['length_m_mean']:.1f}"
        )
    if "dig_count_median" in summary:
        lines.append(
            f"Dig count: median={summary['dig_count_median']:.0f}, "
            f"mean={summary['dig_count_mean']:.2f}"
        )
    provinces = summary.get("provinces_top") or []
    if provinces:
        lines.append("Top provinces: " + ", ".join(f"{name} ({count})" for name, count in provinces))
    commodities = summary.get("commodities_top") or []
    if commodities:
        lines.append("Top commodities: " + ", ".join(f"{name} ({count})" for name, count in commodities))
    return "\n".join(lines)


def export_csv(records: Iterable[OmRecord], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(DISPLAY_COLUMNS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "event_number": record.event_number,
                    "company_name": record.company_name,
                    "pipeline_name": record.pipeline_name,
                    "pipeline_outside_diameter_mm": record.pipeline_outside_diameter_mm or "",
                    "pipeline_length_m": record.pipeline_length_m or "",
                    "commodity_carried": record.commodity_carried,
                    "integrity_dig": record.integrity_dig,
                    "dig_count": record.dig_count or "",
                    "commencement_date": record.commencement_date,
                    "province_territory": record.province_territory,
                    "nearest_populated_centre": record.nearest_populated_centre,
                    "circumstances": record.circumstances,
                }
            )
    return path

