"""Create the human-review report for candidate registry version 1."""

from __future__ import annotations

import argparse
import html
import runpy
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

POOL_PATH = Path("data/processed/candidates_v1.parquet")
SOURCE_PATH = Path("data/interim/candidate_registry_resolved_v0.csv")
CHECKLIST_PATH = Path("data/processed/candidate_review_v1.csv")
BENCHMARK_PATH = Path("data/processed/benchmark_registry_v1.csv")
REPORT_PATH = Path("results/reports/candidate_pool_review.html")


def text(value: object) -> str:
    """Return an HTML-safe display value."""
    return html.escape(str(value))


def normalize_cid(value: object) -> int:
    """Normalize a PubChem CID read from CSV or Parquet."""
    return int(float(str(value)))


def molecule_svg(smiles: str) -> str:
    """Render one molecule as an embeddable SVG."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse canonical SMILES: {smiles}")

    drawer = rdMolDraw2D.MolDraw2DSVG(320, 220)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.index("<svg") :]


def build_benchmark_registry(
    source: pd.DataFrame,
    pool: pd.DataFrame,
    benchmark_cids: dict[int, str],
) -> pd.DataFrame:
    """Tag every preregistered reference as benchmark-only or both."""
    source = source.copy()
    source["normalized_pubchem_cid"] = source["pubchem_cid"].map(normalize_cid)
    selected_keys = set(pool["standardized_inchikey"])
    rows: list[dict[str, object]] = []

    for cid, benchmark_name in benchmark_cids.items():
        matches = source.loc[source["normalized_pubchem_cid"].eq(cid)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one resolved row for benchmark CID {cid}; "
                f"found {len(matches)}"
            )

        row = matches.iloc[0]
        in_candidate_pool = row["standardized_inchikey"] in selected_keys
        rows.append(
            {
                "benchmark_name": benchmark_name,
                "pubchem_cid": cid,
                "preferred_name": row["preferred_name"],
                "candidate_family": row["candidate_family"],
                "canonical_smiles": row["canonical_smiles"],
                "standardized_inchikey": row["standardized_inchikey"],
                "registry_role": "both" if in_candidate_pool else "benchmark_only",
                "source_name": row["source_name"],
                "source_url": row["source_url"],
            }
        )

    return pd.DataFrame(rows)


def build_checklist(pool: pd.DataFrame) -> pd.DataFrame:
    """Create or refresh the review checklist without losing prior decisions."""
    previous: dict[str, dict[str, str]] = {}

    if CHECKLIST_PATH.exists():
        old = pd.read_csv(CHECKLIST_PATH, dtype=str, keep_default_na=False)
        previous = {
            row["standardized_inchikey"]: row.to_dict() for _, row in old.iterrows()
        }

    rows: list[dict[str, str]] = []

    for _, candidate in pool.iterrows():
        key = str(candidate["standardized_inchikey"])
        prior = previous.get(key, {})
        rows.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "preferred_name": str(candidate["preferred_name"]),
                "standardized_inchikey": key,
                "review_status": prior.get("review_status", "pending"),
                "review_note": prior.get("review_note", ""),
                "reviewer": prior.get("reviewer", ""),
                "review_date": prior.get("review_date", ""),
            }
        )

    return pd.DataFrame(rows)


def record_pass(
    checklist: pd.DataFrame,
    *,
    reviewer: str,
    review_date: str,
) -> pd.DataFrame:
    """Record review only after the reviewer has inspected all 60 cards."""
    updated = checklist.copy()
    updated["review_status"] = "passed"
    updated["review_note"] = "2D depiction and metadata inspected; no issue observed"
    updated["reviewer"] = reviewer
    updated["review_date"] = review_date
    return updated


def build_report(pool: pd.DataFrame, checklist: pd.DataFrame) -> str:
    """Build a self-contained HTML report containing all 60 structures."""
    status_by_key = checklist.set_index("standardized_inchikey")[
        "review_status"
    ].to_dict()
    cards: list[str] = []

    for _, row in pool.iterrows():
        key = str(row["standardized_inchikey"])
        source_url = html.escape(str(row["source_url"]), quote=True)

        cards.append(
            f"""
            <article class="card">
              <h2>{text(row["candidate_id"])}: {text(row["preferred_name"])}</h2>
              <div class="structure">
                {molecule_svg(str(row["canonical_smiles"]))}
              </div>
              <dl>
                <dt>Review status</dt>
                <dd>{text(status_by_key[key])}</dd>
                <dt>Registry role</dt>
                <dd>{text(row["registry_role"])}</dd>
                <dt>Family</dt>
                <dd>{text(row["candidate_family"])}</dd>
                <dt>Formula</dt>
                <dd>{text(row["molecular_formula"])}</dd>
                <dt>Molecular weight</dt>
                <dd>{text(row["molecular_weight_g_mol"])} g/mol</dd>
                <dt>Rotatable bonds</dt>
                <dd>{text(row["rotatable_bonds"])}</dd>
                <dt>InChIKey</dt>
                <dd>{text(key)}</dd>
                <dt>Canonical SMILES</dt>
                <dd>{text(row["canonical_smiles"])}</dd>
                <dt>Safety flag</dt>
                <dd>{text(row["safety_alert"])}</dd>
                <dt>Source</dt>
                <dd>
                  <a href="{source_url}">{text(row["source_name"])}</a>
                </dd>
              </dl>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CEI-Scout candidate pool review</title>
  <style>
    body {{
      font-family: sans-serif;
      margin: 1rem;
      background: #f5f7fa;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1rem;
    }}
    .card {{
      background: white;
      padding: 1rem;
      border: 1px solid #ccd3dc;
    }}
    h1 {{ margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1rem; min-height: 2.4rem; }}
    .structure {{ text-align: center; }}
    dl {{
      display: grid;
      grid-template-columns: 9rem 1fr;
      gap: 0.3rem;
    }}
    dt {{ font-weight: bold; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    @media (max-width: 850px) {{
      main {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>CEI-Scout candidate pool review</h1>
  <p>Inspect structure, identity, family, formula, mass, source, and safety flag.</p>
  <main>{"".join(cards)}</main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-pass", action="store_true")
    parser.add_argument("--reviewer")
    parser.add_argument("--review-date")
    args = parser.parse_args()

    if args.record_pass and (not args.reviewer or not args.review_date):
        parser.error("--record-pass requires --reviewer and --review-date")

    return args


def main() -> None:
    """Generate benchmark roles, checklist, and visual review report."""
    args = parse_args()
    pool = pd.read_parquet(POOL_PATH)
    source = pd.read_csv(SOURCE_PATH, dtype=str, keep_default_na=False)

    if len(pool) != 60:
        raise ValueError(f"Expected 60 candidates; found {len(pool)}")

    freeze_module = runpy.run_path("scripts/freeze_candidate_pool.py")
    benchmark_cids = freeze_module["BENCHMARK_CIDS"]

    benchmarks = build_benchmark_registry(
        source,
        pool,
        benchmark_cids,
    )
    checklist = build_checklist(pool)

    if args.record_pass:
        checklist = record_pass(
            checklist,
            reviewer=args.reviewer,
            review_date=args.review_date,
        )

    CHECKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    benchmarks.to_csv(
        BENCHMARK_PATH,
        index=False,
        lineterminator="\n",
    )
    checklist.to_csv(
        CHECKLIST_PATH,
        index=False,
        lineterminator="\n",
    )
    REPORT_PATH.write_text(
        build_report(pool, checklist),
        encoding="utf-8",
        newline="\n",
    )

    counts = checklist["review_status"].value_counts().to_dict()
    roles = benchmarks["registry_role"].value_counts().to_dict()

    print(f"candidate_cards: {len(pool)}")
    print(f"review_status_counts: {counts}")
    print(f"benchmark_roles: {roles}")
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
