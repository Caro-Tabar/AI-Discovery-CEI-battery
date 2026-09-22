"""Resolve and filter the CEI-Scout candidate source inventory."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

SOURCE_PATH = Path("data/processed/candidate_sources_v0.csv")
POLICY_PATH = Path("configs/candidate_registry.yaml")
OUTPUT_PATH = Path("data/interim/candidate_registry_resolved_v0.csv")

POLICY_VERSION = "cei-scout-candidate-standardization-v1"
ALLOWED_ELEMENTS = frozenset({"C", "H", "N", "O", "S"})
MIN_MOLECULAR_WEIGHT = 70.0
MAX_MOLECULAR_WEIGHT = 220.0
MAX_ROTATABLE_BONDS = 8

PUBCHEM_ENDPOINT = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/"
    "Title,SMILES,ConnectivitySMILES,InChI,InChIKey,MolecularFormula/JSON"
)

EXCLUDED_MOTIF_SMARTS = {
    "peroxide": "[O;X2]-[O;X2]",
    "azide": "[$([N-]=[N+]=N),$(N=[N+]=[N-])]",
    "diazo": "[#6]=[N+]=[N-]",
    "nitrate_ester": "[O;X2]-[N+](=O)[O-]",
}

OUTPUT_FIELDS = [
    "source_entry_id",
    "preferred_name",
    "candidate_family",
    "source_name",
    "source_identifier",
    "source_url",
    "retrieval_date",
    "source_version",
    "identity_status",
    "pubchem_cid",
    "pubchem_title",
    "pubchem_smiles",
    "pubchem_connectivity_smiles",
    "pubchem_inchi",
    "pubchem_inchikey",
    "canonical_smiles",
    "standardized_inchi",
    "standardized_inchikey",
    "connectivity_key",
    "molecular_formula",
    "molecular_weight_g_mol",
    "elements",
    "formal_charge",
    "radical_electrons",
    "fragment_count",
    "rotatable_bonds",
    "has_peroxide",
    "has_azide",
    "has_diazo",
    "has_nitrate_ester",
    "rdkit_version",
    "standardization_policy_version",
    "policy_source_sha256",
    "resolution_status",
    "filter_status",
    "rejection_reasons",
]


def read_source_rows(path: Path) -> list[dict[str, str]]:
    """Read and validate the immutable source-lead table."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 64:
        raise ValueError(f"Expected 64 source rows; found {len(rows)}")

    source_ids = [row["source_entry_id"].strip() for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Source-entry IDs are not unique")

    return sorted(rows, key=lambda row: row["source_entry_id"])


def extract_cid(row: dict[str, str]) -> str:
    """Extract the numeric PubChem CID from a source identifier."""
    cid = "".join(
        character for character in row["source_identifier"] if character.isdigit()
    )
    if not cid:
        raise ValueError(f"No numeric CID in {row['source_entry_id']}")
    return cid


def fetch_pubchem(
    cids: list[str],
) -> tuple[dict[str, dict[str, Any]], str]:
    """Retrieve PubChem properties in one POST request with bounded retries."""
    encoded = urlencode({"cid": ",".join(cids)}).encode("ascii")
    last_error = "unknown PubChem error"

    for attempt in range(3):
        request = Request(
            PUBCHEM_ENDPOINT,
            data=encoded,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": ("cei-scout/0.1.0 (research registry resolution)"),
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=60) as response:
                payload = json.load(response)

            properties = payload["PropertyTable"]["Properties"]

            return {str(item["CID"]): item for item in properties}, ""

        except (
            HTTPError,
            URLError,
            TimeoutError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

            if attempt < 2:
                time.sleep(2**attempt)

    return {}, last_error


def empty_output_row(
    source: dict[str, str],
    policy_hash: str,
) -> dict[str, str]:
    """Create an output row while preserving every source field."""
    row = {field: "" for field in OUTPUT_FIELDS}

    for field in source:
        if field in row:
            row[field] = source[field]

    row["rdkit_version"] = rdBase.rdkitVersion
    row["standardization_policy_version"] = POLICY_VERSION
    row["policy_source_sha256"] = policy_hash

    return row


def compile_motifs() -> dict[str, Chem.Mol]:
    """Compile and validate the registered exclusion SMARTS."""
    motifs: dict[str, Chem.Mol] = {}

    for name, smarts in EXCLUDED_MOTIF_SMARTS.items():
        query = Chem.MolFromSmarts(smarts)

        if query is None:
            raise ValueError(f"Invalid exclusion SMARTS for {name}: {smarts}")

        motifs[name] = query

    return motifs


def resolve_row(
    source: dict[str, str],
    record: dict[str, Any] | None,
    pubchem_error: str,
    motifs: dict[str, Chem.Mol],
    policy_hash: str,
) -> dict[str, str]:
    """Resolve, standardize, describe, and initially filter one row."""
    row = empty_output_row(source, policy_hash)
    cid = extract_cid(source)
    row["pubchem_cid"] = cid

    if record is None:
        row["resolution_status"] = "failed"
        row["filter_status"] = "rejected"
        reason = pubchem_error or "CID omitted from PubChem response"
        row["rejection_reasons"] = f"pubchem_resolution_failed:{reason}"
        return row

    row["pubchem_title"] = str(record.get("Title", ""))
    row["pubchem_smiles"] = str(record.get("SMILES", ""))
    row["pubchem_connectivity_smiles"] = str(record.get("ConnectivitySMILES", ""))
    row["pubchem_inchi"] = str(record.get("InChI", ""))
    row["pubchem_inchikey"] = str(record.get("InChIKey", ""))

    smiles = row["pubchem_smiles"] or row["pubchem_connectivity_smiles"]

    if not smiles:
        row["resolution_status"] = "failed"
        row["filter_status"] = "rejected"
        row["rejection_reasons"] = "pubchem_record_missing_smiles"
        return row

    try:
        molecule = Chem.MolFromSmiles(smiles)

        if molecule is None:
            raise ValueError("RDKit could not parse the PubChem SMILES")

        fragment_count = len(Chem.GetMolFrags(molecule))
        cleaned = rdMolStandardize.Cleanup(molecule)

        enumerator = rdMolStandardize.TautomerEnumerator()
        standardized = enumerator.Canonicalize(cleaned)
        Chem.SanitizeMol(standardized)

        canonical_smiles = Chem.MolToSmiles(
            standardized,
            canonical=True,
            isomericSmiles=True,
        )
        standardized_inchi = Chem.MolToInchi(standardized)
        standardized_inchikey = Chem.MolToInchiKey(standardized)

    except (RuntimeError, ValueError) as exc:
        row["resolution_status"] = "failed"
        row["filter_status"] = "rejected"
        row["rejection_reasons"] = f"rdkit_standardization_failed:{exc}"
        return row

    elements = sorted({atom.GetSymbol() for atom in standardized.GetAtoms()})
    disallowed_elements = sorted(set(elements) - ALLOWED_ELEMENTS)
    formal_charge = sum(atom.GetFormalCharge() for atom in standardized.GetAtoms())
    radical_electrons = sum(
        atom.GetNumRadicalElectrons() for atom in standardized.GetAtoms()
    )
    molecular_weight = Descriptors.MolWt(standardized)
    rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(standardized)
    motif_matches = {
        name: standardized.HasSubstructMatch(query) for name, query in motifs.items()
    }

    row.update(
        {
            "canonical_smiles": canonical_smiles,
            "standardized_inchi": standardized_inchi,
            "standardized_inchikey": standardized_inchikey,
            "connectivity_key": standardized_inchikey.split(
                "-",
                maxsplit=1,
            )[0],
            "molecular_formula": (rdMolDescriptors.CalcMolFormula(standardized)),
            "molecular_weight_g_mol": (f"{molecular_weight:.6f}"),
            "elements": ";".join(elements),
            "formal_charge": str(formal_charge),
            "radical_electrons": str(radical_electrons),
            "fragment_count": str(fragment_count),
            "rotatable_bonds": str(rotatable_bonds),
            "has_peroxide": str(motif_matches["peroxide"]).lower(),
            "has_azide": str(motif_matches["azide"]).lower(),
            "has_diazo": str(motif_matches["diazo"]).lower(),
            "has_nitrate_ester": str(motif_matches["nitrate_ester"]).lower(),
            "resolution_status": "resolved",
        }
    )

    reasons: list[str] = []

    if fragment_count != 1:
        reasons.append("mixture_or_multifragment")

    if disallowed_elements:
        reasons.append(f"disallowed_elements:{','.join(disallowed_elements)}")

    if molecular_weight < MIN_MOLECULAR_WEIGHT:
        reasons.append("molecular_weight_below_70")

    if molecular_weight > MAX_MOLECULAR_WEIGHT:
        reasons.append("molecular_weight_above_220")

    if formal_charge != 0:
        reasons.append("nonzero_formal_charge")

    if radical_electrons != 0:
        reasons.append("radical_electrons_present")

    if rotatable_bonds > MAX_ROTATABLE_BONDS:
        reasons.append("rotatable_bonds_above_8")

    reasons.extend(
        f"excluded_motif:{name}" for name, matched in motif_matches.items() if matched
    )

    row["filter_status"] = "eligible_prebalance" if not reasons else "rejected"
    row["rejection_reasons"] = ";".join(reasons)

    return row


def flag_duplicates(rows: list[dict[str, str]]) -> None:
    """Reject later standardized-InChIKey duplicates."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        inchikey = row["standardized_inchikey"]

        if inchikey:
            groups[inchikey].append(row)

    for duplicate_rows in groups.values():
        duplicate_rows.sort(key=lambda row: row["source_entry_id"])
        keeper = duplicate_rows[0]["source_entry_id"]

        for duplicate in duplicate_rows[1:]:
            reason = f"duplicate_standardized_inchikey:{keeper}"
            existing = duplicate["rejection_reasons"]

            duplicate["rejection_reasons"] = ";".join(
                item for item in (existing, reason) if item
            )
            duplicate["filter_status"] = "rejected"


def write_rows(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Write atomically to avoid leaving partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(path)


def main() -> None:
    """Build the P05-11 through P05-15 interim registry."""
    source_rows = read_source_rows(SOURCE_PATH)
    policy_hash = hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()

    cids = [extract_cid(row) for row in source_rows]
    pubchem_records, pubchem_error = fetch_pubchem(cids)
    motifs = compile_motifs()

    resolved_rows = [
        resolve_row(
            source=row,
            record=pubchem_records.get(extract_cid(row)),
            pubchem_error=pubchem_error,
            motifs=motifs,
            policy_hash=policy_hash,
        )
        for row in source_rows
    ]

    flag_duplicates(resolved_rows)
    write_rows(OUTPUT_PATH, resolved_rows)

    resolution_counts: dict[str, int] = defaultdict(int)
    filter_counts: dict[str, int] = defaultdict(int)

    for row in resolved_rows:
        resolution_counts[row["resolution_status"]] += 1
        filter_counts[row["filter_status"]] += 1

    print(f"rows: {len(resolved_rows)}")
    print(f"resolution_status: {dict(sorted(resolution_counts.items()))}")
    print(f"filter_status: {dict(sorted(filter_counts.items()))}")
    print(f"rdkit_version: {rdBase.rdkitVersion}")
    print(f"policy_version: {POLICY_VERSION}")
    print(f"output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
