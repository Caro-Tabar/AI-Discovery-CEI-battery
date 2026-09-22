#!/usr/bin/env python3
"""Download the pinned LIBE v2 source file from Figshare."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ARTICLE_ID = 14226464
VERSION = 2
EXPECTED_FILE_NAME = "libe.json"
EXPECTED_LICENSE = "CC BY 4.0"

FIGSHARE_API = "https://api.figshare.com/v2"
VERSION_URL = f"{FIGSHARE_API}/articles/{ARTICLE_ID}/versions/{VERSION}"

USER_AGENT = "cei-scout/0.1 (https://github.com/Caro-Tabar/AI-Discovery-CEI-battery)"

CHUNK_SIZE = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 90
MAX_ATTEMPTS = 3

LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def request_json(url: str) -> dict[str, Any]:
    """Retrieve a JSON object with a small retry policy."""
    request = Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urlopen(
                request,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as response:
                return cast(dict[str, Any], json.load(response))
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Unable to retrieve Figshare metadata from {url}"
                ) from exc

            delay_seconds = 2**attempt
            LOGGER.warning(
                "Metadata request failed on attempt %d/%d; retrying in %d s.",
                attempt,
                MAX_ATTEMPTS,
                delay_seconds,
            )
            time.sleep(delay_seconds)

    raise RuntimeError("Unreachable retry state.")


def get_libe_file_record() -> tuple[dict[str, Any], dict[str, Any], str]:
    """Return pinned article metadata, the LIBE file record, and license name."""
    metadata = request_json(VERSION_URL)

    if metadata.get("version") != VERSION:
        raise RuntimeError(
            f"Requested Figshare version {VERSION}, "
            f"but API returned {metadata.get('version')!r}."
        )

    license_record = metadata.get("license")

    if isinstance(license_record, dict):
        license_name = str(license_record.get("name", ""))
    else:
        license_name = str(license_record or "")

    if license_name != EXPECTED_LICENSE:
        raise RuntimeError(
            f"Expected license {EXPECTED_LICENSE!r}, "
            f"but Figshare returned {license_name!r}."
        )

    files = metadata.get("files")
    if not isinstance(files, list):
        raise RuntimeError("Figshare metadata does not contain a file list.")

    matches = [
        record
        for record in files
        if isinstance(record, dict) and record.get("name") == EXPECTED_FILE_NAME
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {EXPECTED_FILE_NAME!r}; found {len(matches)}."
        )

    return metadata, matches[0], license_name


def download_file(
    download_url: str,
    output: Path,
    expected_size: int,
) -> None:
    """Download one file atomically and verify its byte count."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.part")

    temporary.unlink(missing_ok=True)

    request = Request(download_url, headers={"User-Agent": USER_AGENT})

    LOGGER.info(
        "Downloading %s (%.2f MiB).",
        output,
        expected_size / (1024 * 1024),
    )

    downloaded_bytes = 0
    next_progress_percent = 10

    try:
        with (
            urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response,
            temporary.open("wb") as handle,
        ):
            while chunk := response.read(CHUNK_SIZE):
                handle.write(chunk)
                downloaded_bytes += len(chunk)

                progress_percent = int(downloaded_bytes * 100 / expected_size)

                if progress_percent >= next_progress_percent:
                    LOGGER.info(
                        "Download progress: %d%% (%.2f / %.2f MiB).",
                        progress_percent,
                        downloaded_bytes / (1024 * 1024),
                        expected_size / (1024 * 1024),
                    )
                    next_progress_percent += 10

        actual_size = temporary.stat().st_size

        if actual_size != expected_size:
            raise RuntimeError(
                f"Downloaded size {actual_size} does not match "
                f"Figshare size {expected_size}."
            )

        temporary.replace(output)

    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download pinned LIBE v2 from Figshare."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for raw libe.json.",
    )
    parser.add_argument(
        "--expected-sha256",
        help="Optional previously recorded SHA256 to enforce.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing file when no checksum is supplied.",
    )
    return parser.parse_args()


def main() -> int:
    """Download and verify the pinned upstream file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = parse_args()

    LOGGER.info(
        "Resolving LIBE Figshare article %d version %d.",
        ARTICLE_ID,
        VERSION,
    )
    metadata, file_record, license_name = get_libe_file_record()

    LOGGER.info(
        "Resolved pinned source: %s, license=%s.",
        EXPECTED_FILE_NAME,
        license_name,
    )
    download_url = file_record.get("download_url")
    file_size = file_record.get("size")

    if not isinstance(download_url, str) or not download_url:
        raise RuntimeError("Figshare file record has no download URL.")

    if not isinstance(file_size, int) or file_size <= 0:
        raise RuntimeError("Figshare file record has no valid byte size.")

    expected_sha256 = (
        args.expected_sha256.lower() if args.expected_sha256 is not None else None
    )

    if expected_sha256 is not None:
        if len(expected_sha256) != 64:
            raise ValueError("Expected SHA256 must contain 64 hexadecimal characters.")
        int(expected_sha256, 16)

    if args.output.exists():
        existing_sha256 = sha256_file(args.output)

        if expected_sha256 is not None:
            if existing_sha256 != expected_sha256:
                raise RuntimeError(
                    "Existing LIBE file does not match the expected SHA256."
                )

            LOGGER.info("Existing LIBE file already matches the pinned SHA256.")
        elif not args.force:
            raise RuntimeError(
                "Output already exists but no expected SHA256 was supplied. "
                "Use --force only if intentional."
            )
        else:
            download_file(download_url, args.output, file_size)
    else:
        download_file(download_url, args.output, file_size)

    LOGGER.info("Calculating SHA256 for downloaded LIBE file.")
    actual_sha256 = sha256_file(args.output)
    LOGGER.info("SHA256 calculation complete.")

    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError("Downloaded LIBE file failed SHA256 verification.")

    receipt = {
        "source": "Figshare",
        "article_id": ARTICLE_ID,
        "version": VERSION,
        "doi": metadata.get("doi"),
        "license": license_name,
        "file_id": file_record.get("id"),
        "file_name": EXPECTED_FILE_NAME,
        "size_bytes": file_size,
        "upstream_md5": file_record.get("computed_md5") or file_record.get("md5"),
        "sha256": actual_sha256,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "metadata_url": VERSION_URL,
        "download_url": download_url,
    }

    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
