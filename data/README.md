# Data directory

This directory separates downloaded source data, intermediate transformations,
and compact analysis-ready datasets. Whether a file enters Git depends on its
size, provenance, reproducibility, and license—not on available local storage.

## Directory layout

### `external/`

Untouched third-party inputs, including downloaded public datasets.

These files remain outside Git. Future download scripts must record the source,
retrieval date, version, license, and checksum.

### `interim/`

Temporary or partially transformed data produced between raw inputs and final
analysis-ready datasets.

These files remain outside Git and must be reproducible from documented source
data and scripts.

### `processed/`

Compact, analysis-ready datasets produced by version-controlled workflows.

A processed file may enter Git only when it is reasonably small, necessary for
reproduction or review, documented, and permitted by its license. Its provenance
and generating configuration must also be recorded.

## Files that may enter Git

- Data documentation and schemas
- Checksums and provenance manifests
- Small, curated metadata tables
- Compact processed datasets needed to reproduce figures or tests
- Small representative structures
- Original CEI-Scout data and figures released under CC BY 4.0

## Files that must not enter Git

- The raw LIBE archive or other large public-data downloads
- Large intermediate datasets
- Full molecular-dynamics trajectories
- Restart files and large simulation outputs
- Model checkpoints
- Scratch files, caches, environments, or secrets

Third-party data retains its original license and attribution requirements.
CEI-Scout does not relicense third-party materials.
