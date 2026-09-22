# LIBE v2 field map

CEI-Scout uses the original LIBE identifiers and scientific fields without
renaming or altering the underlying values during ingestion.

| CEI-Scout concept | LIBE v2 field | Treatment |
|---|---|---|
| Source record identifier | `molecule_id` | Preserved exactly as published |
| Molecular charge | `charge` | Preserved as integer charge |
| Spin state | `spin_multiplicity` | Preserved as published multiplicity |
| Element sequence | `species` | Preserved in atom order |
| Cartesian geometry | `xyz` | Preserved in the published atom order |
| Composition | `composition` | Preserved |
| Element set | `elements` | Preserved |
| Chemical system | `chemical_system` | Preserved |
| Molecular formula | `formula_alphabetical` | Preserved |
| Connectivity | `bonds` | Preserved; no rebonding during ingestion |
| Serialized molecular object | `molecule` | Retained as source metadata when needed |
| Serialized graph | `molecule_graph` | Retained as source metadata when needed |
| Electronic/thermochemical data | `thermo` | Preserved as the original nested LIBE object |
| Vibrational data | `vibration` | Preserved as the original nested LIBE object |

## Thermochemical convention

LIBE reports electronic energy and thermochemical quantities generated from
the published Q-Chem workflow. The `thermo` object is preserved unchanged
during ingestion.

No CEI-Scout retrospective target is defined at this stage. P06-11 will
select and document the exact energy convention used for retrospective
learning after neutral and charge-state matching is defined.

## Geometry convention

`species` and `xyz` are treated as an ordered pair: coordinate row `i`
corresponds to species `i`.

No geometry optimization, atom reordering, charge modification, rebonding,
or unit conversion occurs during raw LIBE ingestion.
