from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .constants import CLASS_NAMES, FRAME_COUNT
from .dataset import SequenceRecord


@dataclass(frozen=True)
class DatasetValidation:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    sequence_count: int
    labeled_count: int
    localized_count: int
    class_counts: dict[str, int]
    scenario_counts: dict[str, int]
    strata_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "sequence_count": self.sequence_count,
            "labeled_count": self.labeled_count,
            "localized_count": self.localized_count,
            "class_counts": self.class_counts,
            "scenario_counts": self.scenario_counts,
            "strata_counts": self.strata_counts,
        }


def validate_dataset(records: list[SequenceRecord], strict_proposal: bool = True) -> DatasetValidation:
    errors: list[str] = []
    warnings: list[str] = []
    class_counter: Counter[str] = Counter()
    scenario_counter: Counter[str] = Counter()
    strata_counter: Counter[str] = Counter()
    labeled = localized = 0
    for record in records:
        if record.total_frame_count != FRAME_COUNT:
            errors.append(f"{record.path}: expected exactly 40 frames, found {record.total_frame_count}")
        if record.annotation is None:
            errors.append(f"{record.path}: missing valid annotation")
            continue
        labeled += 1
        class_name = CLASS_NAMES[record.annotation.class_id]
        class_counter[class_name] += 1
        scenario_counter[record.scenario] += 1
        strata_counter[f"{class_name}/{record.scenario}"] += 1
        if record.annotation.bbox_xywh is None:
            errors.append(f"{record.path}: annotation has no median-frame bounding box")
        else:
            localized += 1
        if record.scenario == "unknown":
            warnings.append(f"{record.path}: could not infer d1/d2/d3/d4 scenario from path")

    if strict_proposal:
        if len(records) != 72:
            errors.append(f"proposal requires 72 sequences; found {len(records)}")
        for class_name in CLASS_NAMES.values():
            if class_counter[class_name] != 12:
                errors.append(f"{class_name}: expected 12 sequences, found {class_counter[class_name]}")
            for scenario in ("d1", "d2", "d3", "d4"):
                key = f"{class_name}/{scenario}"
                if strata_counter[key] != 3:
                    errors.append(f"{key}: expected 3 sequences, found {strata_counter[key]}")
    return DatasetValidation(
        not errors, tuple(errors), tuple(warnings), len(records), labeled, localized,
        dict(sorted(class_counter.items())), dict(sorted(scenario_counter.items())),
        dict(sorted(strata_counter.items())),
    )

