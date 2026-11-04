from dataclasses import fields

from sdc.records.schema import (
    SCHEMA_VERSION,
    BitGroup,
    CrashReason,
    FaultSite,
    Label,
    LayerStats,
    RunRecord,
)

FROZEN_FIELDS = {
    1: [
        "run_id", "plan_idx", "seed", "split", "image_id", "image_path", "true_class",
        "fault_site", "layer", "element", "bit_position", "bit_group",
        "original_value", "flipped_value",
        "clean_pred", "faulty_pred", "label", "crash_reason", "faulty_logits",
        "top1_prob", "entropy",
        "checkpoint_sha256", "git_commit", "device", "timestamp", "inference_time_s",
        "layer_stats", "model_format", "schema_version",
    ],
}


def make_record(**overrides) -> RunRecord:
    values = dict(
        run_id="r0", plan_idx=0, seed=42, split="fit", image_id=7,
        image_path="Forest/Forest_1.jpg", true_class=1,
        fault_site=FaultSite.WEIGHT, layer="layer3.1.conv2", element=[0, 1, 2, 3],
        bit_position=30, bit_group=BitGroup.EXPONENT,
        original_value=0.5, flipped_value=1.7e38,
        clean_pred=1, faulty_pred=4, label=Label.SDC, crash_reason=None,
        faulty_logits=[0.0] * 10, top1_prob=0.9, entropy=0.3,
        checkpoint_sha256="abc", git_commit="def", device="cpu",
        timestamp="2026-11-03T00:00:00+00:00", inference_time_s=0.2,
    )
    return RunRecord(**{**values, **overrides})


def test_changing_fields_requires_a_version_bump():
    # If this fails: bump SCHEMA_VERSION, add a new entry above, and tell the team.
    assert [f.name for f in fields(RunRecord)] == FROZEN_FIELDS[SCHEMA_VERSION]


def test_layer_stats_fields_match_the_proposal():
    assert [f.name for f in fields(LayerStats)] == ["min", "max", "mean", "std", "nonfinite"]


def test_record_gets_current_version_and_default_format():
    record = make_record()
    assert record.schema_version == SCHEMA_VERSION
    assert record.model_format == "pytorch_fp32"
    assert record.layer_stats == {}


def test_crashed_run_can_leave_outcome_fields_empty():
    record = make_record(
        faulty_pred=None, label=Label.CRASH, crash_reason=CrashReason.TIMEOUT,
        faulty_logits=None, top1_prob=None, entropy=None, inference_time_s=None,
    )
    assert record.label == Label.CRASH
    assert record.crash_reason == CrashReason.TIMEOUT


def test_enum_values_are_stable_strings():
    assert {e.value for e in Label} == {"masked", "sdc", "crash"}
    assert {e.value for e in CrashReason} == {"exception", "timeout", "nonfinite"}
    assert {e.value for e in FaultSite} == {"weight", "activation"}
    assert {e.value for e in BitGroup} == {"sign", "exponent", "mantissa"}
