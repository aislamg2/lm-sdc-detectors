from pathlib import Path

import pytest

from sdc.config import REPO_ROOT, load_config


def write_overlay(tmp_path, text: str) -> Path:
    path = tmp_path / "overlay.yaml"
    path.write_text(text)
    return path


def test_default_config_has_expected_values():
    config = load_config()
    assert config["seed"] == 42
    assert config["device"] == "cpu"
    assert config["data"]["split_fractions"] == [0.8, 0.1, 0.1]
    assert config["model"]["monitored_points"] == [
        "stem", "layer1", "layer2", "layer3", "layer4", "fc",
    ]
    assert config["injection"]["sites"] == ["weight", "activation"]
    assert config["eval"]["target_fpr"] == 0.05


def test_config_split_matches_the_notebook_defaults():
    from sdc.data.eurosat import load_test_split
    import inspect

    defaults = inspect.signature(load_test_split).parameters
    config = load_config()["data"]
    assert tuple(config["split_fractions"]) == defaults["fractions"].default
    assert config["split_seed"] == defaults["seed"].default


def test_monitored_points_match_the_monitor():
    from sdc.model.monitor import MONITORED_POINTS

    assert load_config()["model"]["monitored_points"] == MONITORED_POINTS


def test_every_path_is_absolute_under_the_repo():
    for name, value in load_config()["paths"].items():
        assert Path(value).is_absolute(), name
        assert Path(value).is_relative_to(REPO_ROOT), name


def test_paths_do_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    expected = load_config()["paths"]
    monkeypatch.chdir(tmp_path)
    assert load_config()["paths"] == expected


def test_gguf_path_moved_to_paths():
    config = load_config()
    assert config["paths"]["qwen_gguf"].endswith("models/qwen3-0.6b-q4.gguf")
    assert "gguf_path" not in config["lm"]


def test_overlay_changes_only_what_it_names(tmp_path):
    overlay = write_overlay(tmp_path, "injection:\n  injections_per_image: 3\n")
    config, default = load_config(overlay), load_config()
    assert config["injection"]["injections_per_image"] == 3
    assert config["injection"]["shard_size"] == default["injection"]["shard_size"]
    assert config["data"] == default["data"]


def test_overlay_accepts_a_string_path(tmp_path):
    overlay = write_overlay(tmp_path, "seed: 7\n")
    assert load_config(str(overlay))["seed"] == 7


def test_overlay_replaces_lists_whole(tmp_path):
    overlay = write_overlay(tmp_path, "injection:\n  sites: [weight]\n")
    assert load_config(overlay)["injection"]["sites"] == ["weight"]


def test_overlay_can_set_a_null_default(tmp_path):
    overlay = write_overlay(tmp_path, "data:\n  images_per_class: 20\n")
    assert load_config(overlay)["data"]["images_per_class"] == 20


def test_overlay_paths_are_resolved_too(tmp_path):
    overlay = write_overlay(tmp_path, "paths:\n  checkpoint: elsewhere/model.pt\n")
    assert load_config(overlay)["paths"]["checkpoint"] == str(REPO_ROOT / "elsewhere/model.pt")


def test_absolute_paths_are_left_alone(tmp_path):
    overlay = write_overlay(tmp_path, "paths:\n  eurosat_dir: /mnt/ssd/EuroSAT_RGB\n")
    assert load_config(overlay)["paths"]["eurosat_dir"] == "/mnt/ssd/EuroSAT_RGB"


@pytest.mark.parametrize("text, name", [
    ("sed: 7\n", "sed"),
    ("injection:\n  injection_per_image: 3\n", r"injection\.injection_per_image"),
])
def test_unknown_keys_are_rejected_with_their_name(tmp_path, text, name):
    with pytest.raises(ValueError, match=name):
        load_config(write_overlay(tmp_path, text))


@pytest.mark.parametrize("text", ["injection: 3\n", "seed:\n  value: 7\n"])
def test_changing_a_section_into_a_value_is_rejected(tmp_path, text):
    with pytest.raises(ValueError, match="shape"):
        load_config(write_overlay(tmp_path, text))


def test_empty_overlay_changes_nothing(tmp_path):
    assert load_config(write_overlay(tmp_path, "")) == load_config()


def test_non_mapping_overlay_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="mapping"):
        load_config(write_overlay(tmp_path, "- just\n- a list\n"))


def test_missing_overlay_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_yaml_cannot_run_code(tmp_path):
    import yaml

    overlay = write_overlay(tmp_path, "seed: !!python/object/apply:os.getcwd []\n")
    with pytest.raises(yaml.YAMLError):
        load_config(overlay)
