from pathlib import Path

import yaml


def test_sugarcane_agml_config_matches_all_eleven_source_classes():
    config_path = Path(__file__).parents[1] / "configs" / "sugarcane_karnataka.yaml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    expected = [
        "Banded Chlorosis",
        "Brown Spot",
        "BrownRust",
        "Dried",
        "Grassy shoot",
        "Healthy",
        "Pokkah Boeng",
        "Sett Rot",
        "Smut",
        "Viral Disease",
        "Yellow Leaf",
    ]
    assert config["taxonomy"]["classes"] == expected
    assert len(set(config["taxonomy"]["classes"])) == 11
    assert config["taxonomy"]["source"] == "Project-AgML"
    assert config["taxonomy"]["label_status"] == "unverified"


def test_sugarcane_gpu_training_defaults_are_conservative():
    config_path = Path(__file__).parents[1] / "configs" / "sugarcane_karnataka.yaml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    assert config["model"]["name"] == "efficientnet_b0"
    assert config["model"]["input_size"] == 224
    assert config["training"]["batch_size"] == 8
    assert config["training"]["num_workers"] == 0
