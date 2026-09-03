def test_coffeebean_environment():
    assert True


def test_config_loads():
    import yaml
    import pathlib

    cfg = yaml.safe_load(
        pathlib.Path("configs/config.yaml").read_text()
    )
    assert cfg["project"]["name"] == "COFFEEBEAN"