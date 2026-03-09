from driftdart.utils.config import load_config


def test_load_config():
    cfg = load_config('configs/cicids2017.yaml')
    assert 'training' in cfg
