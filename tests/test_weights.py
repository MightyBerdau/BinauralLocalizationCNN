import numpy as np
import pytest

from blcnn.weights import WeightsError, find_networks, load_config_array, select_networks


def _make_network(weights_dir, name, steps=(100000,)):
    folder = weights_dir / name
    folder.mkdir(parents=True)
    np.save(str(folder / "config_array.npy"),
            np.array([[["/gpu:0"], ["conv", [1, 8, 32], [1, 1]]]], dtype=object))
    for step in steps:
        (folder / "model.ckpt-{}.index".format(step)).write_bytes(b"")
    return folder


def test_finds_download_and_original_folder_names(tmp_path):
    _make_network(tmp_path, "net1")
    _make_network(tmp_path, "net10")
    _make_network(tmp_path, "arch_number_3_init_0")
    (tmp_path / "unrelated").mkdir()
    networks = find_networks(str(tmp_path))
    assert sorted(networks) == [1, 3, 10]
    assert networks[1].checkpoint.endswith("model.ckpt-100000")


def test_latest_checkpoint_is_used(tmp_path):
    _make_network(tmp_path, "net2", steps=(50000, 100000))
    assert find_networks(str(tmp_path))[2].checkpoint.endswith("model.ckpt-100000")


def test_duplicate_network_number(tmp_path):
    _make_network(tmp_path, "net1")
    _make_network(tmp_path, "arch_number_1_init_0")
    with pytest.raises(WeightsError, match="found twice"):
        find_networks(str(tmp_path))


def test_missing_files(tmp_path):
    (tmp_path / "net1").mkdir()
    with pytest.raises(WeightsError, match="config_array"):
        find_networks(str(tmp_path))
    with pytest.raises(WeightsError, match="not found"):
        find_networks(str(tmp_path / "nope"))


def test_select_networks(tmp_path):
    for n in (1, 2, 3):
        _make_network(tmp_path, "net{}".format(n))
    assert [n.number for n in select_networks(str(tmp_path), "all")] == [1, 2, 3]
    assert [n.number for n in select_networks(str(tmp_path), "3, 1")] == [1, 3]
    with pytest.raises(WeightsError, match="not found"):
        select_networks(str(tmp_path), "1,7")
    with pytest.raises(WeightsError, match="invalid"):
        select_networks(str(tmp_path), "one")


def test_load_config_array(tmp_path):
    folder = _make_network(tmp_path, "net1")
    config = load_config_array(find_networks(str(tmp_path))[1])
    assert config[0][0] == ["/gpu:0"]
    assert folder.exists()
