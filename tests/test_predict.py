import pytest

from blcnn.predict import decode_class, errors_path, find_audio_files


@pytest.mark.parametrize("index, expected", [
    (0, (0, 0)),
    (1, (5, 0)),
    (71, (355, 0)),
    (72, (0, 10)),
    (18, (90, 0)),
    (503, (355, 60)),
])
def test_decode_class(index, expected):
    assert decode_class(index) == expected


def test_decode_matches_original_label_construction():
    # original code: class = 36 * elev_label + azim_label, labels in 5-degree units
    for azim in range(0, 360, 5):
        for elev in range(0, 70, 10):
            index = 36 * (elev // 5) + azim // 5
            assert decode_class(index) == (azim, elev)


def test_find_audio_files_recursive_and_sorted(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ["b.wav", "a.WAV", "sub/c.wav", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    assert find_audio_files(str(tmp_path)) == ["a.WAV", "b.wav", "sub/c.wav"]


def test_find_audio_files_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_audio_files(str(tmp_path / "missing"))


def test_errors_path():
    assert errors_path("out/results.csv") == "out/results_errors.csv"
    assert errors_path("results") == "results_errors.csv"
