import pathlib

from training import train


def test_train(data_dirs, tmpdir):
    log_dir = tmpdir / "log_dir"
    log_dir.mkdir()
    weights, trace_file = train(data_dirs, max_epochs=10, batch_size=5, log_dir=log_dir)

    assert pathlib.Path(weights).exists()
    assert pathlib.Path(trace_file).exists()
