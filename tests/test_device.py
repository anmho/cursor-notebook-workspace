from ml_lab.device import device_report, run_smoke_test, select_device


def test_cpu_selection_is_explicit() -> None:
    assert str(select_device("cpu")) == "cpu"


def test_device_report_has_actual_capabilities() -> None:
    report = device_report("cpu")
    assert report.selected == "cpu"
    assert isinstance(report.cuda_available, bool)
    assert isinstance(report.mps_available, bool)


def test_smoke_test_cpu() -> None:
    result = run_smoke_test("cpu", steps=2)
    assert result["device"] == "cpu"
    assert result["matmul_shape"] == [16, 4]
    assert len(result["losses"]) == 2
