from pathlib import Path


def test_windows_real_backend_smoke_script_exists():
    """PowerShell real-backend VOC smoke script should be present."""

    script = Path("scripts/run_voc_real_smoke.ps1")

    assert script.exists()


def test_windows_real_backend_smoke_script_mentions_required_environment_variables():
    """Smoke script should document and use the required environment variables."""

    script_text = Path("scripts/run_voc_real_smoke.ps1").read_text(encoding="utf-8")

    assert "VOC_ROOT" in script_text
    assert "SAM_CHECKPOINT" in script_text
    assert "OUTPUT_DIR" in script_text
    assert "LIMIT" in script_text
    assert "PYTHON" in script_text


def test_readme_documents_real_backend_voc_smoke_test():
    """README should include real-backend VOC smoke-test instructions."""

    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Real Backend VOC Smoke Test" in readme
    assert "VOC_ROOT" in readme
    assert "SAM_CHECKPOINT" in readme
