from pathlib import Path


def test_resource_download_scripts_exist():
    """Resource preparation scripts should be available for Windows users."""

    assert Path("scripts/download_voc2012.ps1").exists()
    assert Path("scripts/download_sam_checkpoint.ps1").exists()


def test_readme_documents_resource_environment_variables():
    """README should show the environment variables used by real backends."""

    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Preparing Real Backend Resources" in readme
    assert "VOC_ROOT" in readme
    assert "SAM_CHECKPOINT" in readme


def test_gitignore_excludes_large_resource_artifacts():
    """Large local datasets, checkpoints, and outputs must stay out of Git."""

    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "outputs/" in gitignore
    assert "VOCdevkit/" in gitignore
    assert "*.pth" in gitignore
    assert "*.pt" in gitignore
    assert "*.ckpt" in gitignore
    assert "*.safetensors" in gitignore
