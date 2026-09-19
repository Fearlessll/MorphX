"""Install the MorphX model and analysis code."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"
DEPENDENCIES = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="morphx",
    version="0.1.0",
    description="MorphX survival models and analysis for HCC and LUAD",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["configs*", "data_preprocess*", "prognosis*"]),
    package_data={"configs": ["metadata/*.cvs"]},
    install_requires=DEPENDENCIES,
    python_requires=">=3.10",
)
