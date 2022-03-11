from pathlib import Path
from pkg_resources import parse_requirements
from setuptools import find_packages, setup

def get_install_requires():
    with Path('requirements.txt').open() as requirements_txt:
        return [str(r) for r in parse_requirements(requirements_txt)]

setup(
    name="cognitive_subtypes",
    version="0.0.1",
    author="Joshua Unrau",
    author_email="joshua@joshuaunrau.com",
    description="Cognitive Subtypes and Associations with Brain Structure in the UK Biobank: A Machine Learning Approach",
    url="https://github.com/joshunrau/CognitiveSubtypes",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires="==3.9.*",
    install_requires=get_install_requires(),
    entry_points = {
        'console_scripts': [
            'autossh=autossh.main:main'
        ]
    },
    package_data={"data" : ["variables/*.json", "variables/coding/*.json"]},
    include_package_data=True,
)