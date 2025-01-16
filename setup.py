import os
import pathlib
from sys import version_info, exit
from setuptools import setup, find_packages
from Cython.Build import cythonize
from pkg_resources import parse_requirements

def write_version_py():
    version_txt_path = os.path.join("IndicTransToolkit", "version.txt")
    with open(version_txt_path, "r", encoding="utf-8") as f:
        version = f.read().strip()

    version_py_path = os.path.join("IndicTransToolkit", "version.py")
    with open(version_py_path, "w", encoding="utf-8") as f:
        f.write(f'__version__ = "{version}"\n')
    return version

# Enforce Python >= 3.8
if version_info < (3, 8):
    exit("Sorry, Python >= 3.8 is required for IndicTransToolkit.")

# Read long description from README
with open("README.md", "r", errors="ignore", encoding="utf-8") as fh:
    long_description = fh.read().strip()

# Write version.py from version.txt
version = write_version_py()

# Parse requirements.txt
req_file = pathlib.Path("requirements.txt")
requirements = [str(req) for req in parse_requirements(req_file.open())]

# Cython files to compile (adjust if your .pyx name differs)
cython_extensions = cythonize(
    [
        "IndicTransToolkit/processor.pyx",
    ],
    compiler_directives={"language_level": "3", "boundscheck": False},
)

setup(
    name="IndicTransToolkit",
    version=version,
    author="Varun Gumma",
    author_email="varun230999@gmail.com",
    description="A simple, consistent, and extendable module for IndicTrans2 tokenizer compatible with HuggingFace models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VarunGumma/IndicTransToolkit",
    packages=find_packages(),  # Auto-detect packages
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    ext_modules=cython_extensions,
    zip_safe=False,
)
