from setuptools import setup
from Cython.Build import cythonize

# Read long description from README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read().strip()

# version
version = "1.1.0"

# Fixed core dependencies
install_requires = [
    "cython",
    "sacremoses",
    "transformers",
    "sacrebleu",
    "indic-nlp-library-itt",
]

# Cython extensions
cython_extensions = cythonize(
    ["IndicTransToolkit/processor.pyx"],
    compiler_directives={"language_level": "3", "boundscheck": False},
)

setup(
    name="IndicTransToolkit",
    author="Varun Gumma",
    author_email="varun230999@gmail.com",
    description="A simple, consistent, and extendable module for IndicTrans2 compatible with HF models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VarunGumma/IndicTransToolkit",
    packages=["IndicTransToolkit"],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    ext_modules=cython_extensions,
    version=version,
    zip_safe=False,
    include_package_data=True,
    package_data={"IndicTransToolkit": ["processor.c"]},
)