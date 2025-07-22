from setuptools import setup
from Cython.Build import cythonize

setup(
    name="IndicTransToolkit",
    ext_modules=cythonize(
        ["IndicTransToolkit/processor.pyx"],
        compiler_directives={"language_level": "3"}
    ),
    packages=["IndicTransToolkit"],
)
