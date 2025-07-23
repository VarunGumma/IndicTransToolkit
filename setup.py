from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

# define extension like this since we have an __init__.py file in the package
# and we want to compile the processor.pyx file as a module
extensions = [
    Extension(
        "IndicTransToolkit.processor",
        sources=["IndicTransToolkit/processor.pyx"],
    )
]

# Cython extensions
cython_extensions = cythonize(extensions, compiler_directives={"language_level": "3"})

setup(
    name="indictranstoolkit",
    ext_modules=cython_extensions,
    include_package_data=True,
    packages=find_packages(),
)
