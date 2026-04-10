"""Build configuration for Cython extensions."""

from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize

    extensions = cythonize(
        [
            Extension(
                "impute_knn_tn._correlation",
                ["src/impute_knn_tn/_correlation.pyx"],
                include_dirs=[np.get_include()],
            ),
        ],
        compiler_directives={"language_level": "3"},
    )
except ImportError:
    # Cython not available — skip extension build
    extensions = []

setup(ext_modules=extensions)
