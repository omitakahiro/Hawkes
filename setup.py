import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hawkes",
    version="1.0.0",
    author="Takahiro Omi",
    author_email="takahiro.omi.em@gmail.com",
    description="a python package for simulation and inference of Hawkes processeses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/omitakahiro/Hawkes",
    project_urls={
        "Bug Tracker": "https://github.com/omitakahiro/Hawkes/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    packages=['Hawkes', 'Hawkes.tools'],
    install_requires=['numpy', 'scipy', 'cython', 'matplotlib'],
    python_requires=">=3.6",
)
