from setuptools import find_packages, setup

setup(
    name="arnold",
    version="0.0.1",
    author="René Schubotz",
    author_email="r.schubotz@googlemail.com",
    description="tf.Keras implementations of Kolmogorov-Arnold Networks (KAN) using orthogonal polynomial bases",
    long_description="TBD",
    long_description_content_type="text/markdown",
    url="https://github.com/resc2801/arnold",
    project_urls={
        "Bug Tracker": "https://github.com/resc2801/arnold/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
#        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
        include=["arnold"]
    ),
)