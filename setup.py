import setuptools

setuptools.setup(
    name="boardAttitude",
    version="1.0.1",
    author="Zhang Bei",
    author_email="rular099@gmail.com",
    description="calculate attitude using acc/gyro/magnet sensor data",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)
