from setuptools import setup, find_packages

setup(
    name="wafer-defect-yolo",
    version="1.0.0",
    description="Semiconductor wafer defect detection with YOLO + ROCm optimization",
    author="penampungdipek",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "ultralytics>=8.2.0",
        "torch>=2.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.5.0",
    ],
    extras_require={
        "rocm": ["torch>=2.0.0+rocm6.2"],
        "cuda": ["torch>=2.0.0+cu121"],
        "migraphx": ["migraphx"],
        "dev": ["pytest>=8.0.0", "pytest-cov>=4.1.0"],
    },
)
