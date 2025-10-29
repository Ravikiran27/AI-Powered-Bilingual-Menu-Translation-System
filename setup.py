from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bilingual-menu-translation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered bilingual menu translation system for English to Hindi and Kannada",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/bilingual-menu-translation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "menu-translate=app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
