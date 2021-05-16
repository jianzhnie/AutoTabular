from pathlib import Path

from setuptools import setup, find_packages


if __name__ == '__main__':

    with open(Path(__file__).parent / 'README.md', encoding='utf-8') as f:
        long_description = f.read()

    setup(
        name="AutoTabulartorchfm",
        version="0.1.0",
        description="PyTorch implementation of autogluon for tabulardata",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jianzhnie/AutoTabular",
        author="jianzhnie",
        author_email="jianzhnie@gmail.com",
        packages=find_packages(exclude=["examples", "docs"]),
    )
