import os
import sys

from setuptools import find_packages, setup

if __name__ == '__main__':

    if sys.version_info < (3, 7):
        raise ValueError(
            'Unsupported Python version %d.%d.%d found. Auto-tabular requires Python '
            '3.7 or higher.' % (sys.version_info.major, sys.version_info.minor,
                                sys.version_info.micro))

    HERE = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(HERE, 'requirements.txt')) as fp:
        install_reqs = [
            r.rstrip() for r in fp.readlines()
            if not r.startswith('#') and not r.startswith('git+')
        ]

    extras_reqs = {
        'test': [
            'pytest>=4.6',
            'mypy',
            'pytest-xdist',
            'pytest-timeout',
            'flaky',
            'openml',
            'pre-commit',
            'pytest-cov',
        ],
        'examples': [
            'matplotlib',
            'jupyter',
            'notebook',
            'seaborn',
        ],
        'docs':
        ['sphinx', 'sphinx-gallery', 'sphinx_bootstrap_theme', 'numpydoc'],
    }

    with open('autotabular/__version__.py') as fh:
        version = fh.readlines()[-1].split()[-1].strip("\"'")

    with open('README.md', encoding='utf-8') as fh:
        long_description = fh.read()

    setup(
        name='autotabular',
        author='Jianzh Nie',
        author_email='jianzhnie@gmail.com',
        description='Automatic machine learning for tabular data.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        version=version,
        packages=find_packages(
            exclude=['test', 'scripts', 'examples', 'docs']),
        extras_require=extras_reqs,
        install_requires=install_reqs,
        include_package_data=True,
        license='Apache License',
        platforms=['Linux'],
        classifiers=[
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Information Technology',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Information Analysis',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        keywords=[
            'automated machine learning', 'automl', 'machine learning',
            'data science', 'data mining', 'autotabular'
        ],
        python_requires='>=3.7',
        url='https://github.com/jianzhnie/AutoTabular',
    )
