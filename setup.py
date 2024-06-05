from setuptools import find_packages, setup

readme = open('README.md').read()

setup(
    name='dinoexplorer',
    packages=find_packages(),
    version='0.2.0',
    description='DINO Explorer: A tool to explore DINO embeddings.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Nityanand Mathur',
    license='MIT',
    install_requires=['tqdm', 'pillow', 'rich', 'fiftyone', 'numpy', 'transformers', 'umap-learn'],
    entry_points={
        'console_scripts': [
            'diex=diex.diex:main'
        ]
    }
)
