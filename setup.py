from setuptools import setup, find_packages

setup(
  name = 'adjacent-attention-pytorch',
  packages = find_packages(),
  version = '0.0.5',
  license='MIT',
  description = 'Adjacent Attention Network - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/adjacent-attention-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'graph neural network',
    'transformers'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
