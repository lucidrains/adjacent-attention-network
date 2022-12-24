from setuptools import setup, find_packages

setup(
  name = 'adjacent-attention-pytorch',
  packages = find_packages(),
  version = '0.0.12',
  license='MIT',
  description = 'Adjacent Attention Network - Pytorch',
  long_description_content_type = 'text/markdown',
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
    'isab-pytorch<0.2'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
