from setuptools import setup, find_packages

setup(
    name='yukarin',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/Hiroshiba/yukarin',
    author='Kazuyuki Hiroshiba',
    author_email='hihokaruta@gmail.com',
    description='Everyone become Yuzuki Yukari with DeepLearning power.',
    license='MIT License',
    install_requires=[
        'numpy',
        'chainer<6.0.0',
        'librosa<0.7.0',
        'pysptk',
        'pyworld',
        'fastdtw',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
    ]
)
