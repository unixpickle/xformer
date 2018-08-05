from setuptools import setup

setup(
    name='xformer',
    version='1.0.1',
    description='A practical implementation of the Transformer neural network architecture.',
    url='https://github.com/unixpickle/xformer',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    license='BSD',
    packages=['xformer'],
    install_requires=[
        'numpy>=1.0.0,<2.0.0',
    ],
    extras_require={
        "tf": ["tensorflow>=1.0.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0.0"],
    }
)
