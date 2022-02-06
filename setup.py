from setuptools import setup, find_packages

setup(
  name='WYNAssociates',
  author='Yiqiao Yin',
  author_email="Yiqiao.Yin@wyn-associates.com",
  version='1.0.0',
  description="This package provides AI-driven solutions.",
  packages=find_packages()
  packages=find_packages(include=["AI-solutions", "YinCapital_forecast"]),
)
