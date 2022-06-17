#!/bin/bash

git clone https://github.com/kymatio/kymatio.git
cd kymatio
pip install -r requirements.txt
pip install -r requirements_optional.txt
python setup.py install

