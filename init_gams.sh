bash
conda activate diplomka_v2
set PYTHONPATH="C:\GAMS\40\apifiles\Python\gams;C:\GAMS\40\apifiles\Python\api_310"
set SETUPTOOLS_USE_DISTUTILS="stdlib"
cd "C:\GAMS\40\apifiles\Python\api_310"
python setup.py install
$SHELL