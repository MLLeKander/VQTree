
from distutils.core import setup, Extension
import numpy as np
from distutils.sysconfig import get_python_inc
CPP_ARGS=['-O3', '-std=c++11', '-Wuninitialized', '-Weffc++']
ktreeExtension = Extension('_ktree',
                    include_dirs = [get_python_inc(), np.get_include()],
                    extra_compile_args = ['-O3', '-std=c++11'],
                    sources = ['ktreemodule.cpp'])
meantreeExtension = Extension('_meantree',
                    include_dirs = [get_python_inc(), np.get_include()],
                    extra_compile_args = ['-O3', '-std=c++11'],
                    sources = ['meantreemodule.cpp'])

#remove annoying strict-prototypes warning...
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "").replace("-DNDEBUG","")

setup (name = 'vqtree',
       version = '1.0',
       description = 'Data structure for vector quantization tree',
       author = 'Michael LeKander',
       author_email = 'm.l.lekander@gmail.com',
       #install_requires=["numpy"],
       ext_modules = [ktreeExtension],#, meantreeExtension],
       py_modules = ['vqtree'],
)
