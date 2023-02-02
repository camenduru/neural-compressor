Modular Design of Neural Coder Launcher
===========================

To enable modular design of Neural Coder launcher in your product, please simply follow the below steps:

### Create a Launcher File

Create a launcher file with file name such as ```launcher.py```, then put the following code lines into this launcher file:

```
from neural_coder.launcher import Launcher

STATIC_QUANT_API_DESIGN = XXX # a multi-line string, representing your API design for static quantization
DYNAMIC_QUANT_API_DESIGN = XXX # a multi-line string, representing your API design for dynamic quantization

args = Launcher.parse_args()

modular_pattern = {}

modular_pattern["pytorch_inc_static_quant_fx"] = STATIC_QUANT_API_DESIGN
modular_pattern["pytorch_inc_dynamic_quant"] = DYNAMIC_QUANT_API_DESIGN
modular_pattern["inc_auto"] = DYNAMIC_QUANT_API_DESIGN # if your default quantization approach is dynamic

Launcher.execute(args, use_modular=True, modular_pattern=modular_pattern)
```

### String for API Design

The multi-line string for API design is easy to construct. For example, if your API adopts the following pattern:

```
my_model = ...
...
from OptimizeAPI import Optimize
my_model = Optimize(my_model)
...
```

where there are two added code lines.

Then your multi-line string looks like

```
s = f"""\
from OptimizeAPI import Optimize
MODEL_NAME = Optimize(MODEL_NAME)
"""
```

### Use Launcher

Then, simply use the launcher file to execute Neural Coder. For example, if the launcher file ```launcher.py``` is positioned under your root path of the Python package `my_package`, then your Neural Coder launcher header is ```python -m my_package.launcher```. If your Deep Learning code is `main.py`, then the final usage will be ```python -m my_package.launcher main.py xxx (arguments)```. For detailed launcher usage, please refer to this [doc](./PythonLauncher.MD).
