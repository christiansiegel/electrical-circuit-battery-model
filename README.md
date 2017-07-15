Electrical Circuit Battery Model
================================

Discrete python implementation of the *electrical circuit battery model* inspired by [Chen et al. 2006](http://ieeexplore.ieee.org/abstract/document/1634598/).

Requirements
------------
* Python
* NumPy¹
* SciPy¹
* Matplotlib¹²

¹ For model extraction
² For plotting example results

The simulator is written in plain Python and can easily be used with [PyPy](https://pypy.org/) for faster execution.

Download
--------

### Option 1: Simulator Only

Clone repository
```
git clone https://github.com/christiansiegel/electrical-circuit-battery-model.git
cd electrical-circuit-battery-model
```

### Option 2: Simulator + Measurements

Clone repository
```
git clone --recursive https://github.com/christiansiegel/electrical-circuit-battery-model.git
cd electrical-circuit-battery-model
```

Unzip [measurement data](https://github.com/christiansiegel/battery-discharge-measurements/)
```
cd measurements
unzip '*.zip'
```

Smooth [measurement data](https://github.com/christiansiegel/battery-discharge-measurements/)
```
python3 filter.py
```

Files
-----

### Model Extraction
The model can be extracted from [measurement data](https://github.com/christiansiegel/battery-discharge-measurements/) with [extract.py](extract.py). The resulting battery model class can be pasted in [batteries.py](batteries.py) (already done).

### Simulation
The discrete simulator is implemented in [simulator.py](simulator.py). The battery models are included from [batteries.py](batteries.py). The different discharge loads are found in [loads.py](loads.py).

For specific examples on how to use the simulator see [examples](examples).

You can also use the simulator in a [Jupyter](http://jupyter.org/) notebook. For an example notebook see [jupyter-examples.ipynb](examples/jupyter-examples.ipynb).

### Measurement Comparison
The [measurement data](https://github.com/christiansiegel/battery-discharge-measurements/) smoothed with [filter.py](https://github.com/christiansiegel/battery-discharge-measurements/blob/master/filter.py) can be loaded using the CSV helper class in [csvhelper.py](csvhelper.py). Functions to calculate the used energy or calculate errors are found in [measurement.py](measurement.py).

For guidance see [5-measurement-compare.py](examples/5-measurement-compare.py) in particular.
