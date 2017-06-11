Electrical Circuit Battery Model
================================

Discrete python implementation of the *electrical circuit battery model* inspired by [Chen et al. 2006](http://ieeexplore.ieee.org/abstract/document/1634598/).

Requirements
------------
* Python 3
* NumPy
* SciPy
* Matplotlib

Download
--------
Clone repository
```
git clone https://github.com/christiansiegel/electrical-circuit-battery-model.git
cd electrical-circuit-battery-model
```

Unzip [measurement data](measurements) (optional)
```
cd measurements
unzip '*.zip'
```

Smooth [measurement data](measurements) (optional)
```
python3 filter.py
```

Files
-----

### Model Extraction
The model can be extracted from [measurement data](measurements) with [extract.py](extract.py). The resulting battery model class can be pasted in [batteries.py](batteries.py) (already done).

### Simulation
The discrete simulator is implemented in [simulator.py](simulator.py). The battery models are included from [batteries.py](batteries.py). The different discharge loads are found in [loads.py](loads.py).

For specific examples on how to use the simulator see [examples](examples).

### Measurement Comparison
The [measurement data](measurements) smoothed with [filter.py](measurements/filter.py) can be loaded using the CSV helper class in [csvhelper.py](csvhelper.py). Functions to calculate the used energy or calculate errors are found in [measurement.py](measurement.py).

For guidance see [5-measurement-compare.py](examples/5-measurement-compare.py) in particular.
