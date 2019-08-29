# LEEM Image Simulation with Fourier Optics

This is a simulation program written in python that help user to simulate sample viewed in Low Energy Electron Microscopy (LEEM) using Fourier Optics (FO). The method and mathematical derivation are documented at [here](https://www.sciencedirect.com/science/article/abs/pii/S0304399118304418).


# Package Requirements
- Python 3.6+
- joblib 0.13.2+
- numba 0.41.0+
- scipy 1.3.1+
- numpy 1.16.4+
- pytz 2019.2+

# Simulation Setup
Users can match the simulation with their LEEM setting by tuning constants and sample in the program.

## 2D FO Simulation Setup
In 2D FO Simulation, user can setup the constants in `FO2Dconstants.py`.  
Details of constants are listed [here](https://github.com/klwlau/LEEM-Fourier-Optics/blob/master/docs/FO2Dconstants.md).

To setup the sample in the simulation user can specify in `FO2Dsample.py`.  
Details of sample setup are listed [here](https://github.com/klwlau/LEEM-Fourier-Optics/blob/master/docs/FO2Dsample.md).

## Program Execution

After the config, user can run the software in terminal or by other editor:

```
python run2DFOSimulation.py
```


## Simulation Output
Two `.npy` files are saved after the simulation. 
- `simObject_TimeStamp.npy` for simulated sample.
- `result_UserNote_TimeStamp.npy` for simulated result.

# Terms of use

