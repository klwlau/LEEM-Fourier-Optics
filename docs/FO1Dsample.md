# Setup the object function in 1D FO Simulation

To setup the object function, user can config a numpy array (np.array) in `create_object_fucntion()` in `runFO1DSimulation.py`. 
The object function is create in `main()`.

## Create object function
To preview the object function user can run `FO2Dsample.py` by itself, a plot of `simulatedObject` is generated accordingly.

For example we generate a periodic step object with step height 1:

```python 

def create_object_fucntion():
    simulated_space = create_simulated_space()
    #####################Step Object#####################
    K = 1
    h = np.zeros_like(l)
    for counter, element in enumerate(simulated_space):
        if element < 0:
            h[counter] = 1
    h = K * h
    simulated_space *= 1e-9
    phase_shift = K * h * np.pi
    amp = 1

    return amp * np.exp(1j * phase_shift)
```


