# Setup the object function in 2D FO Simulation

To setup the object function, user can config a numpy array (np.array) in `FO2Dsample.py`. 
The object funciton is called `simulatedObject` that is generated within a function called 
`create2DSimulatedObject()`. Alternatively, user can load the object function from a np.array file `.npy` that the user 
prepared separately. (A sample script is prepared in [here](#Loading-object-function-from-separate-file))


      

## Preview object function
To preview the object function user can run `FO2Dsample.py` by itself, a plot of `simulatedObject` is generated accordingly.

For example we generate a square step object with amplitude 1 with the object 0 outside:

```python 

def create2DSimulatedObject():
    # define a zero array with dimension 501 by 501.
    simulatedObject = np.zeros((501, 501))

    #define the square step object
    amp = 1
    simulatedObject[251 - 50:251 + 50, 251 - 50:251 + 50] = amp

    return simulatedObject

```


After that we run `FO2Dsample.py` in terminal:

    python FO2Dsample.py
    
Output:

![preview sample](./img/viewSample.PNG)

## Loading object function from separate file


Here is a sample code for loading a object function stored in `objectFunction.npy` to the simulation.

```python 
import numpy as np

def create2DSimulatedObject():
    simulatedObject = np.load("objectFunction.npy")
    return simulatedObject

```