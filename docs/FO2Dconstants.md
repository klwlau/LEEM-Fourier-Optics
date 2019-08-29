# Parameters in 2D FO Simulation

## Computing Parameters

`resultFileNote`: A string which will be added to the saved `.npy` result file name.  
`timezone`: Recorded time's timezone, list of timezones can be found [here](https://stackoverflow.com/questions/13866926/is-there-a-list-of-pytz-timezones).   
`numberOfThreads`: Number of threads used in the simulation with -1 means using all threads, -2 means using all threads minus 1.

## Simulation Parameters
`simulatingSpaceSize`: The 2D simulation space in both direction measure in nanometer(nm).  
`simulatingSpaceTotalStep`: Simulating space steps in both direction.  

## LEEM Parameters
`U_a`: Accelerating Voltage in eV  
`U_o`: Electron Voltage in eV  
`delta_E`: Energy Spread in eV  
`C_3`, `C_5`: Spherical Aberration Coefficient in meter   
`C_c`, `C_3c`, `C_cc`: Chromatic Aberration Coefficient in meter  
`alpha_ill`: Illumination Divergence Angle in radian  
`alpha_ap`: Acceptance Aangle of the Contrast Aperture in radian  
`M_L`: Lateral Magnification  
`defocus`: Defocus current in mA  
