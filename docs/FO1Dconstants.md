# Parameters in 1D FO Simulation

## Computing Parameters

`resultFileNote`: A string which will be added to the saved `.npy` result file name.     
`numberOfThreads`: Number of threads used in the simulation with -1 means using all threads, -2 means using all threads minus 1.

## Simulation Parameters
`simulatingSpaceSize`: The 1D simulation space in both direction measure in nanometer(nm).  
`simulatingSpaceTotalStep`: Simulating space steps in both direction.  

## LEEM Parameters
| Parameter | Type   | Description  |
|---:      |:---:  |:---|
|`U_a`|`float`| Accelerating Voltage in eV|  
|`U_o`|`float`| Electron Voltage in eV|  
|`delta_E`|`float`| Energy Spread in eV|  
|`C_3`, `C_5`|`float`| Spherical Aberration Coefficient in meter|   
|`C_c`, `C_3c`, `C_cc`|`float`| Chromatic Aberration Coefficient in meter|  
|`alpha_ill`|`float`| Illumination Divergence Angle in radian|  
|`alpha_ap`|`float`| Acceptance Aangle of the Contrast Aperture in radian|  
|`M_L`|`float`| Lateral Magnification|
|--------------|---|--------------|  
|`defocus_current_series`|`float`| Defocus current series measured in mA| 
|--------------|or|--------------|
|`delta_z_series`|`float`| Real space distance series between the sample to the objective lens (if using delta_z directly, pls disable line 49-51 and enable line 55 in `FO1Dconstants.py`)|
|--------------|---|--------------|

## Optimum LEEM Parameters

LEEM parameters such as `C_3`, `C_5`, `C_c`, `C_3c` and `C_cc` are energy dependent. User can recalculate the optimum parameters according to this [paper](https://www.sciencedirect.com/science/article/abs/pii/S0304399111002294).
(Tony's comment: This paper don't have the parameters for `C_3c` and `C_cc`.)

There are calculated aberration coefficient for IBM LEEM system at image plane *M* = 1, sample diatance *L* = 1.5mm and microscope potential *U_a* = 15010eV in table 1 in [Ultramicroscopy, **115**, 88–108 (2012)](https://www.sciencedirect.com/science/article/pii/S030439911100266X).

For  **Non-aberration-corrected**:

| U_o (eV) | C_3   | C_5  | C_c    | C_3c   | C_cc  |
|---:      |:---:  |:---: |:---:   |:---:   |:---:  |
| 1        | 0.492 | 768  | -0.13  | -1484  | 719   |
| 10       | 0.345 | 39.4 | -0.075 | -59.37 | 23.09 |
| 30       | 0.305 | 14.5 | -0.052 | -16.12 | 4.58  |

For  **Aberration-corrected**:     

| U_o (eV) | C_3 | C_5  | C_c | C_3c  | C_cc |
|---:      |:---:|:---: |:---:|:---:  |:---: |
| 1        | 0   | 749  | 0   | -1433 | 731  |
| 10       | 0   | 92.8 | 0   | -67.4 | 27.9 |
| 30       | 0   | 66.4 | 0   | -23.2 | 8.2  |


