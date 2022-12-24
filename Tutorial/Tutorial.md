# Tutorial

This tutorial will take you through all the basic steps of using EXPH post-process code to calculate exciton-phonon 
related properties, such as exciton band structure,exciton-phonon scattering matrix and non-radiative exciton lifetime.

---
### 1. Installation and Setup
 - Installation
   - You can directly download [EXPH.zip](https://github.com/Qiu-Group/EXPH.git) from our github. You can also type: 
   ``git clone git@github.com:Qiu-Group/EXPH.git`` in command line to get this package. 
   - Move the package to anywhere you want:  `mv ./EXPH ~/your_software_path/`
   - Then install EXPH:   
   `cd ~/your_software_path/EXPH/;`  
   `bash install.sh`
   - Add `./bin` to your environment
 
 - Required package:
    - Install [Anaconda](https://www.anaconda.com/) firstly
 
    - Then ``conda install mpi4py``

---    

### 2. Finite-Q Exciton from BGW
In this step, we will use BGW to generate finite-Q exciton wavefunction A(c,v,S,Q,k) and exciton dispersion on a uniform 
Q-grid (such as 24*24*1), which will be used in later exciton-phonon matrix calculation.


- DFT (QE)  
  (1) What are we doing here?
  
  In order to diagonalize the finite-Q BSE, we need two sets of k-grid: one is unshifted k-grid and the other is shifted 
  k-grid. The shifted vector is exactly negative momentum Q of exciton.

  Since a uniform Q-grid is needed in the further exciton-phonon calculation, we need to generate a lot of shifted k-grid WFN 
  (they will be stored in *./4.2-wfnq_co-n/WFN*) corresponding to all Q points in the 1st BZ of exciton. And all finite-Q BSE 
  could share same unshifted k-grid, we only need to generate one unshifted WFN (it will be stored in *./4.1-wfn_co_fullgrid/WFN*). 
  
  It seems pretty troublesome right? Don't worry, I have already written some script to help you automatically doing this!
  
  (2) Practical Steps

  - 
  -
  -
 
- GW + BSE (BGW)  
In this step, xxx
  - 
  -
  -

### 3. Electron-Phonon Matrix from EPW

 - Electron-Phonon Matrix (EPW)

### 4. Exciton-Phonon Interaction