# EXPH

EXPH_v 3.1

Author: Bowen Hou

Creating Time: 10/03/2022

Last Updated: 09/23/2023

---
## Exciton-Phonon Interaction

**Functions**:

1. Exciton lifetime over 1st BZ could be calculated

2. Exciton Scattering Rate

3. Exciton-Phonon Matrix 

4. rt-Boltzmann Equation (***new!***, and parallel is coming soon!)

**Other Properties**:

1. Parallel is available!

2. Interpolation is available (parallel is working for interpolation!)

3. Symmetry is available for finite Q!

4. Exciton Band Input is available for BGW! (see Tutorial second section finite Q-BGW)

5. Tutorial is available! 

6. Time complexity of exciton lifetime will be reduced to O(N^2) from O(N^2) (coming soon!) However, space complexity will increase.

7. GPU acceleration is available for ODE and PDE (only for small system, used for benchmark and test)!! 

8. CPU parallel is available for PDE!! 

9. Phase issue is solved!! (*new!*)

10. G_qm (exciton-phonon interaction) could be plotted along high symmetry line (*new!*)

## Workflow

Please read Tutorial/Tutorial.md to learn how to use this tool! 

![avatar](./fig/workflow.png)
---

## Convergence Test

(a) Convergence test for delta function (approximated as a Gaussian function) vs. Interpolation 

![avatar](./fig/Gaussian_Interpolation_Convergence.png)

---


## Parallel Efficiency of EX_PH_scat.py (Calculate Scattering Rate of Exciton State over 1st BZ):

(a) Test on PC Windows

| mpi task | wall time (s) | CPU time (s) |
|:--------:|:-------------:|:------------:|
|    1     |    197.743    |   196.203    |
|    2     |    124.767    |   120.906    |
|    4     |    83.884     |    82.344    |
|    6     |    66.071     |    64.609    |
|    8     |    53.952     |    48.594    |

(b) Test on Frontera

| mpi task    | wall time (s)    | CPU time (s) |
| :---: |   :---:       | :---: |
| 1 | 280.088 | 279.437|
|2 | 156.251 | 155.896|
|4 |77.328 | 77.139|
| 8 | 39.579| 39.215|
|16| 19.389 |19.276|
| 32 | 10.068|  9.885|
|40 | 8.068 | 8.057|
|48| 6.130  | 6.119|
|56| 6.382 | 6.148|

![avatar](./fig/para_eff.jpg)

---
Some Results:

(i) Exciton-Phonon Matrix:
![avatar](./fig/Exciton_Phonon.png)

(ii) Exciton Life Time over the first BZ
![avatar](./fig/lifetime.png)

(iii) Real Time Exciton Scattering in Reciprocal Space 
![avatar](./fig/occupation_evoluation.gif)

(iv) Real Space and Time Evolution of Two Exciton State Model:  
![avatar](./fig/simple_model.gif)


---
Todo:

 <!--(i) check everything of EX_PH_scat and EX_PH_lifetime
 
 (ii) normalization has been added, test more about it 
    
    (i) convergence
    (ii) compare it with non-renormalized situation!

 (iii) add some input file and rading system

 (iv) double check this: skip q=0 and omega = 0-->