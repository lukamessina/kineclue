KineCluE - Kinetic Cluster Expansion (README file)
Thomas Schuler, Luca Messina, Maylise Nastar
Version 1.0 2018-06-26

Licence information
-------------------
KineCluE - Copyright (C) 2018 CEA, École Nationale Supérieure des Mines de Saint-Étienne.
This program comes with ABSOLUTELY NO WARRANTY; for details, see the license terms.
This is free software, and you are welcome to redistribute it under certain conditions.
See the LGPL license terms for details (COPYING and COPYING.LESSER files).
You are required to cite the following paper when using KineCluE or part of KineCluE.
T. Schuler, L. Messina and M. Nastar, Computational Materials Science (2019) [doi: https://doi.org/10.1016/j.commatsci.2019.109191]  

Contact information
-------------------
thomas.schuler -at- cea.fr
luca.messina -at- cea.fr
maylise.nastar -at- cea.fr

Quick start instructions (detailed information in the KineCluE_1.0_User_Manual):
------------------------
You will need a working version of Python 3.6 (along with some modules, see file KineCluE_1_0_Input_documentation.pdf).
If you do not have Python 3.6 installed on your computer, the Anaconda package is the easiest to get it along with the required modules.
Edit the first line of both 'kineclue_main.py' and 'kineclue_num.py' files and indicate the path to your Python 3.6 executable.
The code is divided in two parts: the analytical part (kineclue_main.py) which must be ran first and the numerical part (kineclue_num.py) which is ran afterwards.
Launching the code is very simple from a terminal (you can use a Python editor instead):
>> kineclue_main.py 'main_input_file'
and afterwards:
>> kineclue_num.py 'num_input_file'
where 'main_input_file' (resp. 'num_input_file') is the location of the input file for the analytical (resp. numerical) part of the program.
The KinecluE_1.0_User_Manual provides detailed information to create your own input files. 
To begin with, you can use the files provided in the 'Examples' folder, which are examples systems.




