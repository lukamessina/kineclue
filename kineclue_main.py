#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
KineCluE - Kinetic Cluster Expansion
T. Schuler, L. Messina, M. Nastar
kineclue_main.py (analytical calculation of transport coefficients)

Copyright 2018 CEA, École Nationale Supérieure des Mines de Saint-Étienne

This file is part of KineCluE.
KineCluE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
KineCluE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License
along with KineCluE. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import sympy as sym
import time as tm
import kinepy as kp
import _pickle as pickle
import os
import sys
import logging
import datetime
import copy
import resource
from itertools import compress, product
from shutil import copyfile

# Check for correct number of arguments
if len(sys.argv) != 2:
    raise Exception("ERROR! Incorrect call of {}. Correct usage: << ./kineclue_main input_file_path >>".format(sys.argv[0]))

sys.setrecursionlimit(kp.recursionlimit)  # for Pickle, else it causes error to save important data
start_time = tm.time() # Start measuring execution time

# Get input file from first argument (#1)
myinput = sys.argv[1]

# Read input file
if os.path.exists(myinput):
    input_string = open(myinput, "r").read()
else:
    raise Exception ("ERROR! Input file {} not found.".format(myinput))

# Remove #-comments from input file (and remove new-line characters)
input_string = " ".join([string.split(sep='#')[0] for string in input_string.split(sep='\n') if len(string.split(sep='#')[0]) > 0])

# Definition of input_dataset dictionary
keywords = ['crystal', 'basis', 'strain',  'cpg', 'normal', 'directory', 'kineticrange', 'uniquepos', 'species', 'bulkspecies',
            'jumpmech', 'iniconf', 'printxyz', 'printsymeqs', 'runnum', 'kiraloop']
input_dataset = {key: None for key in keywords}  # dictionary for reading user input
# Split input in keywords
input_list = input_string.split(sep='& ')
del input_list[0]  # delete all comments before first keyword
# Save input data into input_dataset
for ipt in input_list:
    keyword = ipt.split()[0].lower()  # get keyword
    input_dataset[keyword] = [i for i in ipt.split()[1:]]  # split all entries of each keyword in a list
del input_string, input_list, ipt

# Read output directory path from input
if input_dataset['directory'] is None:  # default directory is ./CALC/
    input_dataset['directory'] = ['./CALC/']
directory = input_dataset['directory'][0]
if directory == "cwd":
    directory = os.getcwd() + "/"
else:
    # Add a slash to the end of the directory path if it's missing
    if directory[-1] != "/":
        directory += "/"
# Save actual directory to input_dataset['directory'][0] for use in kp.produce_strain_input();
# the user-input string goes into input_dataset['directory'][1]
input_dataset['directory'].insert(0, directory)
# Create output directory
if not os.path.isdir(directory):
    os.makedirs(directory)

# Copying input file into directory
copy_input_path = directory + myinput.split("/")[-1]
if ".txt" not in myinput:
    copy_input_path += ".txt"
if not os.path.exists(copy_input_path):  # necessary when using the 'cwd' directory option
    copyfile(myinput, copy_input_path)

# Setting up logfile
logger = logging.getLogger('kinecluelog')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(directory + 'kineclue_main.log', mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(message)s'))
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(' __________________________________________________')
logger.info(' |                                                |')
logger.info(' |           KineCluE v{} - {}           |'.format(kp.version, kp.date))
logger.info(' |        T. Schuler, L. Messina, M. Nastar       |')
logger.info(' |________________________________________________|')
logger.info('')
kp.print_license_notice()
logger.info('')
logger.info('Calculation date: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
logger.info('Working in directory:  {}'.format(directory))

# Check that required information is in the input
if input_dataset['kineticrange'] is None:
    kp.produce_error_and_quit("!! You MUST provide a KINETICRANGE keyword.")
if input_dataset['species'] is None:
    kp.produce_error_and_quit("!! You MUST provide a SPECIES keyword.")
if input_dataset['jumpmech'] is None:
    kp.produce_error_and_quit("!! You MUST provide a JUMPMECH keyword.")
if input_dataset['crystal'] is None:
    kp.produce_error_and_quit("!! You MUST provide a CRYSTAL keyword.")
if input_dataset['uniquepos'] is None:
    kp.produce_error_and_quit("!! You MUST provide a UNIQUEPOS keyword.")

# Convert kin range to float and round up to avoid numerical issues
KiRa = float(input_dataset['kineticrange'][0]) + 0.01
logger.info("Kinetic range is set to {:.2f} lattice parameters".format(KiRa))
if len(input_dataset['kineticrange']) > 1:
    ThRa = float(input_dataset['kineticrange'][1]) + 0.01
else:
    ThRa = KiRa
logger.info("Thermodynamic range is set to {:.2f} lattice parameters".format(ThRa))

# Create crystal
crystal_input = input_dataset['crystal']  # !! Change of variable: crystal becomes an object
# SaveCheck for basis atoms (in a list of arrays)
if input_dataset['basis'] is None:
    logger.info("!! Bravais lattice because (BASIS keyword not found)")
    input_dataset['basis'] = ['s', '1', '0', '0', '0', '0']  # default crystal basis: Bravais lattice
n_basis_atoms =int(input_dataset["basis"][1])

# Check dimensionality of the crystal from input file
dim = int(np.sqrt(len(crystal_input)-1))
if n_basis_atoms > 1:
    logger.info('Creating {:.0f}D crystal {} with {:.0f} basis atoms'.format(dim, crystal_input[0], n_basis_atoms))
else:
    logger.info('Creating {:.0f}D crystal {}'.format(dim, crystal_input[0]))
if len(crystal_input) == 2: #1D crystal
    crystal_input += ['0.0', '0.0', '0.0', '6.5713', '0.0', '0.0', '0.0', '3.5187'] # adding small components in y and z directions
elif len(crystal_input) == 5: #2D crystal, adding the third direction, perpendicular to the others
    crystal_input = crystal_input[0:3] + ['0.0'] + crystal_input[3:5] + ['0.0', '0.0', '0.0', '5.4618']
elif len(crystal_input) == 10: # 3D crystal, nothing to add
    pass
else:
    kp.produce_error_and_quit("ERROR! Wrong definition of the crystal. Must be a string (name of the crystal), followed by 1, 4 or 9 numbers depending on the dimensionality of the system.")
# Create new crystal
crystal_input[1:10] = [kp.evalinput(i) for i in crystal_input[1:10]]
crystal = kp.Crystal(name=crystal_input[0], vec1=np.array(crystal_input[1:4]), vec2=np.array(crystal_input[4:7]),
                    vec3=np.array(crystal_input[7:10]), dim=dim)
# Set basis list and atomic volume
basis_list = []
for i in range(2, len(input_dataset["basis"])):
    input_dataset['basis'][i] = kp.evalinput(input_dataset['basis'][i])
for ba in range(n_basis_atoms):
    basis_list.append(np.array(input_dataset["basis"][ba*4+2:ba*4+6], dtype=float))
    if input_dataset["basis"][0] == "o":  # convert position to supercell base
        basis_list[ba][1:4] = kp.change_base(arr=basis_list[ba][1:4], crystal=crystal)
crystal.set_basislist(basis_list)
crystal.set_atomicvolume()

# Find all valid symmetry operations
# crys_symop_list is a dictionary with following objects as keys: crystal, crystal_deformed
logger.info("Searching for symmetry operations in crystal without strain...")
tmp_symop_list = kp.find_symmetry_operations(crystal=crystal)
# Saving (perfect) crystal object to file
pickle.dump([crystal, tmp_symop_list], open(directory + 'crystal_' + crystal.get_name() + '.pkl', 'wb'), -1)

# Find dumbbell orientations if the "d" symbol appears in uniquepos definitions
if 'd' in ''.join(input_dataset['uniquepos']): # dumbbell calculation
    # Finding dumbbell orientations
    for i in range(0, int(input_dataset['uniquepos'][0])):
        if 'd' in ''.join(input_dataset['uniquepos'][i*4+2:i*4+5]):  # this defect is a dumbbell
            dbcoords = []  # list of symmetry-unique orientations
            for k in range(0, 3):
                dbcoords.append(kp.evalinput(input_dataset['uniquepos'][i*4+2+k]+'-'+input_dataset['uniquepos'][i*4+2+k].replace('d','0')))
            dbcoords = np.array(dbcoords)/kp.db_displ
            if input_dataset['uniquepos'][i*4+1] == 'o':  # change base to supercell for distance calculation
                dbcoords = kp.change_base(arr=dbcoords, crystal=crystal)
            # Compute max distance between two atoms in the dumbbell (used in kp.Configuration.findjumps)
            # 'max' is relevant if more orientations are defined (for instance, <110> and <111>)
            kp.maxdbdist = max(kp.maxdbdist, kp.tol_pbc+kp.distance(vect1=dbcoords, vect2=-dbcoords, crystal=crystal))
            logger.info("!! This calculation contains a dumbbell with orientation [{:4.1f} {:4.1f} {:4.1f}] ([{:4.1f} {:4.1f} {:4.1f}] in supercell coordinates)".format(*kp.change_base(arr=dbcoords, crystal=crystal, inv=True), *dbcoords))
else:
    kp.maxdbdist = 1.0

# Check if input file contains non-zero strain
strain_flag = False
if input_dataset['strain'] is not None:
    if ''.join(input_dataset['strain']).find('e') >= 0:
        logger.info("!! Non-zero strain found.")
        strain_flag = True
if not strain_flag:
    logger.info('!! Unstrained calculation (STRAIN keyword not found)')
# Apply strain to crystal for strain calculations
strcrystal = copy.copy(crystal)  # strained crystal
if strain_flag:
    # Read strain
    if len(input_dataset['strain']) != dim*dim+1:
        kp.produce_error_and_quit("ERRROR! Found strain but the dimensionality is not consistent with that of crystal")
    if dim == 1:  # 1D crystal
        input_dataset['strain'] += ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']  # adding 0.0 components to have 3x3 matrix
    elif dim == 2:  # 2D crystal, adding the third direction, perpendicular to the others
        input_dataset['strain'] = input_dataset['strain'][0:3] + ['0.0'] + input_dataset['strain'][3:5] + ['0.0', '0.0', '0.0', '0.0'] # adding 0.0 components to have 3x3 matrix
    input_dataset['strain'][1:10] = [kp.evalinputS(i) for i in input_dataset['strain'][1:10]]
    strain_matrix = np.transpose(np.array([input_dataset['strain'][1:4], input_dataset['strain'][4:7],
                                           input_dataset['strain'][7:10]]))  # (strain vectors are in column)
    if input_dataset['strain'][0] is 'o':  # Convert strain to supercell base
        strain_matrix = kp.change_base(arr=strain_matrix, crystal=crystal, matrix=True)
    # Apply strain to crystal vectors
    strcrystal.set_strain(strain_matrix)
    # Update atomic volume( N.B. Everything that is expressed in supercell basis does not change with strain!)
    strcrystal.set_atomicvolume()
    # Find all valid symmetry operations
    logger.info("Searching for symmetry operations in strained crystal...")
    tmp_symop_list = kp.find_symmetry_operations(crystal=strcrystal)
    # Save deformed crystal to file
    pickle.dump([strcrystal, tmp_symop_list], open(directory + 'crystal_' + crystal.get_name() + '_strained' + '.pkl', 'wb'), -1)
    # Update input_dataset dictionary to add new defects and jumps arising from broken symmetries, and produce strained input file
    input_dataset = kp.produce_strain_input(dataset=input_dataset, input_file=myinput)
    # from here on, the code uses the updated strained input_dataset
    del strain_matrix

del input_dataset['kineticrange'], crystal_input, input_dataset['strain']
del input_dataset['basis'], basis_list, input_dataset['crystal']

# Save and convert CPG to supercell base if necessary
if input_dataset['cpg'] is not None:
    cpg_input = input_dataset['cpg']  # Change of variable: cpg becomes an array
else:
    logger.info("!! Setting default CPG direction as [1,0,0] in supercell basis. Might not be optimal (CPG keyword not found)")
    cpg_input = ['s', '1', '0', '0']  # Default CPG direction: 1st crystal direction
cpg = np.array([kp.evalinput(i) for i in cpg_input[1:4]], dtype=float)
if cpg_input[0] is 's':  # If CPG is given in the supercell base, switch to orthonormal to make normalization
    cpg = kp.change_base(arr=cpg, crystal=strcrystal, inv=True)
cpg /= np.linalg.norm(cpg)  # Normalization in the orthonormal base
Ndirs = 1 # number of directions in which the flux is computed
diffusion_dir = [cpg.copy()]  # this is a list of three items, each being a diffusion direction in the orthonormal base
logger.info('CPG direction (supercell, cartesian) [{:6.3f} {:6.3f} {:6.3f}] ; [{:6.3f} {:6.3f} {:6.3f}]'.format(*kp.change_base(arr=cpg, crystal=crystal), *diffusion_dir[0]))
if input_dataset['normal'] is not None:
    dir1 = np.array(input_dataset['normal'][1:4], dtype=float)
    if input_dataset['normal'][0] is 's': # If NORMAL is given in the supercell base, switch to orthonormal to make normalization
        dir1 = kp.change_base(arr=dir1, crystal=strcrystal, inv=True)
    dir1 /= np.linalg.norm(dir1)  # Normalization in the orthonormal base
    dir2 = np.cross(cpg, dir1)
    if kp.are_equal_arrays(A=dir2, B=0*dir2):
        logger.info('!! The NORMAL and CPG keywords give parallel diffusion directions.')
        logger.info('!! Cluster transport coefficients are computed along the cpg direction only (NORMAL keyword ignored)')
    else:
        Ndirs = 3
        dir2 /= np.linalg.norm(dir2)
        diffusion_dir.append(dir1.copy())
        diffusion_dir.append(dir2.copy())
        logger.info('Normal direction (supercell, cartesian) [{:6.3f} {:6.3f} {:6.3f}] ; [{:6.3f} {:6.3f} {:6.3f}]'.format(
            *kp.change_base(arr=diffusion_dir[1], crystal=crystal), *diffusion_dir[1]))
        logger.info('Normal direction (supercell, cartesian) [{:6.3f} {:6.3f} {:6.3f}] ; [{:6.3f} {:6.3f} {:6.3f}]'.format(
            *kp.change_base(arr=diffusion_dir[2], crystal=crystal), *diffusion_dir[2]))
else:
    logger.info('!! Cluster transport coefficients are computed along the cpg direction only (NORMAL keyword not found)')
del cpg_input, input_dataset['normal']

# Find symmetry operations that maintain the cpg symmetry
symop_cpg_list = kp.find_cpg_symmetries(crystal=strcrystal, symop_list=tmp_symop_list, cpg=diffusion_dir[0])

# Write symop_list, containing all valid symmetry operations: first the ones that maintain the cpg symmetry, then all the others
symop_list = kp.flat_list([symop_cpg_list, list(compress(tmp_symop_list, [j.get_cpgsign() is None for j in tmp_symop_list]))])
del tmp_symop_list
logger.info("  Found {} symmetries that include the cpg.".format(len(symop_cpg_list)))

# Create list of defects from UNIQUEPOS
# N.B. In strain calculations, the defect lists are computed with the deformed crystal, but
# this is not an issue because in strain calculations the input file has been treated with the
# produce_strain_input function, which gives the defect positions in the supercell basis. Also, if
# the strain input is provided directly by the user, then it's correct to use the deformed crystal.
logger.info("Creating list of species and components, starting at {:.3f} s".format(tm.time()-start_time))
[defect_list, all_defect_list, doubledefects] = kp.dataset_to_defects(dataset=input_dataset, crystal=crystal, symop_list=symop_list)
n_defects = len(defect_list)
# defect_list contains all symmetry unique defects
# all_defect_list contains all defects
# doubledefects contains list of user-input defects that are symmetrically equivalent
# (they do not appear in defect_list/all_defect_list and need to be removed from the permission list in species - this is done in dataset_to_species)

# Create lists of species and components
[species_list, component_list] = kp.dataset_to_species(dataset=input_dataset, defect_list=defect_list, doubledefects=doubledefects)
n_species = len(species_list) - 1  # bulk is not included in the number of species
n_components = len(component_list)
# Analyze special "bulkspecies" if the bulkspecies input has been defined by the user
# (bulkspecies are symmetric species that lead to a double counting of the symmetry equivalent configurations - e.g. pure dumbbells)
if input_dataset['bulkspecies'] is not None:
    # This function creates a set of global bulkspecies variables stored in kinepy (see function in kinepy for more details)
    kp.analyze_bulkspecies(component_list=component_list, all_defect_list=all_defect_list)

# Creating list of component label permutations (for configuration names)
specs = []
for cp in component_list:
    specs.append(cp.get_species())  # list of species for each components
name_perms = kp.find_possible_permutations(specs=specs, species_list=species_list)
del cp, specs, doubledefects

# Create list of jumps (jump symmetries and constraints are analyzed in dataset_to_jumps)
# Jump_list contains complete list of jumps, including symmetry equivalents
logger.info("Creating list of jumps, starting at {:.3f} s".format(tm.time()-start_time))
jump_list = kp.dataset_to_jumps(dataset=input_dataset, crystal=crystal, symop_list=symop_list, all_defect_list=all_defect_list, species_list=species_list, sym_unique=False, component_list=component_list)
n_jumps = len(jump_list)  # total number of jump mechanisms (including symmetry equivalent ones)
# Create jump dictionary, giving for each jump label the corresponding JumpMech object (for efficient comparisons between jumps)
jump_catalogue = {}
for j in jump_list:
    jump_catalogue[j.get_label()] = j
# For each jump in jump_list, find symmetry equivalents and compute the net species displacements
logger.info("Computing jump symmetry equivalent and average species displacements.. (starts at {:.3f} s)".format(tm.time() - start_time))
for j in jump_list:
    j.set_symeqs(symops=symop_list, jump_catalogue=jump_catalogue, all_defect_list=all_defect_list)
    j.set_net_species_displacement(strcrystal=strcrystal, directions=diffusion_dir, spec=species_list)  # species net displacement, including projections on cpg
    # Find defect type and integer translation of each constraint, and record them in the JumpConstraint object
    for k in j.get_constraints():
        k.vect2deftrans_constraint(all_defect_list=all_defect_list)
# Compute average species displacements
for i, j in enumerate(jump_list):
    if j.get_summednetdisp() is None:
        j.compute_summednetdisp(spec=len(species_list)-1, Ndirs=Ndirs)  # average species displacement over all symmetry equivalents
    if j.get_cpgsummednetdisp() is None:
        j.compute_cpgsummednetdisp(spec=len(species_list)-1, symopcpg=symop_cpg_list, Ndirs=Ndirs)  # same as above, but averaging displacement projections on cpg
del j, k, input_dataset['jumpmech']

# Prints symmetry equivalents for defects and jumps in symeqs file
if input_dataset['printsymeqs'] is not None:
    nobulk = "bulk"
    if len(input_dataset['printsymeqs']) == 0:
        input_dataset['printsymeqs'].append('s')
    elif len(input_dataset['printsymeqs']) == 2:
        if input_dataset['printsymeqs'][1] == "bulk":
            nobulk = ""
    logger.info("Writing symeqs file with symmetry equivalents for defects and jumps (starts at {:.3f} s)".format(tm.time() - start_time))
    with open(directory + 'symeqs.dat','w') as output:
        if input_dataset['printsymeqs'][0] == 'o':
            output.writelines("Printing symmetry equivalents in orthonormal coordinates\n")
        else:
            output.writelines("Printing symmetry equivalents in supercell coordinates\n")
            for e in range(0, 3):
                output.writelines('  {:8.4f} {:8.4f} {:8.4f}\n'.format(*crystal.get_primvectors()[e]))
        output.writelines("\n### DEFECTS ###\n")
        for d in all_defect_list:
            if input_dataset['printsymeqs'][0] == 'o':
                output.writelines("Defect {:4.0f}: {:8.4f} {:8.4f} {:8.4f}\n".format(d.get_sudefect().get_index(), *kp.change_base(crystal=crystal, inv=True, arr=d.get_sublattice())))
            else:
                output.writelines("Defect {:3.0f}: {:8.4f} {:8.4f} {:8.4f}\n".format(d.get_sudefect().get_index(), *d.get_sublattice()))
        output.writelines("\n### JUMPS ###\n")
        for j in jump_list:
            ncons = 3*len([c for c in j.get_constraints() if c.get_inispecies().get_name() != nobulk])
            if input_dataset['printsymeqs'][0] == 'o':
                output.writelines(
                    ("Jump {:s}: " + "{:8.4f} "*ncons + " > " + "{:8.4f} "*ncons + "\n").format(
                        j.get_name(), *kp.flat_list([kp.change_base(arr=c.get_iniposition(), crystal=crystal, inv=True) for c in j.get_constraints() if c.get_inispecies().get_name() != nobulk]),
                        *kp.flat_list([kp.change_base(arr=c.get_finposition(), crystal=crystal, inv=True) for c in j.get_constraints() if c.get_inispecies().get_name() != nobulk])))
            else:
                output.writelines(("Jump {:s}: " + "{:8.4f} "*ncons + " > " + "{:8.4f} "*ncons + "\n").format(
                    j.get_name(),*kp.flat_list([c.get_iniposition() for c in j.get_constraints() if c.get_inispecies().get_name() != nobulk]),
                    *kp.flat_list([c.get_finposition() for c in j.get_constraints() if c.get_inispecies().get_name() != nobulk])))

# Check that defects, components and jumps create a connected exploration space (i.e. each defect type is reachable by a list of jumps)
logger.info("Checking if exploration space is connected.. (starts at {:.3f} s)".format(tm.time()-start_time))
for sp in species_list[:-1]:  # loop on all species but bulk atoms
    mylist = []  # list of defects accessible by species sp and appearing in all jump mechanisms
    for j in jump_list:  # loop on symmetry unique (???) jump list
        for cons in j.get_constraints(): # loop on constraint for each jump
            if cons.get_inispecies() == sp:
                mylist.append([cons.get_inidefect().get_sudefect(), cons.get_findefect().get_sudefect()])
            if cons.get_finspecies() == sp:
                mylist.append([cons.get_findefect().get_sudefect(), cons.get_inidefect().get_sudefect()])
    found = [sp.get_defects()[0]] # sp starts from its first allowed defect; is it able to reach all other defects?
    k0 = 0
    while k0 < len(found):
        for k in mylist: # loop on shortlisted jump mechanisms
            if k[0] == found[k0]: # this is the good initial defect, let's try to add the final defect for this jump
                found0 = False
                for j in found: # checking that this defect has not been already added to the list
                    if k[1] == j:
                        found0 = True
                        break
                if not found0:
                    found.append(k[1])
        k0 += 1
    if len(found) == len(sp.get_defects()): # checking whether list of defects for this species and list of accessible defects via jump mechanisms are identical
        # Comparison between list of objects - Sorting is done by the defect index (can be changed if needed)
        if not np.all(sorted(sp.get_defects(), key=lambda defect: defect.get_index()) == sorted(found, key=lambda defect: defect.get_index())):
            logger.info("!! Inconsistencies between SPECIES and JUMPMECH tags for species {}: some configurations seem unreachable".format(sp.get_name()))
    else:
        logger.info("!! Inconsistencies between SPECIES and JUMPMECH tags for species {}: some configurations seem unreachable".format(sp.get_name()))
del sp, j, cons, mylist, k0, k, found, found0

# Create initial configuration
logger.info("Setting up initial configuration.. (starts at {:.3f} s)".format(tm.time()-start_time))
inivalid = False
if input_dataset['iniconf'] is not None:
    if not (int(input_dataset['iniconf'][0]) == n_components):
        logger.info("!! Invalid number of components for the initial configuration")
    else:
        inivalid = True
        tmp = []
        for comp in range(0, int(input_dataset['iniconf'][0])):  # reordering components based on species labels
            tmp.append(input_dataset['iniconf'][(1 + 5 * comp):(6 + 5 * comp)])
        iniconf = [input_dataset['iniconf'][0]] + kp.flat_list(sorted(tmp, key=lambda x: int(float(x[1]))))
        for comp in range(0, int(iniconf[0])):  # loop over components
            vec = np.array([kp.evalinput(e) for e in iniconf[5 * comp + 3:5 * comp + 6]], dtype=float)  # from string list to float array
            if iniconf[5 * comp + 1] == 'o':
                vec = kp.change_base(arr=vec, crystal=crystal)  # base change to supercell coordinates
            # Setting up translation vectors such that the first component has (000) translation, and all other components are expressed in the reference frame of the first one
            if comp == 0:
                mylist = kp.vect2deftrans(vec=vec, all_defect_list=all_defect_list)  # changing the component's vector to defect index and integer supercell translation vector
                tvec = np.array(mylist[1], dtype=float)
                def_list = [mylist[0]]
                trans_list = [np.array([0, 0, 0])]
            else:
                mylist = kp.vect2deftrans(vec=vec - tvec, all_defect_list=all_defect_list)  # changing the component's vector to defect index and integer supercell translation vector
                def_list = def_list + [mylist[0]]
                trans_list.append(np.array([int(e) for e in mylist[1]]))
        # Creating initial configuration
        iniconf_label, def_list, trans_list = kp.set_confname(defects=def_list, translations=trans_list, perms=name_perms,species=[c.get_species for c in component_list])
        myiniconf = kp.Configuration(defects=copy.copy(def_list), translations=copy.copy(trans_list))
        # Check if initial configuration is a) consistent (site occupancies and sublattice permissions), b) connected
        if not kp.check_configuration_consistence(label=iniconf_label, component_list=component_list, all_defect_list=all_defect_list, crystal=strcrystal):
            logger.info("!! The initial configuration provided in the input is not consistent in terms of site occupancy and/or sublattice permissions. I'll create a new one..")
            inivalid = False
            del myiniconf
        else:
            connex, _, _, myinikira = kp.check_connectivity(configuration=myiniconf, kira=KiRa, crystal=strcrystal)
            if not connex:
                logger.info("!! The initial configuration provided in the input is not connected. I'll create a new one..")
                inivalid = False
                del myiniconf
        del vec, tvec, mylist, tmp

# Create initial configuration, if it was not given in the input or the one provided is not connected or consistent
if (input_dataset['iniconf'] is None) or (not inivalid):
    logger.info("!! Automatically generating the initial configuration, starting at {:.3f} s".format(tm.time()-start_time))
    def_list = [component_list[0].get_species().get_defects()[0].get_symeq()[0]]  # take first available symmetry equivalent of first defect type of first component
    trans_list = [np.array([0, 0, 0])]  # first component is placed at the center
    # Place other components in cluster, trying out all translations (in order of distance) until a valid one is found
    if len(component_list) > 1:
        spec_list = [component_list[0].get_species()]  # list of species (to be used in subconfiguration_consistence check)
        pos_list = [def_list[0].get_sublattice() + trans_list[0]]  # list of positions (to be used in subconfiguration_consistence check)
        # Create list of all possible translations
        # ("product" generates all combinations of 3 elements picking from range(-3,4) with repetitions)
        iterlist = [np.array(x, dtype=int) for x in product(range(-3, 4), repeat=3)]
        iterlist = sorted(iterlist, key=lambda x: x[0] * x[0] + x[1] * x[1] + x[2] * x[2])  # sort translations by distance from center
        # For each component, try out all possible defects and translations until a consistent configuration is found
        for comp in component_list[1:]:  # loop over remaining components
            check = False
            idx_iter = -1
            while not check and idx_iter < len(iterlist)-1:  # loop over possible translations in iterlist
                idx_iter += 1
                defsyms = [sym for d in comp.get_species().get_defects() for sym in d.get_symeq()]  # all symmetry equivalents of all defects of this component
                idx_syms = -1
                while not check and idx_syms < len(defsyms)-1:  # loop over possible symmetry equivalents in defsyms
                    idx_syms += 1
                    tmp_pos = defsyms[idx_syms].get_sublattice() + iterlist[idx_iter]  # tentative position for this component
                    # Check consistence of the subconfiguration made of all components placed so far plus the tentative one
                    if kp.check_subconfiguration_consistence(species_list=spec_list + [comp.get_species()], position_list=pos_list + [tmp_pos], all_defect_list=all_defect_list, crystal=crystal):
                        # This component can be placed here! Update all relevant lists
                        def_list.append(defsyms[idx_syms])
                        trans_list.append(iterlist[idx_iter])
                        spec_list.append(comp.get_species())
                        pos_list.append(tmp_pos)
                        check = True
            if not check:  # this component could not be placed anywhere - quit job
                kp.produce_error_and_quit("Unable to set up the initial configuration automatically.")
        del iterlist, spec_list, pos_list, comp, check, idx_iter, idx_syms, defsyms, tmp_pos
    # Create initial configuration
    iniconf_label, def_list, trans_list = kp.set_confname(defects=def_list, translations=trans_list, perms=name_perms, species=[c.get_species() for c in component_list])
    myiniconf = kp.Configuration(defects=copy.copy(def_list), translations=copy.copy(trans_list))
    logger.info("New initial configuration (orthonormal coordinates):\n" + "  {}".format(["{}: [{:8.4f} {:8.4f} {:8.4f}]".format(comp.get_species().get_name(), *kp.change_base(arr=myiniconf.get_defect_position(c), crystal=crystal, inv=True)) for c, comp in enumerate(component_list)]))
    connex, _, _, myinikira = kp.check_connectivity(configuration=myiniconf, kira=KiRa, crystal=crystal)
    if not connex:
        kp.produce_error_and_quit("!! The initial configuration computed automatically is not connex.")
else:
    logger.info("User initial configuration (orthonormal coordinates):\n" + "  {}".format(["{}: [{:8.4f} {:8.4f} {:8.4f}]".format(comp.get_species().get_name(), *kp.change_base(arr=myiniconf.get_defect_position(c), crystal=crystal, inv=True)) for c, comp in enumerate(component_list)]))
# Find available jumps from initial configuration
myiniconf.set_species([c.get_species() for c in component_list])
myiniconf.findjumps(alljumps=jump_list, components=component_list, crystal=crystal, all_defect_list=all_defect_list, KiRa=KiRa, name_perms=name_perms)
# Stop execution if no valid jumps from initial configuration are found
if len(myiniconf.get_jumplist()) != 0:
    logger.info("{} valid jumps from initial configuration (orthonormal coordinates)".format(len(myiniconf.get_jumplist())))  # Number of jumps from the initial configuration
else:
    kp.produce_error_and_quit("!! No valid jumps found from initial configuration.")
# Print configurations that are accessible from the initial configuration
for inijump in myiniconf.get_jumplist():
    if inijump[0][0] != 'd':
        # orthonormal coordinates
        _, finpos = kp.untranslatedfinal(jumpmech=inijump[1], component_list=component_list, iniconf=myiniconf, init=True, name_perms=name_perms,
                                         finconf=kp.name2conf(name=inijump[0], all_defect_list=all_defect_list, species=[c.get_species() for c in component_list]))
        positions = [list(a) for a in [kp.change_base(inv=True, crystal=crystal, arr=b) for b in finpos]]
        #positions = [list(kp.change_base(inv=True, crystal=crystal, arr=kp.name2conf(name=inijump[0],
        #            all_defect_list=all_defect_list, species=[c.get_species() for c in component_list]).get_defect_position(e))) for e in range(len(component_list))]
        logger.info('  ' + inijump[1].get_name() + '  [' + " , ".join([" ".join(["{:8.4f}".format(
                                                np.round(e,4)) for e in lista]) for lista in positions]) + ']')
    else:
        # orthonormal coordinates
        positions = [list(kp.change_base(inv=True, crystal=crystal, arr=kp.name2conf(name=inijump[0].split(sep="$")[0], all_defect_list=all_defect_list,
                    species=[c.get_species() for c in component_list]).get_defect_position(e))) for e in range(len(component_list))]
        logger.info('  ' + inijump[1].get_name() + '  [' + " , ".join([" ".join(["{:7.4f}".format(
                                                np.round(e, 4)) for e in lista]) for lista in positions]) + '] - dissociated')

del def_list, trans_list, inijump, inivalid, input_dataset['iniconf'], myiniconf

# END READING INPUT AND FORMATTING DATA-------------------------

# Creating configuration space starting from iniconf and applying successive jumps to symmetry unique configurations
logger.info("Creating configuration space (starts at {:.3f} s)".format(tm.time() - start_time))
thermoidx = -1
kinidx = 0
threp_list = []  # symmetry unique configurations
kirep_list = []  # cpg-symmetry unique configurations
null_kirep_list = []  # cpg-symmetry unique configurations with zero contribution
freq_list = {}  # list of jump frequencies (w)
config_list = {}  # comprehensive configuration list
disso_list = {}  # dissociation configurations
subconfig_list = {}  # dictionary of dictionnaries for subconfigs of a chemical composition and size
subperms = {}  # dictionary of species permutations for subconfigurations

# Begin exploration of connected configurations, starting from iniconf
tmp_list = [[iniconf_label, myinikira]]  # configurations that need to be explored (labels)
del myinikira
maxkira = {}
while len(tmp_list) > 0:
# while(thermoidx)<1000: #other possibility to stop the loop
    # Check if the first configuration in tmp_list has already been found. If true, remove it from tmp_list and repeat.
    # If all configurations in tmp_list are already known, tmp_list will be progressively emptied and the loop will stop.
    if config_list.get(tmp_list[0][0]) is not None:
        del tmp_list[0]
        continue
    # making sure that the configuration is not dissociated
    if tmp_list[0][0][0] == 'd': # new dissociated configuration
        newdisso = kp.name2conf(name=tmp_list[0][0].split(sep='$')[0], all_defect_list=all_defect_list,
                     species=[c.get_species() for c in component_list])
        newdisso.find_subconfigurations(component_list=component_list, crystal=crystal,
                                        ThRa=ThRa, disso=True)  # looking for subconfigurations
        name, subconfig_list, subperms = newdisso.findsubname(species_list=species_list, all_defect_list=all_defect_list,
                                                              symop_list=symop_list, symop_cpg_list=symop_cpg_list,
                                                              subconfig_list=subconfig_list, subperms=subperms)
        if disso_list.get(name) is None:
            thermoidx += 1
            newdisso.set_thermoint(thermoidx)
            newdisso.set_kinint1(0)
            disso_list[name] = copy.copy(newdisso) #deepcopy
            config_list[name] = disso_list[name]
        # changing names of dissociated configurations in jump lists
        threp_list[int(tmp_list[0][0].split(sep='$')[1])][0].replace_jumplist(old=tmp_list[0][0], new=name)
        del tmp_list[0]
        continue # no exploration from dissociated configurations

    # Find all symmetry equivalent configurations of tmp_list[0], create the objects, and set their thermodynamic and kinetic interaction classes
    results = kp.set_interaction_classes(conf=kp.name2conf(name=tmp_list[0][0], all_defect_list=all_defect_list, species=[c.get_species() for c in component_list]),
                                         thermo_index=thermoidx, kinetic_index1=kinidx, symops=symop_list,
                                         symops_cpg=symop_cpg_list, all_defect_list=all_defect_list,
                                         name_perms=name_perms)
    # Update all lists and quantities that need to be updated
    config_list.update(results[0])  # dictionary: configuration name -> Configuration object
    threp_list = threp_list + results[1]  # list of Configuration objects
    for cidx in threp_list[-1][1]:
        cidx.set_posthreplist(len(threp_list) - 1)
    kirep_list = kirep_list + results[2]  # list of Configuration objects + number of kinetic symmetry equivalents
    null_kirep_list = null_kirep_list + results[5]  # list of Configuration objects
    thermoidx = results[3]
    kinidx = results[4]
    if input_dataset['kiraloop'] is not None:
        if (float(tmp_list[0][1])-ThRa) >= -0.01: #if above thermodynamic radius
            if maxkira.get(tmp_list[0][1]) is None: # maxkira is a dictionary containing pairs threp_list indexes and kinetic interaction indexes
                maxkira[tmp_list[0][1]] = []
            maxkira[tmp_list[0][1]] += [[len(threp_list)-1, c[0].get_kinint1()] for c in results[2]]
            if len(results[2])==0:
                maxkira[tmp_list[0][1]] += [[len(threp_list) - 1, None]]
    # For each symmetry unique kinetic interaction (first instances of newly added items to kirep_list and null_kirep_list), look for all jumps and accessible configurations
    for conf in results[2]+results[5]:
        conf[0].findjumps(alljumps=jump_list, components=component_list, crystal=crystal,
                          all_defect_list=all_defect_list, KiRa=KiRa, name_perms=name_perms)
    # Add newly found configurations to tmp_list
    tmp_list = tmp_list + [[e[0],e[2]] for e in results[1][0][0].get_jumplist()]
    # Delete the configuration that was just dealt with from tmp_list
    del tmp_list[0]
# Replacing dissociated configuration names in threp_list
for inter in threp_list:
    toreplace = [j[0] for j in inter[0].get_jumplist() if j[0][0] == 'd']
    for interk in inter[1][1:]:  # changing the name in threp list to be able to find it when identifying jumps
        interk.replace_disso_jumplist(toreplace)

# Setting up site interactions to account for chemical potential relaxation
siteinter_index = 0
if n_components > 1: # if monomer, than site interactions and effective interactions are similar; no need for both
    for sudefect in defect_list:
        siteinter_index = sudefect.find_siteinter_classes(species_list=species_list, index=siteinter_index, symopcpg=symop_cpg_list, all_defect_list=all_defect_list)
    del sudefect
logger.info('  Found {} configurations'.format(len(config_list)))
logger.info('  Found {} thermodynamic interaction classes (including {} dissociated)'.format(thermoidx+1, len(disso_list)))
logger.info('  Found {} kinetic interaction classes (and {} site interactions)'.format(kinidx, siteinter_index))

# Reducing the number of configuration and jump frequency variables according to ThRa
thconfig_list = {}  # dictionary relating configuration labels with configuration objects (unique with respect to ThRa)
th2label = {}  # dictionary relating thermodynamic interactions with configuration labels
if ThRa < KiRa:
    logger.info("Using the thermodynamical range to reduce the number of variables (starts at {:.3f} s)".format(tm.time() - start_time))
    thermoidx = -1
    for disso in disso_list:
        thermoidx += 1 # renumbering thermodynamic interactions for dissociated configurations
        disso_list[disso].set_thermoint(thermoidx)
        th2label[str(thermoidx)] = disso_list[disso].get_label()
        thconfig_list[disso_list[disso].get_label()] = disso_list[disso]
    for conf in threp_list:
        # assigning subconfigurations to each element in threp_list
        conf[0].find_subconfigurations(component_list=component_list, crystal=crystal, ThRa=ThRa)
        if len(conf[0].get_subconfigurations()) == 0:  # we already know for sure that it is symmetrically unique
            thermoidx += 1
            th2label[str(thermoidx)] = conf[0].get_label()
            thconfig_list[conf[0].get_label()] = conf[0]
            for symconf in conf[1]:
                symconf.set_thermoint(thermoidx)
                symconf.set_beyond()
        else:
            name, subconfig_list, subperms = conf[0].findsubname(species_list=species_list,
                                                                  all_defect_list=all_defect_list,
                                                                  symop_list=symop_list, symop_cpg_list=symop_cpg_list,
                                                                  subconfig_list=subconfig_list, subperms=subperms)
            if disso_list.get(name) is not None:
                disso_list[name].set_kineticinter(conf[0].get_kineticinter())
                for symconf in conf[1]:
                    symconf.set_thermoint(disso_list[name].get_thermoint())
            else:
                kp.produce_error_and_quit("!! In ThRa routine, a dissociated configuration was not found previously")
    logger.info("  Now {} configuration variables are left".format(thermoidx+1))
else:
    logger.info("!! Thermodynamic interaction range not specified or equal to or larger than the kinetic range")
    for disso in disso_list:
        th2label[str(thermoidx)] = disso_list[disso].get_label()
        thconfig_list[disso_list[disso].get_label()] = disso_list[disso]
    for conf in threp_list:
        thconfig_list[conf[0].get_label()] = conf[0]
        th2label[str(conf[0].get_thermoint())] = conf[0].get_label()
        for symconf in conf[1]:
            symconf.set_beyond()

# Searching for jump frequencies
logger.info("Searching for jump frequencies (starts at {:.3f} s)".format(tm.time() - start_time))
Nfreq = 0
for conf in kirep_list+null_kirep_list:  # for all sample configurations of kinetic interactions
    for idx, jump in enumerate(conf[0].get_jumplist()):
        freq_output = kp.id_jump_freq(config1=conf[0], config2=config_list[jump[0]], jumpmech=jump[1],
                                      freq_list=freq_list, symop_list=symop_list,
                                      name_perms=name_perms, all_defect_list=all_defect_list, Nfreq=Nfreq)
        conf[0].set_jumpfrequency(pos=idx, freq=freq_output[0])
        # If it's a new jump frequency, add it to freq_list[label] = JumpFreq object
        if freq_output[1]:
            Nfreq += 1
            if freq_list.get(freq_output[2]) is None:
                freq_list[freq_output[2]] = [freq_output[0]]
            else:
                freq_list[freq_output[2]].append(freq_output[0])
logger.info('  Found {} jump frequencies'.format(Nfreq))

# Now that we know how many jump frequencies we have we can set up jump frequency labels as components of a symbolic matrix
W = sym.MatrixSymbol('W', Nfreq, 1)
dissocoeff = {} # preparing dictionary of dissociation frequencies
for freq0 in freq_list:
    for freq in freq_list[freq0]:
        freq.set_label(W[freq.get_number(), 0])
        if freq.get_disso():
            dissotype = '|'.join([a.split(sep='s')[0] for a in freq.get_config_fin().get_subname()[1:].split(sep='|')])
            freq.set_dissotype(dissotype)
            if dissocoeff.get(dissotype) is None:
                dissocoeff[dissotype] = len(dissocoeff)
del conf, idx, freq_output, jump, tmp_list, kinidx, results

# Adding thermodynamic interactions to maxkira dictionary. Each key is the corresponding kira distance.
# The dictionary returns a list of interactions each corresponding to this particular kira
# Each "interaction" is represented itself by a list containing 3 integers:
# kinetic interaction index, thermodynamic interaction index (after ThRa modifications), kinetic interaction multiplicity
for key in maxkira:
    kitr = [] # kinetic interaction to remove from Klambda and T; list of kinetic interaction indexes
    for idx, k in enumerate(maxkira[key]):
        if k[1] is not None:
            kitr.append(kirep_list[k[1]-1][0].get_kinint1())
    ctrfz = {} # configurations to remove from the partition function
    tmp = [(threp_list[f][0], len(threp_list[f][1])) for f in set([e[0] for e in maxkira[key]])]
    for k in tmp:
        if ctrfz.get(k[0].get_thermoint()) is None:
            ctrfz[k[0].get_thermoint()] = k[1]
        else:
            ctrfz[k[0].get_thermoint()] += k[1]
    jtr = {} # jumps to remove from lambda_0; keys are kinetic radius for final configuration
    for k in tmp:
        for j in k[0].get_jumplist():
            if jtr.get(j[2]) is None:
                jtr[j[2]] = {} # dictionary keys: jump frequency index; returns summednetdisp*number of equivalent threp
            if jtr[j[2]].get(j[4].get_number()) is None:
                jtr[j[2]][j[4].get_number()] = j[1].get_summednetdisp()*k[1]
            else:
                jtr[j[2]][j[4].get_number()] += j[1].get_summednetdisp()*k[1]
    for key2 in jtr: # loop over kira distance of final configuration
        for w in jtr[key2]: # loop over jump frequency indexes
            l0 = np.zeros((Ndirs, n_species, n_species), dtype=object)
            for direction in range(Ndirs):
                for alpha in range(n_species):
                    for beta in range(n_species):
                        l0[direction, alpha, beta] = sym.simplify(jtr[key2][w][direction*n_species*n_species + alpha*n_species + beta])
            jtr[key2][w] = l0
    maxkira[key] = [copy.copy(kitr), copy.copy(ctrfz), copy.copy(jtr)]
if len(maxkira.keys())>0:
    logger.info("Found {} radius values in between the thermal range and the kinetic range".format(len(maxkira.keys())))
else:
    logger.info("!! Will not be able to perform the convergence study (KIRALOOP keyword not found)")

# Writing equations
logger.info("Writing symbolic equations (starts at {:.3f} s)".format(tm.time() - start_time))
removedint = []
Tcoords = []
Tsparse = []
Tvector = np.zeros((1, len(kirep_list)+1), dtype=object)
Kvec = np.zeros((Nfreq, n_species * Ndirs), dtype=object)
if siteinter_index > 0:
    Dmap = np.zeros((len(kirep_list),siteinter_index), dtype=object)
    dissocontrib = np.zeros((siteinter_index, siteinter_index), dtype=object)
    dlambda = np.zeros((Ndirs, n_species, siteinter_index), dtype=object)
if len(kirep_list)==0: # If there is no kinetic interaction, create 1-element T and K matrices
    Tdiag = -np.ones((1, 1), dtype=object)
    Klambda = np.zeros((Ndirs, n_species, 1), dtype=object)
else: # otherwise, create T and K matrices with proper dimensions
    Tdiag = np.zeros((len(kirep_list), 1), dtype=object)
    Klambda = np.zeros((Ndirs, n_species, len(kirep_list)), dtype=object)
# Begin loop over kinetic interactions
for interaction in kirep_list:
    Tvector *= 0
    Kvec *= 0
    eq_idx = np.abs(interaction[0].get_kinint1())
    eq_sign = np.sign(interaction[0].get_kinint1())
    for conf_fin in interaction[0].get_jumplist():  # Loop over all final configurations that are reached from initial configuration "interaction"
        # T matrix # conf_fin = [ final configuration label, jump mechanism, jump frequency W ]
        Tvector[0, eq_idx] += - conf_fin[4].get_label()*eq_sign  # Contribution from initial configuration (negative) (conf_fin[3] contains the Jump Frequency W[x,x]
        Tvector[0, np.abs(config_list[conf_fin[0]].get_kinint1())] += conf_fin[4].get_label()*np.sign(config_list[conf_fin[0]].get_kinint1())  # Contribution from final configuration (positive)
        # M and K matrices
        for species in range(n_species):  # Loop over species for projection on CPG direction
            for dir0 in range(Ndirs):
                Kvec[conf_fin[4].get_number(), dir0*n_species+species]+= interaction[1]*conf_fin[1].get_cpgsummednetdisp()[species][dir0]

    # Checking that this interaction is non-zero
    if np.all(Kvec[:, 0:n_species] == 0) and np.all(Tvector[0, 1:eq_idx]==0) and np.all(Tvector[0, eq_idx + 1:]==0):
        removedint.append(eq_idx)
        thconfig_list[th2label[str(interaction[0].get_thermoint())]].get_kineticinter().remove(eq_idx)
    else:
        # storing the resulting vector in sparse notation (for the upper triangle part) and vector for the diagonal part
        Tdiag[eq_idx-1-len(removedint), 0] = Tvector[0, eq_idx]*interaction[1]
        for b, val in enumerate(Tvector[0][eq_idx+1:]):
            if val != 0:
                Tsparse.append(val*interaction[1])
                Tcoords.append((eq_idx-1-len(removedint), b+eq_idx))
        # symbolic equations for Mvectors and Klambda
        for species in range(n_species):
            for freq in range(Nfreq):
                for dir0 in range(Ndirs):
                    if Kvec[freq, dir0*n_species+species] != 0:
                        Klambda[dir0, species, eq_idx-1-len(removedint)] += Kvec[freq, dir0*n_species+species]*W[freq, 0]
        # Site interactions
        if siteinter_index > 0:
            for conf_fin in interaction[0].get_jumplist():  # Loop over all final configurations that are reached from initial configuration
                if conf_fin[0][0]=='d':
                    Sdisso = np.zeros((1,siteinter_index), dtype=int)  #site interaction signs for dissociated configurations
                    for icp, cp in enumerate(component_list):
                        Sdisso[0,np.abs(conf_fin[5][icp].get_siteinter()[cp.get_species()]) - 1] += np.sign(conf_fin[5][icp].get_siteinter()[cp.get_species()])
                        Dmap[eq_idx-1-len(removedint), np.abs(conf_fin[5][icp].get_siteinter()[cp.get_species()])-1] += \
                                np.sign(conf_fin[5][icp].get_siteinter()[cp.get_species()]) * conf_fin[4].get_label() * interaction[1]
                    for dir0 in range(Ndirs):
                        for species in range(n_species):
                            dlambda[dir0, species, :] -= interaction[1]*conf_fin[1].get_cpgsummednetdisp()[species][dir0]*conf_fin[4].get_label()*Sdisso[0]
                    dissocontrib -= np.dot(Sdisso.transpose(), Sdisso) * conf_fin[4].get_label() * interaction[1]

if len(removedint)>0:  # if some kinetic interactions were removed
    for interaction in thconfig_list.values():  # for other configurations
        for idx, k in enumerate(interaction.get_kineticinter()):
            interaction.get_kineticinter()[idx] -= len([e for e in removedint if e < k])
    # Renumbering Tcoords
    todel = []
    for idx, interaction in enumerate(Tcoords):
        if interaction[1]+1 in removedint:
            todel.append(idx)
        else:
            Tcoords[idx] = (interaction[0], interaction[1]-len([e for e in removedint if e < interaction[1]+1]))
    todel = sorted(todel, reverse=True)
    for idx in todel:
        del Tcoords[idx]
        del Tsparse[idx]
    del todel
    # Removing extra Klambda columns
    for _ in removedint:
        Tdiag = np.delete(Tdiag, -1, axis=0)
        Klambda = np.delete(Klambda, -1, axis=2)
        if siteinter_index > 0:
            Dmap = np.delete(Dmap, -1, axis=0)
    if Tdiag.shape[0]==0 and Klambda.shape[2]==0: # if there are no kinetic interactions left
        Tdiag = -np.ones((1, 1), dtype=object)
        Klambda = np.zeros((Ndirs, n_species, 1), dtype=object)
        corr = -1 #correcting the number of non-zero elements (for print purpose only)
        if siteinter_index > 0:
            Dmap = np.zeros((1, siteinter_index), dtype=object)
    # Renumbering maxkira lists:
    for key in maxkira:
        for k in range(len(maxkira[key][0])-1, -1, -1):
            if maxkira[key][0][k] in removedint:
                del maxkira[key][0][k]
            else:
                maxkira[key][0][k] -= len([e for e in removedint if e < maxkira[key][0][k]])
    logger.info("  Removed {} kinetic interaction(s)".format(len(removedint)))
logger.info("  T matrix contains {} non-zero elements".format(2*len(Tsparse)+len(Tdiag)+np.min([1,len(kirep_list)])-1))

# Site correction
if siteinter_index > 0:
    SiteCorr = [Dmap, dlambda, dissocontrib]
else:
    SiteCorr = []

# Compute L0matrix (non-correlated part)
logger.info("Computing non-correlated matrix (starts at {:.3f} s)".format(tm.time() - start_time))
L0matrix = np.zeros((Ndirs, n_species, n_species), dtype=object)
FreqCoefArr = np.zeros((Ndirs * n_species * n_species, Nfreq), dtype=object)
dissoFreqArr = np.zeros((len(dissocoeff), Nfreq), dtype=object)
for threp in threp_list:
    for jump in threp[0].get_jumplist():
        FreqCoefArr[:, jump[4].get_number()] += jump[1].get_summednetdisp() * len(threp[1])
        if jump[4].get_disso():
            dissoFreqArr[dissocoeff[jump[4].get_dissotype()],jump[4].get_number()] += len(threp[1])
for a in dissocoeff:
    dissocoeff[a] = W.as_mutable().dot(dissoFreqArr[dissocoeff[a],:])
for direction in range(0, Ndirs):
    for alpha in range(0,n_species):
        for beta in range(0,n_species):
            L0matrix[direction, alpha, beta] = W.as_mutable().dot(FreqCoefArr[direction*n_species*n_species+alpha*n_species+beta,:])

# Compute partition function
logger.info("Computing partition function (starts at {:.3f} s)".format(tm.time() - start_time))
z_function = 0
Carray = np.zeros((thermoidx+1,), dtype=int)
for idx, threp in enumerate(threp_list):
    Carray[threp[0].get_thermoint()] += len(threp[1])
C=sym.MatrixSymbol('C',thermoidx+1, 1)
z_function = C.as_mutable().dot(Carray)
del Carray

# Writing configuration file (ThRa)
logger.info("Writing configuration values file (starts at {:.3f} s)".format(tm.time() - start_time))
conf_file = directory + "configurations.txt"
disfmt="{:^"+str(2*(len(species_list)-1)*len(component_list)+2)+"s}"
with open(conf_file, 'w') as output:
    output.writelines("1) Configuration class; 2) Entropy prefactor (no units); 3) Binding energy (eV, >0 means attraction);"+
                      "4) Dissociated; 5) Number of symmetry equivalents; 6) Position of each defect in orthonormal coordinates\n")
    for conf in thconfig_list:
        tofile = []
        tofile += ["{:<5.0f}".format(thconfig_list[conf].get_thermoint())] #configuration thermodynamic interaction
        tofile += ["{:5.3f}".format(-1)]  # default prefactor
        tofile += ["{:4.3f}".format(0)] #default binding energy
        output.writelines("{:s}  {:s}  {:s} ".format(*tofile))
        if thconfig_list[conf].get_label()[0] == 'd':
            #output.writelines(" dissociated ")  # specifying that it is a dissociated configuration
            output.writelines(disfmt.format('d'+'|'.join([a.split(sep='s')[0] for a in thconfig_list[conf].get_subname()[1:].split(sep='|')])))  # specifying that it is a dissociated configuration
        else:
            output.writelines(disfmt.format("-"))
        n = 0 # writing the number of configurations with this energy
        for conf2 in threp_list:
            if conf2[0].get_thermoint() == thconfig_list[conf].get_thermoint():
                n += len(conf2[1])
        output.writelines("{:10.0f} ".format(n))
        tofile = []
        for idefe, defe  in enumerate(thconfig_list[conf].get_defects()):  # print total position of each defect in orthonormal base
            pos = kp.change_base(arr=thconfig_list[conf].get_defect_position(def_idx=idefe), crystal=crystal, inv=True)
            if np.sum(np.abs(defe.get_sublattice()-defe.get_sublattice_nod()))<1e-8:
                tofile += ["[{:+.2f} {:+.2f} {:+.2f}]".format(pos[0], pos[1], pos[2])]
            else:
                tofile += ["[{:+.4f} {:+.4f} {:+.4f}]".format(pos[0], pos[1], pos[2])]
        output.writelines(["%s" % item + ' ' for item in tofile])
        output.writelines("%s\n" % "")

# Writing jump frequency file (ThRa)
logger.info("Writing jump frequency values file (starts at {:.3f} s)".format(tm.time() - start_time))
freq_file = directory + 'jump_frequencies.txt'
fnfmt = "{:^"+str(np.max([len(j.get_name()) for j in jump_list])+2)+"s}"
with open(freq_file, 'w') as output:
    output.writelines("1) Jump frequency number; 2) Jump prefactor (no units) 3) Saddle-point energy (eV);"+
        " 4) Configuration class 1; 5) Configuration class 2; 6) Jump mechanism name; 7) Number of symmetry equivalent jumps\n")
    for freq0 in freq_list:
        for freq in freq_list[freq0]:
            tofile = []
            tofile += ["{:<5.0f}".format(freq.get_number())]
            tofile += ["{:5.3f}".format(-1)] #default prefactor
            tofile += ["{:4.3f}".format(0)] #default binding energy
            tofile += ["{:^5.0f}".format(freq.get_config_ini().get_thermoint())]
            tofile += ["{:^5.0f}".format(freq.get_config_fin().get_thermoint())]
            tofile += [fnfmt.format(freq.get_jump().get_name())]
            tofile += ["{:5.0f}".format(len(freq.get_symeq()))]
            output.writelines("{:s}  {:s}  {:s} {:s} {:s} {:s} {:s}\n".format(*tofile))
del tofile, output, freq, conf

# Writing configuration and jump frequency coordinates files if specified by the user in the input
if input_dataset['printxyz'] is not None:
    logger.info("Writing configuration and jump frequency xyz files (starts at {:.3f} s)".format(tm.time() - start_time))
    kp.printxyz(direc=directory, input=input_dataset['printxyz'], thconfig_list=thconfig_list, name_perms=name_perms,
                n_components=n_components, crystal=crystal, component_list=component_list, freq_list=freq_list)

# Remove unnecessary attributes from Configuration objects to reduce memory demand
uniqueconf = [0 for _ in range(thermoidx+1)]
for conf in thconfig_list:
    thconfig_list[conf].cleanconfig()
    uniqueconf[thconfig_list[conf].get_thermoint()] = thconfig_list[conf]

# Remove unnecessary attributes from JumpFrequency objects to reduce memory demand
uniquefreq = [0 for _ in range(Nfreq)]
for freq0 in freq_list:
    for freq in freq_list[freq0]:
        freq.cleanfreq()
        freq.get_config_ini().cleanconfig()
        freq.get_config_fin().cleanconfig()
        uniquefreq[freq.get_number()] = freq

# Saving objects and symbolic matrices
logger.info("Saving objects to file (starts at {:.3f} s)".format(tm.time() - start_time))
pickle.dump([Tcoords, Tsparse, Tdiag, L0matrix, Klambda, z_function,
         dissocoeff, ThRa, component_list, crystal.get_name(),
         all_defect_list, species_list[-1], uniqueconf, uniquefreq,
             maxkira, kp.bulkspecies_list, kp.bulkspecies_symdefects,
             kp.maxdbdist*kp.db_displ, SiteCorr, disfmt, fnfmt], open(directory + 'analytical_kineclue_output.pkl', 'wb'), -1)

# Stop measuring execution time and print elapsed time
stop_time = tm.time()
logger.info("Execution time: {:.3f} s.".format(stop_time - start_time))
logger.info("Peak memory usage: {:.3f} MB (or kB on MAC)".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e3))  # peak memory usage

if input_dataset['runnum'] is not None:
    if os.path.exists(input_dataset['runnum'][0]):
        os.rename(input_dataset['runnum'][0], input_dataset['runnum'][0]+".tmp")
        with open(input_dataset['runnum'][0], 'w') as output:
            for line in open(input_dataset['runnum'][0]+".tmp", "r").read().split("\n"):
                if '& directory' in line.lower():
                    output.writelines('& DIRECTORY '+ input_dataset['directory'][1] +'\n')
                else:
                    output.writelines(line+'\n')
        os.remove(input_dataset['runnum'][0]+".tmp")
        from kineclue_num import numrun
        numrun(myinput=input_dataset['runnum'][0], stream_log=False)
    else:
        kp.produce_error_and_quit("Numerical input file {} not found.".format(input_dataset['runnum'][0]))

# END CODE-----------------------------------
