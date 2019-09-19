"""
KineCluE - Kinetic Cluster Expansion
T. Schuler, L. Messina, M. Nastar
kinepy.py (module containing classes and functions)

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
import copy
import os
import _pickle as pickle
import mpmath as mp
import logging
import datetime
from scipy.sparse import linalg
from itertools import product, permutations, combinations
from shutil import copyfile

# Definition of variables
version = "1.0"
date = "13/09/2018"
tol = 1e-12  # numeric tolerance on the comparisons (machine error)
tol_strain = 1e-6
db_displ = 0.0003  #  fictitious displacement of each atom in a dumbbell (in lattice parameter units, with respect to dumbbell center)
maxdbdist = 0 # CHECK FOR POSSIBLE ROUNDING ERRORS FOR DUMBBELLS! HARD CODED '4' valid for 100,110,111 dumbbells.
              # The correct value should be sqrt(or.or) where or is the dumbbells orientation vector
              # Other possibility is to normalize dumbbells d distances to always have the same dumbbell bond length
tol_pbc = 0.001  # tolerance when applying periodic boundary conditions (to exclude dumbbell from automatic translations)
recursionlimit = 10000  # for Pickle, else it causes error to save important data
e = sym.Symbol('e')  # symbolic variable for strain
strainval = 0.00435427869  # numerical value for strain if needed
# Special variables to treat bulkspecies (symmetric species that lead to a double counting of the symmetry equivalent configurations - e.g. pure dumbbells)
bulkspecies_list = {}  # list of special "bulkspecies" (list of indices corresponding to the bulkspecies components)
bulkspecies_symdefects = {}  # dictionary where for each key (each defect in all_defect_list) corresponds the defect with the opposite sublattice, so that the first component is positive

# Definition of logger
logger = logging.getLogger('kinecluelog')

# Definition of classes
class Crystal:
    """The crystal object contains the primitive vectors (as numpy arrays) and the methods to perform a base change"""

    def __init__(self, name, vec1, vec2, vec3, dim):
        self.__name = name
        self.__primvectors = np.transpose(np.array([vec1, vec2, vec3],dtype='float'))   # Primitive vectors (vec1, vec2, vec3 in columns)
        self.__invvectors = np.linalg.inv(self.__primvectors)  # Compute inverse matrix of prim. vectors for base changes
        self.__strain = np.zeros((3,3), dtype=float)  # strain in the cell
        self.__orthostrain = np.zeros((3,3), dtype=float)  # strain in orthonormal coordinates
        self.__atomicvolume = 0
        self.__basislist = []
        self.__dimensionality = dim # 1D, 2D or 3D
        self.__primvectorsnum = self.get_primvectors()  # with symbolic strain, primvectors become object array, not float array
        self.__invvectorsnum = self.get_invvectors()  # with symbolic strain, invvectors become object array, not float array

    def get_name(self):
        return self.__name

    def get_primvectors(self):
        return self.__primvectors

    def get_primvectorsnum(self):
        return self.__primvectorsnum

    def get_invvectors(self):
        return self.__invvectors

    def get_invvectorsnum(self):
        return self.__invvectorsnum

    def get_strain(self):
        return self.__strain

    def get_orthostrain(self):
        return self.__orthostrain

    def get_basislist(self):
        return self.__basislist

    def get_atomicvolume(self):
        return self.__atomicvolume

    def get_dim(self):
        return self.__dimensionality

    def set_basislist(self, a):
        self.__basislist = a

    def set_atomicvolume(self):
        # self.__atomicvolume = np.abs(np.linalg.det(self.__primvectors))/len(self.__basislist)
        self.__atomicvolume = np.abs(sym.simplify(sym.Matrix(self.__primvectors).det())) / len(self.__basislist)

    def set_strain(self, strain: np.ndarray):  # Apply user-input strain
        self.__strain = strain
        self.__orthostrain = change_base(arr=strain, crystal=self, inv=True, matrix=True)
        self.__primvectors = np.dot(self.__primvectors, (np.identity(3) + self.__strain))   # Check this formula!
        self.__primvectorsnum = sym.matrix2numpy(sym.Matrix(self.__primvectors).subs(e, strainval), dtype='float')
        #self.__invvectors = np.linalg.inv(self.__primvectors)  # Could be written without computing linalg.inv
        self.__invvectors = sym.simplify(sym.Matrix(self.__primvectors).inv())
        self.__invvectorsnum = sym.matrix2numpy(self.__invvectors.subs(e, strainval), dtype='float')
        self.__invvectors = sym.matrix2numpy(self.__invvectors, dtype='object')


class Component:
    """A component belongs to a species which defines the possible defects (sublattice/orientation) available"""

    def __init__(self, species, index):
        self.__species = species   # species that the component belong to
        self.__index = index  # component index

    def get_species(self):
        return self.__species

    def get_sublattices(self):
        return tuple([o.get_sublattice() for o in self.__species.get_defects()])

    def get_defects(self):
        return flat_list([o.get_symeq() for o in self.__species.get_defects()])

    def get_info(self):
        logger.info("    Component {} is a {}". format(self.__index, self.__species.get_name()))


class Configuration:
    """A Configuration contains defect types and position for each component"""

    def __init__(self, defects, translations, label=None):
        self.__defects = defects  # List of defects (a defect for each component)
        self.__translations = translations   # Indices of the supercell where each component is sitting, besides the first one, which is always in [0, 0, 0].
        # Position is given by self.__translations * crystal_primvectors + self__defects
        self.__th0 = None  # Thermodynamic interaction
        self.__nu1 = None  # Kinetic interaction
        self.__jumplist = []  # List of list of accessible configurations and corresponding jump mechanism
        self.__label = label # name of the configuration in c[defects_][translations] or d[defects_][translations] if kinetically dissociated
        self.__subconfigurations = [] # list of subconfigurations to apply thermodynamic range reduction
        self.__species = [] # list of species, needed to compare subconfigurations
        self.__beyond = True  # true if this configuration is thermodynamically but not kinetically dissociated
        self.__kineticinter = [] # list of kinetic interaction indexes associated with this configuration
        self.__posthreplist = None
        self.__subname = None


    def get_defects(self):
        return self.__defects

    def get_translations(self):
        return self.__translations

    def get_defect_position(self, def_idx: int) -> np.ndarray:
        """Return defect position as sublattice+translation (defect order is that defined in list self.__defects)"""
        return np.array(self.get_defects()[def_idx].get_sublattice(), dtype=float) + np.array(self.get_translations()[3*def_idx:3*def_idx+3], dtype=float)

    def get_thermoint(self):
        return self.__th0

    def get_kinint1(self):
        return self.__nu1

    def get_jumplist(self):
        return self.__jumplist

    def get_label(self):
        return self.__label

    def get_kineticinter(self):
        return self.__kineticinter

    def get_posthreplist(self):
        return self.__posthreplist

    def set_posthreplist(self, a):
        self.__posthreplist = a

    def get_subname(self):
        return self.__subname

    def set_translations(self, a):
        self.__translations = a

    def set_kineticinter(self, a):
        if isinstance(a, list):
            self.__kineticinter.extend(a)
        else:
            self.__kineticinter.append(a)

    def set_thermoint(self, a):
        self.__th0 = a

    def set_kinint1(self, a):
        self.__nu1 = a

    def set_label(self, a):
        self.__label = a

    def set_jumpfrequency(self, pos, freq):
        self.__jumplist[pos][4]=freq

    def get_beyond(self):
        return self.__beyond

    def set_beyond(self):
        self.__beyond = False


    def cleanconfig(self):
        #self.__beyond = None
        self.__nu1 = None
        #self.__subconfigurations = None
        self.__species = None
        self.__jumplist = None

    def get_subconfigurations(self):
        return self.__subconfigurations

    def get_species(self):
        return self.__species

    def set_subconfiguration(self, a):
        self.__subconfigurations.append(a)

    def set_species(self, a):
        self.__species += a

    def find_subconfigurations(self, component_list: list, ThRa: float, crystal: Crystal, disso: bool=False):
        # Check if this threp configuration is still connected with the smaller ThRa
        [foundall, connected, disconnected, _] = check_connectivity(configuration=self, kira=ThRa, crystal=crystal)
        if not foundall:
            self.set_subconfiguration(Configuration(defects=[self.get_defects()[e] for e in connected], label='c',
                                                          translations=flat_list([self.get_translations()[3*e:3*e+3] for e in connected])))
            self.__subconfigurations[-1].set_species([component_list[e].get_species() for e in connected])
            translat = [np.array(self.get_translations()[3*e:3*e+3]) for e in disconnected]
            translat = flat_list([translat[e]-translat[0] for e in range(0, len(translat))])
            remains = Configuration(defects=[self.get_defects()[e] for e in disconnected], translations=translat)
            remaining_species = [component_list[e].get_species() for e in disconnected]
        while not foundall:
            [foundall, connected, disconnected, _] = check_connectivity(configuration=remains, kira=ThRa, crystal=crystal)
            self.set_subconfiguration(Configuration(defects=[remains.get_defects()[e] for e in connected], label='c',
                                        translations=flat_list([remains.get_translations()[3*e:3*e+3] for e in connected])))
            self.__subconfigurations[-1].set_species([remaining_species[e] for e in connected])
            translat = [np.array(remains.get_translations()[3*e:3*e+3]) for e in disconnected]
            translat = flat_list([translat[e]-translat[0] for e in range(0, len(translat))])
            remains = Configuration(defects=[remains.get_defects()[e] for e in disconnected], translations=translat)
            remaining_species = [remaining_species[e] for e in disconnected]

    def findsubname(self, species_list, all_defect_list, symop_list, symop_cpg_list, subconfig_list, subperms):
        for subconf in self.__subconfigurations: # looping over subconfigurations
            # Looking for species key
            spec_key = [0 for _ in species_list[:-1]]
            for sp in subconf.get_species():
                spec_key[sp.get_index() - 1] += 1
            spec_key = '_'.join(str(e) for e in spec_key)
            if subconfig_list.get(spec_key) is None:
                # New species key
                subconfig_list[spec_key] = {}
                # Looking for corresponding permutations
                subperms[spec_key] = find_possible_permutations(specs=subconf.get_species(), species_list=species_list)
            # Translation format to find subconfiguration name
            translat = [np.array(subconf.get_translations()[3*e:3*e + 3]) for e in range(0, len(subconf.get_species()))]
            # Looking for name
            name = set_confname(defects=subconf.get_defects(), translations=translat, perms=subperms[spec_key],
                                   species=subconf.get_species())[0]
            # Searching if subconfiguration already exists
            if subconfig_list[spec_key].get(name) is not None:
                subconf.set_thermoint(spec_key + 's' + str(subconfig_list[spec_key][name]))
            else:
                # If new, apply symmetry operations and fill-in dictionnary with name keys
                symeqs = apply_configuration_symmetry_operations(config=subconf, symops=symop_list,
                                                                    ncpgops=len(symop_cpg_list),
                                                                    name_perms=subperms[spec_key],
                                                                    all_defect_list=all_defect_list)
                inter = max([-1] + [subconfig_list[spec_key][e] for e in subconfig_list[spec_key]]) + 1
                for name in symeqs[0]:
                    subconfig_list[spec_key][name] = inter
                for name in symeqs[1]:
                    subconfig_list[spec_key][name] = inter
                subconf.set_thermoint(spec_key + 's' + str(inter))
        # Now set the name of the configuration and add it to thconfig_list if needed
        name = 'd'+'|'.join(sorted([e.get_thermoint() for e in self.__subconfigurations]))
        self.__subname = name
        return name, subconfig_list, subperms

    def replace_jumplist(self, old: str, new: str):
        for i, k in enumerate(self.__jumplist):
            if k[0] == old:
                self.__jumplist[i][0] = new

    def replace_disso_jumplist(self, toreplace: list):
        n = -1
        for i, k in enumerate(self.__jumplist):
            if k[0][0] == 'd': #disso jump; remove it and replace by another name from toreplace list
                n += 1
                self.__jumplist[i][0] = toreplace[n]

    def findjumps(self, alljumps: list, components: list, crystal: Crystal, all_defect_list: list, KiRa: float, name_perms: list):
        """Find possible jumps available from this configuration, and set up the jumplist attribute (list of: [Configuration object, Jump object]).
        The function tests each potential jump (contained in alljumps), searching for those that are applicable to this configuration,
        finding also the resulting configuration."""
        for jump in alljumps:  # loop over each jump
            k = 0
            cons0 = jump.get_constraints()[k]  # start by checking first constraint
            while cons0.get_inispecies().get_index() == 0:
                k += 1
                cons0 = jump.get_constraints()[k]  # looking for the first non-bulk constraint
            for comp0 in range(0, len(components)):  # loop over each component
                if components[comp0].get_species() == cons0.get_inispecies():  # check species
                    used = [None for _ in range(0, len(components))]  # for each component, equals None if the component was not used in a constraint, or equals the constraint index
                    # remember that all constraints have been translated so that the first constrained species is in the first supercell (no integer translation)
                    if self.__defects[comp0] == cons0.get_inidefect():  # check sublattice
                        tvec = -np.array(self.__translations[3 * comp0:3 * comp0 + 3], dtype=float)  # setting translation vector
                        used[comp0] = k
                        check = True
                        for l in range(k + 1, len(jump.get_constraints())):  # for now jump seems possible, let's check the other constraints
                            cons = jump.get_constraints()[l]
                            if cons.get_inispecies().get_index() == 0:
                                continue  # at this point we are not checking bulk constraints
                            check = False
                            for comp in range(0, len(components)):
                                if used[comp] is None:  # check that this component is not used for another constraint
                                    if components[comp].get_species() == cons.get_inispecies():  # check that this component has the correct species
                                        position = self.get_defect_position(comp) + tvec
                                        # position = np.array(self.__defects[comp].get_sublattice(), dtype=float) + np.array(self.__translations[3*comp:3*comp+3], dtype=float) + tvec
                                        if are_equal_arrays(position,cons.get_iniposition()):  # check that this component has the correct position
                                            check = True
                                            used[comp] = l
                                            break
                            if not check:  # jump is not possible because one non-bulk constraint is not met
                                break
                        # at this point, all non bulk constraints are met. The jump is possible if none of the remaining
                        # components ("False" statement in "used") is close (<2d) to any of the constrained sites
                        if check:  # let's see if one component that is not constrained is too close to a constrained site
                            for comp in range(0, len(components)):
                                if used[comp] is None:
                                    position = self.get_defect_position(comp) + tvec
                                    for cons in jump.get_constraints():
                                        if distance(vect1=cons.get_iniposition(), vect2=position, crystal=crystal) < maxdbdist * db_displ:
                                            # this component is too close from a constrained site! jump cannot be performed!
                                            check = False
                                            break
                                    if not check:
                                        break  # jump is not possible
                            if check:  # jump is possible!
                                translations = []
                                defects = []
                                for comp in range(0, len(components)):  # loops on the components in the configuration
                                    if used[comp] is None:  # this component has not moved
                                        translations = translations + self.__translations[3 * comp:3 * comp + 3]
                                        defects = defects + [self.__defects[comp]]
                                    else:
                                        newpos = self.get_defect_position(comp) + jump.get_constraints()[
                                            used[comp]].get_finposition() - jump.get_constraints()[
                                                     used[comp]].get_iniposition()
                                        # newpos = np.array(self.__translations[3*comp:3*comp+3], dtype=float) + self.__defects[comp].get_sublattice() + jump.get_constraints()[used[comp]].get_finposition()-jump.get_constraints()[used[comp]].get_iniposition()
                                        mylist = vect2deftrans(vec=newpos, all_defect_list=all_defect_list)
                                        defects = defects + [mylist[0]]
                                        translations = translations + list(mylist[1])
                                # Compute and store displacements of each component
                                delta_trans = np.array([translations[i*3:i*3+3] for i in range(len(components))], dtype=float) - np.array([self.get_translations()[i*3:i*3+3] for i in range(len(components))], dtype=float)
                                # Now we need to translate everyone so that the first component has 0 translation
                                tvec = -np.array(translations[0:3])
                                translations0 = []
                                for comp in range(0, len(components)):
                                    translations0.append(np.array(translations[3 * comp:3 * comp + 3]) + tvec)
                                # Final configuration name
                                conf, _, _ = set_confname(defects=defects, translations=translations0, perms=name_perms, species=self.__species)
                                # Creating temporary final configuration
                                tmp_conf = name2conf(name=conf, all_defect_list=all_defect_list, species=self.__species)
                                # Connectivity analysis
                                connect, _, _, maxkira = check_connectivity(configuration=tmp_conf, kira=KiRa, crystal=crystal)
                                if not connect:
                                    conf, _, _ = set_confname(defects=defects, translations=translations0,
                                                              perms=name_perms, disso=True, species=self.__species, thermoint=self.__posthreplist)
                                self.__jumplist.append([conf, jump, maxkira, delta_trans, None, copy.copy(defects)])  # conf label, jump mechanism, max kira, displacements of each component


class Defect:
    """A defect object is defined by a sub-lattice"""
    def __init__(self, crystal, sublattice, index, sublattice_nod):
        self.__crystal = crystal  # Crystal object in which the defect is defined
        self.__sublattice = sublattice  # position of the defect in the primitive cell
        self.__sublattice_nod = sublattice_nod # same but removing the dumbbell displacement if any
        self.__index = index  # index labeling each defect
        # List of symmetry equivalent defect objects in the supercell (e.g. there are 12 for a mixed dumbbell)
        # This is a list of defect objects (object in the object!) where, for each of them, the list __symeqdefects is empty,
        # sublattice and direction are obtained with method find_symeq(symop), and oriented and index are the same as the mother class.
        self.__symeqdefects = []
        # symindexlist is a list of lists showing the resulting defect from a given symmetry operation
        # first item in the sublist is the index of the defect resulting from the symmetry operation
        # next three items in the sublists are translations to account for translations of defect positions
        # because defects are always defined inside the base supercell (coordinates between 0 and 1)
        # For instance an inversion on +0.5,0,0 gives -0.5,0,0 which is in fact described by +0.5,0,0
        # and a translation of -1,0,0 supercells such that the sublist would be [a,-1,0,0] if this defect is indexed a
        self.__symindexlist = []
        self.__sudefect = None # symmetry unique defect from which the current defect was obtained by symmetry
        self.__align_dist = None # distance between two "aligned" defects to make sure that dumbbell configurations are correct
        self.__siteinter = None # dictionary relating species at this defect position to site interactions

    def get_crystal(self):
        return self.__crystal

    def get_sublattice(self):
        return self.__sublattice

    def get_sublattice_nod(self):
        return self.__sublattice_nod

    def get_index(self):
        return self.__index

    def get_symeq(self):
        return self.__symeqdefects

    def get_symindexlist(self):
        return self.__symindexlist

    def append_symindexlist(self, indexitem):
        self.__symindexlist.append(indexitem)

    def get_info(self):
        logger.info("  Defect {} on sublattice {} has {} symmetry equivalent.".format(self.__index, self.__sublattice, len(self.__symeqdefects)))

    def get_sudefect(self):
        return self.__sudefect

    def set_sudefect(self, a):
        self.__sudefect = a

    def set_align_dist(self): # to check distance between atmoms sharing a site if it is a dumbbell
        self.__align_dist = np.max([distance(vect1=self.__symeqdefects[a].get_sublattice()-self.__symeqdefects[a].get_sublattice_nod(),
                                             vect2=self.__symeqdefects[b].get_sublattice()-self.__symeqdefects[b].get_sublattice_nod(),
                                             crystal=self.__crystal) for a in range(len(self.__symeqdefects)) for b in range(a,len(self.__symeqdefects))])

    def get_align_dist(self):
        return self.__align_dist

    # Find list of symmetry equivalent defects, based on list of symmetry operations symop.
    def find_symeq(self, symop):
        # Apply symmetry operations to sublattice (defect position in the primitive cell) and direction (orientation)
        symeq = apply_symmetry_operations([self.__sublattice, self.__sublattice_nod], symop)
        # Apply "periodic boundary conditions"
        for eq in symeq:
            for sub in range(2): # with and without dumbbell displacement
                eq[sub] = apply_pbc(np.array(eq[sub], dtype=float))
            # Check that the new found defect was not already in the list.
            found = False
            for r in range(0, len(self.__symeqdefects)):
                refdefect = np.array(self.get_symeq()[r].get_sublattice(), dtype=float)
                if are_equal_arrays(refdefect, eq[0]):
                    found = True
                    break
            if not found:
                self.__symeqdefects.append(Defect(crystal=self.__crystal, sublattice=eq[0], sublattice_nod=eq[1], index=self.__index+len(self.__symeqdefects)))
        # Find the defect indices for each symmetry operation to speed up the identification of symmetry equivalent configurations
        for test_defect in self.__symeqdefects:
            test_defect.set_sudefect(self) # set symmetry unique defect
            for operator in symop:
                eq = apply_symmetry_operations(vector_list=[test_defect.get_sublattice()], symop_list=[operator], unique=False, applytrans=True)[0][0]
                trans = [0., 0., 0.]
                for r in range(3):
                    if eq[r] < (0 - tol_pbc):
                        trans[r] = -np.ceil(-eq[r])  # we need to get this translation correct that is why we are not using apply_pbc function
                        eq[r] += np.ceil(-eq[r])
                    elif eq[r] >= (1 - tol_pbc):
                        trans[r] = +np.floor(eq[r]+tol_pbc)  # we need to get this translation correct that is why we are not using apply_pbc function
                        eq[r] += -np.floor(eq[r]+tol_pbc)
                for r in range(0, len(self.__symeqdefects)):
                    refdefect = np.array(self.get_symeq()[r].get_sublattice(), dtype=float)
                    if are_equal_arrays(refdefect, np.array(eq, dtype=float)):
                        test_defect.append_symindexlist(flat_list([[self.get_symeq()[r].get_index()],trans]))
                        break
            if len(test_defect.get_symindexlist()) != len(symop):
                produce_error_and_quit("In Defect.find_symeq - Some defect symmetry equivalent indices of defect {} were not found.".format(self.__get_sublattice()))

    #def set_siteinter(self, species_list, species, siteinter):
    #    if self.__siteinter is None:
    #        self.__siteinter={x: 0 for x in species_list} # site inter
    #    self.__siteinter[species] = siteinter # interaction 0 is for bulk species or forbidden defects or a given species

    def get_siteinter(self): # requires a species object, not bulk species
        return self.__siteinter

    def set_siteinter(self, a):
        self.__siteinter = a

    def find_siteinter_classes(self, species_list: list, index: int, symopcpg: list, all_defect_list: list): # must be applied to a su_defect
        tmp = copy.copy(self.__symeqdefects) # list containing all symmetry equivalent defects of this Symmetry-Unique defect
        while len(tmp) > 0: # the first element of tmp is the defect that is currently being looked into
            if tmp[0].get_siteinter() is None:
                found = False
                for symeq,symop in zip(tmp[0].get_symindexlist()[:len(symopcpg)], symopcpg):
                    if symeq[0] == tmp[0].get_index() and symop.get_cpgsign() == -1: # there is a symmetry operation that reverses the CPG while preserving this defect. Site interaction is not necessary
                        found =True
                        break
                if found: # site interaction is not needed
                    for a in set([x[0] for x in tmp[0].get_symindexlist()[:len(symopcpg)]]):
                        all_defect_list[a].set_siteinter(a={x: 0 for x in species_list})  # site inter set to zero for all species because no site interaction is required
                        if all_defect_list[a] in tmp:
                            tmp.remove(all_defect_list[a]) # this defect has already been assigned a site interaction so it does not need to be studied further on
                else: # site interaction is needed
                    siteinterdict = {x: 0 for x in species_list}
                    minussiteinterdict = {x: 0 for x in species_list} # dict for symmetry equivalents reversing the CPG direction
                    for sp in species_list: # checking if this species can occupy this type of defect
                        if self in sp.get_defects(): # species sp can be in this type of defect (non zero permission)
                            index += 1 # new interaction
                            siteinterdict[sp] = index
                            minussiteinterdict[sp] = -index
                    # assigning the same siteinterdict dictionary to all symmetry equivalents
                    for symeq in set([(x[0],y.get_cpgsign()) for x,y in zip(tmp[0].get_symindexlist()[:len(symopcpg)], symopcpg)]):
                        if all_defect_list[symeq[0]].get_siteinter() is not None:
                            produce_error_and_quit("In find_siteinter_classes - defect has already been assigned a site interaction...this is odd.")
                        if symeq[1] == 1:
                            all_defect_list[symeq[0]].set_siteinter(a=siteinterdict)
                        elif symeq[1] == -1:
                            all_defect_list[symeq[0]].set_siteinter(a=minussiteinterdict)
                        else:
                            produce_error_and_quit("In find_siteinter_classes - CPG conserving symmetry operation is neither +1 or -1...this is odd")
                        if all_defect_list[symeq[0]] in tmp:
                            tmp.remove(all_defect_list[symeq[0]])
            else:
                produce_error_and_quit("In find_siteinter_classes - defect has already been assigned a site interaction...this is odd.")
        return index


class JumpConstraint:
    """A JumpConstraint defines the initial position and orientation of a given species
    for a jump mechanism to occur. Note that a jump will only be possible if all constraints
    are verified"""

    def __init__(self, inispecies, iniposition, iniposition_nod, finspecies, finposition, finposition_nod):
        # String label (for comparison purposes)
        self.__label = str(inispecies.get_index()) + "_" + ' '.join(['{:.6f}'.format(np.round(a+1e-8, 6)) for a in iniposition]) +\
                    "_" + str(finspecies.get_index()) + "_" + ' '.join(['{:.6f}'.format(np.round(a+1e-8, 6)) for a in finposition])
        # "_nod" positions are juste the same as position but the eventual dumbbell displacement is set 0 to have the exact jump vector.
        # Attributes before the jump
        self.__inispecies = inispecies  # species type
        self.__iniposition = np.array(iniposition)  # position of the defect
        self.__iniposition_nod = np.array(iniposition_nod)  # position of the defect without the dumbbell displacement from site
        self.__inidefect = None  # defect type
        self.__initrans = None  # integer supercell translation
        # Attributes after the jump
        self.__finspecies = finspecies
        self.__finposition = np.array(finposition)
        self.__finposition_nod = np.array(finposition_nod)
        self.__findefect = None
        self.__fintrans = None

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            complist = []
            complist.append(self.__inispecies == other.get_inispecies())
            complist.append(self.__finspecies == other.get_finspecies())
            complist.append(are_equal_arrays(self.__iniposition, other.get_iniposition()))
            complist.append(are_equal_arrays(self.__finposition, other.get_finposition()))
            return np.all(complist)
        else:
            return False

    def get_inispecies(self):
        return self.__inispecies

    def get_iniposition(self):
        return self.__iniposition

    def get_iniposition_nod(self):
        return self.__iniposition_nod

    def get_finspecies(self):
        return self.__finspecies

    def get_finposition(self):
        return self.__finposition

    def get_finposition_nod(self):
        return self.__finposition_nod

    def get_inidefect(self):
        return self.__inidefect

    def get_initrans(self):
        return self.__initrans

    def get_findefect(self):
        return self.__findefect

    def get_fintrans(self):
        return self.__fintrans

    def get_label(self):
        return self.__label

    def get_info(self):
        return "From species {} as defect {}  in position {} to species {} as defect {}  in position {}"\
            .format(str(self.get_inispecies()), str(self.get_inidefect()), str(self.get_iniposition()),\
                    str(self.get_finspecies()), str(self.get_findefect()), str(self.get_finposition()),)

    def vect2deftrans_constraint(self, all_defect_list):
        """Assign defect and translation attributes to the initial and final configuration of the constraint"""
        # (third item in mylist is a defect object)
        mylist = vect2deftrans(vec=copy.copy(self.__iniposition), all_defect_list=all_defect_list)
        self.__inidefect = mylist[0]
        self.__initrans = mylist[1]
        mylist = vect2deftrans(vec=copy.copy(self.__finposition), all_defect_list=all_defect_list)
        self.__findefect = mylist[0]
        self.__fintrans = mylist[1]

    def check_sublattices(self, all_defect_list: list) -> bool:
        """Check that all species in the constraint are sitting in the correct sublattices"""
        self.vect2deftrans_constraint(all_defect_list=all_defect_list)
        check_ini = check_subconfiguration_consistence(species_list=[self.__inispecies], position_list=[self.__iniposition], all_defect_list=all_defect_list)
        check_fin = check_subconfiguration_consistence(species_list=[self.__finspecies], position_list=[self.__finposition], all_defect_list=all_defect_list)
        return (check_ini and check_fin)


class JumpMech:
    """"A Jump is read from the input and contains a number of constraints"""

    def __init__(self, index, name):
        self.__index = index   # index identifying the jump mechanism
        self.__label = name  # label containing jump name and constraint labels (for comparison purposes)
        self.__nconstraints = 0   # number of constraints in the jump mechanism
        self.__name = name  # name of this jump mechanism
        self.__constraints = []  # list of constraints
        self.__netdisp = []  # list of net displacements for each species ordered by species index
        self.__netdisp_cpg = []  # list of net displacements for each species by species index projected on cpg and perpendicular directions
        self.__symeqs = [] # list of symmetric equivalents of this jump
        self.__summednetdisp = None  # average of net displacements over all symmetry equivalent
        self.__cpgsummednetdisp = None  # average over symmetry equivalents of net displacements projected on cpg and perp. directions

    def __eq__(self, other):
        #if isinstance(other, self.__class__)  and  self.get_name() == other.get_name():
        if isinstance(other, self.__class__):
            complist = [constr1 == constr2 for constr1, constr2 in zip(self.__constraints, other.get_constraints())]
            return np.all(complist)
        else:
            return False

    def get_nconstraints(self):
        return self.__nconstraints

    def get_name(self):
        return self.__name

    def get_constraints(self):
        return self.__constraints

    def get_index(self):
        return self.__index

    def get_label(self):
        return self.__label

    def get_netdisp(self):
        return self.__netdisp

    def get_netdisp_cpg(self):
        return self.__netdisp_cpg

    def get_symeqs(self):
        return self.__symeqs

    def get_summednetdisp(self):
        return self.__summednetdisp

    def get_cpgsummednetdisp(self):
        return self.__cpgsummednetdisp

    def add_constraint(self, c):  # c is a JumpConstraint object
        self.__constraints.append(c)
        self.__nconstraints = len(self.__constraints)
        self.__label += "_" + c.get_label()

    def set_symeqs(self, symops: list, jump_catalogue: list, all_defect_list: list):
        self.__symeqs = self.find_symeqs(symops=symops, old_jumps=jump_catalogue, unique=False, addreverse=False, all_defect_list=all_defect_list)

    def set_name(self, new_name: str):
        self.__name = new_name

    def set_summednetdisp(self, a):
        if self.__summednetdisp is None:
            self.__summednetdisp = a

    def set_cpgsummednetdisp(self, a):
        if self.__cpgsummednetdisp is None:
            self.__cpgsummednetdisp = a

    def set_net_species_displacement(self, strcrystal: Crystal, directions: list, spec: list):
        species_disp = [np.array([0, 0, 0], dtype=float) for _ in range(0, len(spec))] # list of displacement vectors for each species
        #comp_disp = [np.array([0, 0, 0], dtype=float) for _ in range(0, len(spec))] # list of displacement vectors for each component
        for cons in self.__constraints:
            species_disp[cons.get_inispecies().get_index()] = species_disp[cons.get_inispecies().get_index()] - np.array(cons.get_iniposition_nod(), dtype=float)
            species_disp[cons.get_finspecies().get_index()] = species_disp[cons.get_finspecies().get_index()] + np.array(cons.get_finposition_nod(), dtype=float)
        self.__netdisp = species_disp
        for species in species_disp:
            # Projection of species displacements on three specific diffusion directions
            self.__netdisp_cpg.append(np.array(proj(vect=species, crystal=strcrystal, direc=directions), dtype=object))

    def compute_summednetdisp(self, spec: int, Ndirs: int):
        self.__summednetdisp = np.zeros((Ndirs*spec*spec,), dtype=object)
        for alpha in range(spec):
            for beta in range(spec):
                for direction in range(Ndirs):
                    for sy in self.__symeqs:
                        if (sy.get_netdisp_cpg()[alpha + 1][direction]+0*e).subs(e, strainval) > 0:
                            self.__summednetdisp[direction*spec*spec+alpha*spec+beta] += sy.get_netdisp_cpg()[alpha + 1][direction]*sy.get_netdisp_cpg()[beta+1][0]
                    self.__summednetdisp[direction * spec * spec + alpha * spec + beta] = sym.simplify(self.__summednetdisp[direction*spec*spec+alpha*spec+beta]/len(self.__symeqs))
        for sy in self.__symeqs:
            sy.set_summednetdisp(self.__summednetdisp)

    def compute_cpgsummednetdisp(self, spec: int, symopcpg: list, Ndirs: int):
        self.__cpgsummednetdisp = np.zeros((spec, Ndirs), dtype=object)
        for alpha in range(spec):
            for direction in range(Ndirs):
                for idx in range(len(symopcpg)):
                    self.__cpgsummednetdisp[alpha][direction] += self.__symeqs[idx].get_netdisp_cpg()[alpha+1][direction] * symopcpg[idx].get_cpgsign()
                self.__cpgsummednetdisp[alpha][direction] = sym.simplify(self.__cpgsummednetdisp[alpha][direction]/len(symopcpg))

    def find_symeqs(self, symops: list, all_defect_list: list, old_jumps: dict = {}, unique: bool = True, addreverse: bool = True, print: bool = True) -> list:
        """Applies a set (symops) of symmetry operations to a jump mechanism to find all symmetry equivalent jumps"""
        jump_list = []  # list of jump mechanism objects
        vectors = []  # list of vectors containing initial positions for each constraint, and then final positions for each constraint
        reversejump = []  # to make sure reverse jumps are included we also try to reverse initial and final positions
        # Build vectors containing all initial and final positions for each constraint
        for jcons in self.__constraints:
            vectors.append(jcons.get_iniposition())
            reversejump.append(jcons.get_finposition())
        for jcons in self.__constraints:
            vectors.append(jcons.get_finposition())
            reversejump.append(jcons.get_iniposition())
        # Apply symmetry operations on initial and final positions for each constraint simultaneously
        tmpsymeqjumps, tmpsymopidx = apply_symmetry_operations(vector_list=vectors, symop_list=symops, unique=unique, symop_indices=True)  # symmetry equivalent jump vectors
        # Apply periodic boundary conditions to make sure the initial position of the first constraint is in the base supercell
        symeqjumps = []
        symopidx = []
        for idx, jump in enumerate(tmpsymeqjumps):
            tvec = -np.floor(jump[0] + tol_pbc * np.array([1, 1, 1]))
            jump = [np.array(i) + tvec for i in jump]  # translates each position
            # Modify dumbbell coordinates if needed (bulkspecies), first for initial configuration then for final
            if len(bulkspecies_list) > 0:
                jump[:self.__nconstraints] = apply_bulkspecies_positions(positions=jump[:self.__nconstraints], all_defect_list=all_defect_list, species=[c.get_inispecies() for c in self.__constraints])
                jump[self.__nconstraints:] = apply_bulkspecies_positions(positions=jump[self.__nconstraints:], all_defect_list=all_defect_list, species=[c.get_inispecies() for c in self.__constraints])
            # Remove jump duplicates
            found = False
            for comp in symeqjumps:
                if are_equal_arrays(np.array(jump, dtype=float), np.array(comp, dtype=float)):
                    found = True
                    break
            if not unique:
                found = False
            if not found:
                symeqjumps.append(jump)
                symopidx.append(tmpsymopidx[idx])
        del tmpsymeqjumps, tmpsymopidx

        # Add reverse jump if it is not already part of the symmetry equivalents (only for symmetry-unique list)
        # Check if reverse jump is already part of symetry equivalent jumps
        isreverseincluded = True
        revprt = ""
        if addreverse:
            # Checking species; if there are transmutations the reverse jump will not be part of symmetrical equivalents
            for jcons in self.__constraints:
                if jcons.get_inispecies() != jcons.get_finspecies():
                    isreverseincluded = False
                    break
            if isreverseincluded:
                isreverseincluded = False
                # There are no transmutations so reverse jump might be among symmetry equivalents, so let's check positions
                # Apply translation to make sure the atom of the first constraint is initially in the first supercell
                tvec = -np.floor(reversejump[0]+ tol_pbc*np.array([1, 1, 1]))
                for i in range(0, len(reversejump)):
                    reversejump[i] = reversejump[i] + tvec # translates each position
                # Modify dumbbell coordinates if needed (bulkspecies), first for initial configuration then for final
                if len(bulkspecies_list) > 0:
                    reversejump[:self.__nconstraints] = apply_bulkspecies_positions(positions=reversejump[:self.__nconstraints], all_defect_list=all_defect_list, species=[c.get_finspecies() for c in self.__constraints])
                    reversejump[self.__nconstraints:] = apply_bulkspecies_positions(positions=reversejump[self.__nconstraints:], all_defect_list=all_defect_list, species=[c.get_inispecies() for c in self.__constraints])
                for symeq in symeqjumps: # compare reverse jump with each symmetry equivalent jump
                    if are_equal_arrays(np.array(flat_list(symeq), dtype=float), np.array(flat_list(reversejump), dtype=float)):
                        isreverseincluded = True
                        break
            if not isreverseincluded:  # reverse jump was not found among symmetry equivalent and must be added
                revprt = "(!! Reverse jump added !!)"
                tmpsymeqjumps, tmpsymopidx = apply_symmetry_operations(vector_list=reversejump, symop_indices=True, symop_list=symops, unique=unique)
                for idx, jump in enumerate(tmpsymeqjumps):
                    tvec = -np.floor(jump[0] + tol_pbc * np.array([1, 1, 1]))
                    jump = [np.array(i) + tvec for i in jump]  # translates each position
                    # Modify dumbbell coordinates if needed (bulkspecies), first for initial configuration then for final
                    if len(bulkspecies_list) > 0:
                        jump[:self.__nconstraints] = apply_bulkspecies_positions(positions=jump[:self.__nconstraints], all_defect_list=all_defect_list, species=[c.get_finspecies() for c in self.__constraints])
                        jump[self.__nconstraints:] = apply_bulkspecies_positions(positions=jump[self.__nconstraints:], all_defect_list=all_defect_list, species=[c.get_inispecies() for c in self.__constraints])
                    # Remove jump duplicates
                    found = False
                    for comp in symeqjumps:
                        if are_equal_arrays(np.array(jump, dtype=float), np.array(comp, dtype=float)):
                            found = True
                            break
                    if not unique:
                        found = False
                    if not found:
                        symeqjumps.append(jump)
                        symopidx.append(tmpsymopidx[idx])
        if unique and print:
            logger.info('    Found {} symmetry equivalent jump mechanisms for {} {}'. format(len(symeqjumps), self.__name, revprt))
        # Create new jumps for all symmetry equivalents (including reverse jumps)
        for isym in range(0, len(symeqjumps)):
            symeq = symeqjumps[isym]
            species = []
            if (not isreverseincluded) and (isym >= 0.5*len(symeqjumps)):
                # Final species become initial species when jump is reversed
                species.append([e.get_finspecies() for e in self.get_constraints()])
                species.append([e.get_inispecies() for e in self.get_constraints()])
            else:
                species.append([e.get_inispecies() for e in self.get_constraints()])
                species.append([e.get_finspecies() for e in self.get_constraints()])
            new = JumpMech(self.__index, self.__name)
            # Add constraints to this JumpMech object
            for cons in range(0, self.__nconstraints):
                if (not isreverseincluded) and (isym >= 0.5 * len(symeqjumps)):
                    inipos_nod = apply_symmetry_operations(vector_list=[np.array(self.__constraints[cons].get_finposition_nod())], symop_list=[symops[symopidx[isym]]])[0][0]
                    finpos_nod = apply_symmetry_operations(vector_list=[np.array(self.__constraints[cons].get_iniposition_nod())], symop_list=[symops[symopidx[isym]]])[0][0]
                else:
                    inipos_nod = apply_symmetry_operations(vector_list=[np.array(self.__constraints[cons].get_iniposition_nod())], symop_list=[symops[symopidx[isym]]])[0][0]
                    finpos_nod = apply_symmetry_operations(vector_list=[np.array(self.__constraints[cons].get_finposition_nod())], symop_list=[symops[symopidx[isym]]])[0][0]
                if cons == 0:  # actualize the translation vector with the initial position of the first constraint
                    tvec = -np.floor(np.array(inipos_nod) + tol_pbc * np.array([1, 1, 1]))
                inipos_nod += tvec
                finpos_nod += tvec
                new.add_constraint(JumpConstraint(inispecies=species[0][cons], iniposition=symeq[cons], iniposition_nod=inipos_nod, finspecies=species[1][cons], finposition=symeq[cons+self.__nconstraints], finposition_nod=finpos_nod))
            # Retrieve the new jump from the already existing catalogue, or create a new one
            if old_jumps.get(new.get_label()) is None:
                jump_list.append(new)
            else:
                jump_list.append(old_jumps[new.get_label()])
        return jump_list

    def check_jump(self, all_defect_list: list, component_list: list, crystal: Crystal):
        """Check that the jump constraints are fully consistent with each other, in terms of proper sublattices and no overlapping components"""
        ini_species = [c.get_inispecies() for c in self.get_constraints()]
        fin_species = [c.get_finspecies() for c in self.get_constraints() if c.get_finspecies().get_name() != 'bulk']
        ini_positions = [c.get_iniposition() for c in self.get_constraints()]
        fin_positions = [c.get_finposition() for c in self.get_constraints() if c.get_finspecies().get_name() != 'bulk']
        check_ini = check_subconfiguration_consistence(species_list=ini_species, position_list=ini_positions, all_defect_list=all_defect_list, component_list=component_list, crystal=crystal)
        check_fin = check_subconfiguration_consistence(species_list=fin_species, position_list=fin_positions, all_defect_list=all_defect_list, component_list=component_list, crystal=crystal)
        return (check_ini and check_fin)


class JumpFrequency:

    def __init__(self, number: int, config_ini: Configuration, config_fin: Configuration, jump: JumpMech, symop_list: list, dissofreq: bool, beyond: bool, name_perms: list, all_defect_list: list):
        self.__number = number-1
        self.__config_ini = config_ini
        self.__config_fin = config_fin
        self.__jump = jump
        self.__disso = dissofreq
        self.__beyond = beyond
        self.__symeq = {}
        self.__label = None
        self.__dissotype = None
        eqconfigs = apply_jump_symmetry_operations(configlist=[config_ini, config_fin], symops=symop_list, name_perms=name_perms, all_defect_list=all_defect_list)
        for i in eqconfigs:
            self.__symeq[i]=0

    def get_number(self):
        return self.__number

    def get_config_ini(self):
        return self.__config_ini

    def get_config_fin(self):
        return self.__config_fin

    def get_jump(self):
        return self.__jump

    def get_symeq(self):
        return self.__symeq

    def get_disso(self):
        return self.__disso

    def get_beyond(self):
        return self.__beyond

    def set_dissotype(self, a):
        self.__dissotype = a

    def get_dissotype(self):
        return self.__dissotype

    def set_label(self, a):
        self.__label = a

    def get_label(self):
        return self.__label

    def set_number(self, a):
        self.__number = a-1

    def cleanfreq(self):
        self.__jump = None
        self.__symeq = None


class Species:
    """Defect species are provided by the user in the input file and are assigned a short and long name"""

    def __init__(self, index, name, defects, permissions):
        self.__index = index  # species index (goes from 0 to n, there are n user-input species (from 0 to n-1), plus the bulk (n))
        self.__name = name  # species written name (user-input)
        self.__bulk = False  # flag marking if species resembles a bulk atom (e.g. a pure dumbbell)
        self.__defects = defects  # list of defects where the species can sit
        # For each allowed defect, self.__defect_permission is a list that gives 1 if the species can be in this defect by itself,
        # or -1 if it needs to have another defect in the same site (useful to treat mixed dumbbells) (default is 1)
        self.__permissions = permissions

    def set_bulk(self):
        self.__bulk = True

    def get_bulk(self):
        return self.__bulk

    def get_index(self):
        return self.__index

    def get_name(self):
        return self.__name

    def get_defects(self):
        defecttuple = self.__defects
        if not(type(defecttuple) is tuple):
            defecttuple = tuple(defecttuple)
        return defecttuple

    def get_permissions(self):
        permissions = self.__permissions
        if not(type(permissions) is tuple):
            permissions = tuple(permissions)
        return permissions

    def get_info(self):
        logger.info("  Species {} is a {} and can be in sublattices {}".format(self.__index, self.__name, [i.get_index() for i in self.__defects]))


class Subconfiguration:
    """A Subconfiguration contains species, defect types and position for each component. It is used to apply some interaction model in the numerical part of the code"""

    def __init__(self, defects: list, translations: list, species, Ebind, Sbind, complist):
        self.__species = species # List of species objects
        self.__defects = defects  # List of defects (a defect for each component)
        self.__translations = np.array(translations)  # Indices of integer supercell translations (one for each component)
        self.__Ebind = Ebind # binding energy associated with this sub configuration
        self.__Sbind = Sbind # binding entropy associated with this sub configuration
        self.__indices = []  # each element corresponds to a non-bulk sub-configuration species, and contains a list of matching species indexes in the cluster
        self.__forbidlist = []  # list of sites that should be free (bulk sites in the description of the sub-configuration)
        self.__nbtranslations = []  # non bulk translations of the subconfiguration
        self.__nbspecies = [] # non bulk species
        self.__nbdefects = [] # non bulk defects
        self.__speciesok = True
        for idx, spsub in enumerate(self.__species):  # loop on species of the sub configuration
            if spsub.get_name() == 'bulk':
                self.__forbidlist.append(self.__translations[idx] + self.__defects[idx].get_sublattice())
            else:
                self.__nbtranslations.append(self.__translations[idx])
                self.__nbdefects.append(self.__defects[idx])
                self.__nbspecies.append(self.__species[idx])
                self.__indices.append([i for i, x in enumerate(complist) if x.get_species() == spsub])
                if len(self.__indices[-1]) == 0:
                    self.__speciesok = False
        self.__indices = [p for p in product(*self.__indices) if len(p) == len(set(p))]  # all possible correspondences between configuration and subconfiguration species

    def get_species(self):
        return self.__species

    def get_defects(self):
        return self.__defects

    def get_nbdefects(self):
        return self.__nbdefects

    def get_nbtranslations(self):
        return self.__nbtranslations

    def get_translations(self):
        return self.__translations

    def get_Ebind(self):
        return self.__Ebind

    def get_Sbind(self):
        return self.__Ebind

    def get_defect_position(self, def_idx: int) -> np.ndarray:
        """Return defect position as sublattice+translation (defect order is that defined in list self.__defects)"""
        return np.array(self.get_defects()[def_idx].get_sublattice(), dtype=float) + self.get_translations()[def_idx]

    def isincluded(self, conf: Configuration):
        """Determines whether this sub-configuration is included in Configuration conf"""
        if self.__speciesok:
            for perm in self.__indices:
                if np.all(self.__nbdefects == [conf.get_defects()[idx] for idx in perm]): # defects are similar?
                    t = np.array(conf.get_translations()[3*perm[0]:(3*perm[0]+3)], dtype=float) - self.__nbtranslations[0]
                    conftrans = np.array([(np.array(conf.get_translations()[3*idx:(3*idx+3)], dtype=float)-t) for idx in perm])
                    if are_equal_arrays(A=conftrans, B=self.__nbtranslations): # this subconfiguration belongs to configuration conf
                        for bulk in self.__forbidlist: # checking that no one in conf is on a bulk subconf site
                            for idx in range(0, len(conf.get_defects())):
                                if are_equal_arrays(A=bulk, B=(conf.get_defect_position(def_idx=idx)-t)):
                                    return [0.00, 1.00, False]
                        return [self.__Ebind, self.__Sbind, True]
        return [0.00, 1.00, False] # if the code reaches this line, sub configuration was not found in conf


class SymOp:
    """A Symop object is a 3x3 matrix (and a translation vector) representing a symmetry operation that is valid for the supercell"""

    def __init__(self, rotation):
        self.__rotation = rotation  # rotation matrix
        self.__translation = None  # 1-D translation vector
        self.__cpgsign = None  # +/-1 depending  on the transformation of CPG with this symetry operation

    def get_rotation(self):
        return self.__rotation

    def get_translation(self):
        return self.__translation

    def get_cpgsign(self):
        return self.__cpgsign

    def set_rotation(self, new_rot):
        self.__rotation = new_rot

    def set_translation(self, new_trans):
        self.__translation = new_trans

    def set_cpgsign(self, cpgsign):
        self.__cpgsign = cpgsign

    def apply_symop(self, vect):
        return np.dot(self.__rotation, vect) + self.__translation


# Definition of functions

def analyze_bulkspecies(component_list: list, all_defect_list: list) -> None:
    """Analyze the bulkspecies defined by the user, i.e., symmetric species that need to be specially treated to avoid
    double counting of perfectly symmetric configurations, such as those involving a pure dumbbell.
    The function creates a flag, a list of bulkspecies and a dictionary that are stored as global variables in kinepy.
    bulkspecies_list contains the indices of the components that have been defined as bulkspecies.
    bulkspecies_symdefects is a dictionary where the keys are all defects in all_defect_list, giving as output the
    corresponding opposite defect (=multiplied by -1) if the first non-null element of the defect sublattice is negative,
    otherwise it gives the same defect. This is used in set_conf to swap the defect with its symmetric opposite if that
    component occupies a site by itself."""
    # Update list of special "bulkspecies" (list of indices corresponding to the bulkspecies components)
    for cidx, comp in enumerate(component_list):
        if comp.get_species().get_bulk():
            bulkspecies_list[comp.get_species()] = 'b' # the "b" tag means nothing, it is just to make sure the key refers to a non-empty reference
    # Build dictionary of symmetric defects
    # For each defect in all_defect_list, we need to look for the exact opposite sublattice
    all_sublattices = [d.get_sublattice()-d.get_sublattice_nod() for d in all_defect_list]  # list of all defect sublattices
    for defect in all_defect_list:
        # Look for first non-zero element of sublattice
        k = 0
        stop = False
        while k < 3  and  not stop:
            if np.abs(defect.get_sublattice()[k]-defect.get_sublattice_nod()[k]) < tol:
                k += 1
            else:
                stop = True
        # (if k > 2, it's the [0, 0, 0] sublattice -> no need to change sign)
        if k > 2  or  defect.get_sublattice()[k]-defect.get_sublattice_nod()[k] > 0:  # if first element is positive, or it is the [0, 0, 0] sublattice
            bulkspecies_symdefects[defect] = defect  # store defect as itself
        else:  # if first element is negative, switch sign of the sublattice and look for the corresponding defect
            bulkspecies_symdefects[defect] = None
            opposite_sublattice = - (defect.get_sublattice()-defect.get_sublattice_nod()) + defect.get_sublattice_nod()
            for s, sub in enumerate(all_defect_list):
                if are_equal_arrays(A=opposite_sublattice,B=sub.get_sublattice()):
                    bulkspecies_symdefects[defect] = all_defect_list[s]
        if bulkspecies_symdefects[defect] is None:
            produce_error_and_quit("In analyze_bulkspecies - Opposite defect {} not found.".format(defect.get_sublattice()))


def apply_bulkspecies(defects: list, translations: list, species: list) -> list:
    defect_list = list(defects)
    # Treat bulkspecies so that the first position component of dumbbells is positive
    positions = [df.get_sublattice() + t for df, t in zip(defect_list, translations)]
    for d, sp in enumerate(species):
        if bulkspecies_list.get(sp) is not None: # this species should be applied the bulkspecies feature
            # (crystal is taken from the attribute of the first defect in the list)
            distances = [distance(vect1=positions[d], vect2=vect2, crystal=defect_list[0].get_crystal()) for vidx, vect2 in enumerate(positions) if vidx != d if species[vidx].get_name()!='bulk']
            if np.all(distances > maxdbdist * db_displ): # this dumbbel is isolated from the others
                defect_list[d] = bulkspecies_symdefects[defect_list[d]]
    return defect_list

def apply_bulkspecies_positions(positions: list, all_defect_list: list, species: list) -> list:
    # We have positions for each component so must split them into defects and translations
    defects = []
    translations = []
    for c in positions:
        [d, t] = vect2deftrans(vec=np.array(c), all_defect_list=all_defect_list)
        defects.append(d)
        translations.append(t)
    defects = apply_bulkspecies(defects=defects, translations=translations, species=species)
    for d, t in enumerate(translations):
        positions[d] = t + defects[d].get_sublattice()
    return positions

def apply_configuration_symmetry_operations(config: Configuration, symops: list, ncpgops: int, name_perms: list, all_defect_list: list, create: bool = False) -> list:
    """Apply a set (SymOp object list) of symmetry operations to a configuration object, giving as output the list of symmetry equivalent configurations.
    create is false by default, which means that the function will only output configuration name.
    If create is set to true, the function actually creates all symmetry equivalent configurations and a third element
    is added to the output list, which is the list of configuration objects created.
    For instance, if vector_list has 2 components and there are 48 symmetries, the output is a list
    of 48 elements, where each element is a list of 2 vectors.
    The number of symmetries is reduced if different symmetry operations yield the same result."""
    # Check if configuration is dissociated
    if config.get_label()[0] == 'd':
        disso = True  # if config is dissociated, its symmetry equivalent are also dissociated
    else:
        disso = False
    # Retrieve translation vectors from base supercell
    translist = []  # list of translation vectors (one per defect) from base supercell
    for i in range(0, int(len(config.get_translations())/3)):
        translist.append(config.get_translations()[3 * i:3 * i + 3])
    # Apply symmetry operations to translation vectors
    eqtrans = apply_symmetry_operations(vector_list=translist, symop_list=symops, unique=False, applytrans=False)
    eqdefects = []  # defect indices
    # Retrieve defect symmetry equivalents stored in defect object
    for i in range(0, len(symops)):  # loop over symmetry operations
        eqdefects.append([])
        defecttrans = []
        for e in config.get_defects():  # loop over defects in configuration
            eqdefects[-1] += [e.get_symindexlist()[i][0]]  # resulting defect index from symmetry operation (=index in all_defect_list)
            defecttrans += [e.get_symindexlist()[i][1:4]]  # resulting translations from symmetry operation
        # Translate back to centered reference frame, in case the reference component has moved upon symmetry operation (and transform translations to arrays)
        for j in range(len(defecttrans)-1,-1,-1):
            defecttrans[j] = np.asarray(defecttrans[j]) - np.asarray(defecttrans[0])
        eqtrans[i] = [np.asarray(a) + b for a, b in zip(eqtrans[i], defecttrans)]
    # Initialize output lists
    symeqconf = []  # configurations that are only thermodynamically equivalent
    cpg_symeqconf = []  # configurations that are thermodynamically and kinetically equivalent
    sign_cpg = []  # if CPG is inverted under specific symmetry operation (sign of the kinetic interaction)
    created_confs = [] # list of configuration objects - remains empty by default unless create is set to true in the function call
    uniquesymops = [] # list of symmetry operations indices that will generate all symmetry unique thermodynamic interactions
    # For each symmetry operation, figure out new configuration label and sort configuration into the right type of interaction
    for i in range(0, len(symops)):
        # Find new configuration label, defects and translations
        conf, new_eqdefects, new_eqtrans = set_confname(defects=[all_defect_list[e] for e in eqdefects[i]], translations=eqtrans[i], perms=name_perms, disso=disso, species=config.get_species())
        found = False
        # Treat first the configurations obtained with operations conserving the cpg (they are sorted on top in the symop list)
        if i < ncpgops:
            # Check if this configuration had been already found (should be in cpg_symeqconf list)
            for j in range(len(cpg_symeqconf)):
                if cpg_symeqconf[j] == conf:  # if it was found before
                    found = True
                    # Set interaction sign to 0 if symop sign is opposite to the sign of the already found interaction
                    if symops[i].get_cpgsign() == -sign_cpg[j]:
                        sign_cpg[j] = 0  # because a kinetic interaction is identical irrespective of the cpg sign (+ or -)
                    break
            if not found:  # if not, this is a new one to be appended to the output lists
                cpg_symeqconf.append(conf)
                sign_cpg.append(symops[i].get_cpgsign())
                uniquesymops.append(i)
                # Create configuration object and append it to created_confs list, if create=True
                if create:
                    created_confs.append(Configuration(defects=copy.copy(new_eqdefects), translations=copy.copy(new_eqtrans), label=conf))
                    created_confs[-1].set_species(config.get_species())
        # Now treat configurations that are obtained with operations not conserving the cpg (bottom of symop list)
        # (same structure as above)
        else:
            for j in range(len(symeqconf)):
                if symeqconf[j] == conf:
                    found = True
                    break
            for j in range(len(cpg_symeqconf)):
                if cpg_symeqconf[j] == conf:
                    found = True
                    break
            if not found:
                symeqconf.append(conf)
                uniquesymops.append(i)
                if create:
                    created_confs.append(Configuration(defects=copy.copy(new_eqdefects), translations=copy.copy(new_eqtrans), label=conf))
                    created_confs[-1].set_species(config.get_species())
    return [cpg_symeqconf, symeqconf, sign_cpg, created_confs, uniquesymops]


def apply_jump_symmetry_operations(configlist: list, symops: list, name_perms: list, all_defect_list: list) -> list:
    """Apply a set (SymOp object list) of symmetry operations to simultaneously to a list (configlist) of configuration objects, giving as output the list of symmetry equivalent configurations."""
    translist = []  # translation vectors from base supercell
    for config in configlist:
        for i in range(0, int(len(config.get_translations()) / 3)):
            translist.append(config.get_translations()[3 * i:3 * i + 3])
    eqtrans = apply_symmetry_operations(vector_list=translist, symop_list=symops, unique=False, applytrans=False)
    eqdefects = []  # defect indexes
    for i in range(0, len(symops)):
        defecttrans = []  # translations resulting from symmetry operations on defects
        eqdefects.append([])
        for config in configlist:
            defecttrans_tmp = []
            for e in config.get_defects():
                eqdefects[-1] += [e.get_symindexlist()[i][0]]
                defecttrans_tmp += [e.get_symindexlist()[i][1:4]]
            for j in range(len(defecttrans_tmp)-1, -1, -1):
                defecttrans_tmp[j] = list(np.asarray(defecttrans_tmp[j]) - np.asarray(defecttrans_tmp[0]))   # Takes into account the fact that our reference component (0) may have moved upon symmetry operation
            defecttrans += defecttrans_tmp
        eqtrans[i] = [np.asarray(a) + b for a, b in zip(eqtrans[i], defecttrans)]
    symeqconf = []  # jumps that are only thermodynamically equivalent
    nconf = len(configlist)
    ncomp =round(len(eqdefects[0]) / nconf)
    for i in range(0, len(symops)):
        conf = []
        for k in range(0, nconf):
            if configlist[k].get_label()[0] == 'd':
                name, _, _ = set_confname(defects=[all_defect_list[e] for e in eqdefects[i][ncomp*k:ncomp*(k+1)]], translations=eqtrans[i][ncomp*k:ncomp*(k+1)], perms=name_perms, disso=True, species=configlist[k].get_species())
            else:
                name, _, _ = set_confname(defects=[all_defect_list[e] for e in eqdefects[i][ncomp*k:ncomp*(k+1)]], translations=eqtrans[i][ncomp*k:ncomp*(k+1)], perms=name_perms, species=configlist[k].get_species())
            conf.append(name)
        conf.sort()
        confname=''.join(conf)
        found = False
        for j in range(0, len(symeqconf)):
            if symeqconf[j] == conf:
                found = True
                break
        if not found:
            symeqconf.append(confname)
    return symeqconf


def apply_pbc(vec: np.ndarray) -> np.ndarray:
    # applies periodic boundary conditions to numpy.array vec
    for r in range(0, 3):
        if vec[r] < (0 - tol_pbc):
            vec[r] += np.ceil(-vec[r])
        elif vec[r] >= (1 - tol_pbc):
            vec[r] += -np.floor(vec[r]+tol_pbc)
    return vec


def apply_symmetry_operations(vector_list: list, symop_list: list, unique: bool = True, applytrans: bool = True, symop_indices: bool = False) -> list:
    """Apply a set (SymOp object list) of symmetry operations to a list of vectors, giving as output, for each symmetry operation,
    the list of equivalent vectors. For instance, if vector_list has 2 components and there are 48 symmetries, the output is a list
    of 48 elements, where each element is a list of 2 vectors. If unique=True, the number of symmetries is reduced if different symmetry operations
    yield the same result."""
    n_vec = len(vector_list)
    # Transform vector list in matrix (so to perform each sym operation once on the whole set of vectors)
    vector_matrix = np.column_stack(tuple(vector_list[i] for i in range(0,n_vec)))
    # Loop over list of symmetry operations
    symm_vec_matrix_list = []
    symm_vec_list_list = []
    symop_idx_list = []
    for idx, symop in enumerate(symop_list):
        # Apply symmetry operation
        if applytrans:
            # Transform translation vector into a 3xn matrix where the translation vector is repeated n times
            new_vector_matrix = np.dot(symop.get_rotation(), vector_matrix) + np.column_stack(tuple(symop.get_translation() for _ in range(0, n_vec)))
        else:
            new_vector_matrix = np.dot(symop.get_rotation(), vector_matrix)
        # Check whether this symmetry equivalent has already been found
        if unique:
            found = False
            for previous_found_matrix in symm_vec_matrix_list:
                if are_equal_arrays(previous_found_matrix, new_vector_matrix):
                    found = True
                    break
            if not found:
                symm_vec_matrix_list.append(new_vector_matrix)  # save new matrix in the database, for comparison in the next iterations
                symm_vec_list_list.append(new_vector_matrix.transpose().tolist())  # save the obtained list of vectors in the output
                symop_idx_list.append(idx)
        else:
            symm_vec_list_list.append(new_vector_matrix.transpose().tolist())  # save the obtained list of vectors in the output
            symop_idx_list.append(idx)
    if symop_indices:
        return symm_vec_list_list, symop_idx_list
    else:
        return symm_vec_list_list


def are_equal_arrays(A: np.ndarray, B: np.ndarray ) -> bool:
    """Check if two arrays (of any shape) have all equal elements (including tolerance)"""
    subtraction_array = np.fabs ( B - A )
    if np.all ( subtraction_array < tol ):
        return True
    else:
        return False


def are_equals(a: float, b: float) -> bool:
    """Check if two numbers are equal based on tolerance defined in kinepy"""
    return np.fabs(a-b) < tol


def batchcreatefunc(ipt:str, wdir:str):
    """Automatically creates a list of files in a batch folder to perform batch calculations"""
    if not os.path.isdir(wdir + 'batch/'): # checking if the batch folder exists, else create it
        os.makedirs(wdir + 'batch/')
    mconfig = [] # list of configurations to modify
    mjumps = [] # list of jump frequencies to modify
    npts = int(float(ipt[1]))  # half number of points
    dE = float(ipt[0]) / npts  # half delta energy in eV
    for i in ipt[2:]: # loop on configurations and jump frequencies to modify
        if i[0].lower()=='c': # it is a configuration
            if '-' in i:
                for j in range(int(i[1:].split(sep='-')[0]), int(i[1:].split(sep='-')[1])+1):
                    mconfig.append(j)
            elif '*' in i:
                mconfig.append([int(e) for e in i[1:].split(sep='*')])
            else:
                mconfig.append(int(i[1:]))
        elif i[0].lower()=='j': # it is a jump
            if '-' in i:
                for j in range(int(i[1:].split(sep='-')[0]), int(i[1:].split(sep='-')[1])+1):
                    mjumps.append(j)
            elif '*' in i:
                mjumps.append([int(e) for e in i[1:].split(sep='*')])
            else:
                mjumps.append(int(i[1:]))

    with open(wdir + 'batch/batch_table.txt', 'w') as btable:
        btable.writelines('# Summary of changes in files for batch calculations: file, configuration/jump frequency index, new value\n')

    # reference configuration file
    refs = [e for e in open(wdir+'configurations.txt','r').read().split(sep='\n') if e != '']
    # numerical values
    iptnum = [e for e in open(wdir + 'num_conf.txt', 'r').read().split(sep='\n')[1:] if e != '']

    k = -1  # files counter
    for lid in mconfig:
        if type(lid) is list:
            rev = (sorted(lid) != lid)
            lid = sorted(lid)
            for e in range(-npts, npts + 1):
                for f in range(-npts, npts + 1):
                    k += 1
                    copyfile(wdir + 'jump_frequencies.txt', wdir + 'batch/jump_frequencies_' + str(k) + '.txt')
                    with open(wdir + 'batch/configurations_' + str(k) + '.txt', 'w') as out:
                        out.writelines('\n'.join(refs[:lid[0] + 1]))
                        wline = iptnum[lid[0]].split()
                        if rev:
                            wline[2] = '{:.4f}'.format(float(wline[2]) + f * dE)
                            tosum = ['configurations_' + str(k), '*'.join([str(p) for p in reversed(lid)]), '0', wline[2]]
                        else:
                            wline[2] = '{:.4f}'.format(float(wline[2]) + e * dE)
                            tosum = ['configurations_' + str(k), '*'.join([str(p) for p in lid]), wline[2], '0']
                        out.writelines('\n' + ' '.join(wline) + '\n')
                        out.writelines('\n'.join(refs[lid[0]+2:lid[1]+1]))
                        wline = iptnum[lid[1]].split()
                        if rev:
                            wline[2] = '{:.4f}'.format(float(wline[2]) + e * dE)
                            tosum[2] = wline[2]
                        else:
                            wline[2] = '{:.4f}'.format(float(wline[2]) + f * dE)
                            tosum[3] = wline[2]
                        out.writelines('\n' + ' '.join(wline) + '\n')
                        out.writelines('\n'.join(refs[lid[1] + 2:]))
                    with open(wdir + 'batch/batch_table.txt', 'a') as btable:
                        btable.writelines(' '.join(tosum)+'\n')
                    del tosum
        else:
            for e in range(-npts, npts+1):
                k += 1
                copyfile(wdir+'jump_frequencies.txt', wdir+'batch/jump_frequencies_' + str(k) + '.txt')
                with open(wdir+'batch/configurations_' + str(k) + '.txt','w') as out:
                    out.writelines('\n'.join(refs[:lid+1]))
                    wline = iptnum[lid].split()
                    wline[2] = '{:.4f}'.format(float(wline[2])+e*dE)
                    out.writelines('\n'+' '.join(wline)+'\n')
                    out.writelines('\n'.join(refs[lid+2:]))
                with open(wdir + 'batch/batch_table.txt', 'a') as btable:
                    btable.writelines(' '.join(['configurations_' + str(k), str(lid), wline[2],'\n']))

    # reference configuration file
    refs = [e for e in open(wdir+'jump_frequencies.txt','r').read().split(sep='\n') if e != '']
    # numerical values
    iptnum = [e for e in open(wdir + 'num_freq.txt', 'r').read().split(sep='\n')[1:] if e != '']
    for lid in mjumps:
        if type(lid) is list:
            rev = (sorted(lid) != lid)
            lid = sorted(lid)
            for e in range(-npts, npts + 1):
                for f in range(-npts, npts + 1):
                    k += 1
                    copyfile(wdir + 'configurations.txt', wdir + 'batch/configurations_' + str(k) + '.txt')
                    with open(wdir + 'batch/jump_frequencies_' + str(k) + '.txt', 'w') as out:
                        out.writelines('\n'.join(refs[:lid[0] + 1]))
                        wline = iptnum[lid[0]].split()
                        if rev:
                            wline[2] = '{:.4f}'.format(float(wline[2]) + f * dE)
                            tosum = ['jump_frequencies_' + str(k), '*'.join([str(p) for p in reversed(lid)]), '0', wline[2]]
                        else:
                            wline[2] = '{:.4f}'.format(float(wline[2]) + e * dE)
                            tosum = ['jump_frequencies_' + str(k), '*'.join([str(p) for p in lid]), wline[2], '0']
                        out.writelines('\n' + ' '.join(wline) + '\n')
                        out.writelines('\n'.join(refs[lid[0]+2:lid[1]+1]))
                        wline = iptnum[lid[1]].split()
                        if rev:
                            wline[2] = '{:.4f}'.format(float(wline[2]) + e * dE)
                            tosum[2] = wline[2]
                        else:
                            wline[2] = '{:.4f}'.format(float(wline[2]) + f * dE)
                            tosum[3] = wline[2]
                        out.writelines('\n' + ' '.join(wline) + '\n')
                        out.writelines('\n'.join(refs[lid[1] + 2:]))
                        with open(wdir + 'batch/batch_table.txt', 'a') as btable:
                            btable.writelines(' '.join(tosum) + '\n')
                        del tosum
        else:
            for e in range(-npts, npts + 1):
                k += 1
                copyfile(wdir + 'configurations.txt', wdir + 'batch/configurations_' + str(k) + '.txt')
                with open(wdir + 'batch/jump_frequencies_' + str(k) + '.txt','w') as out:
                    out.writelines('\n'.join(refs[:lid + 1]))
                    wline = iptnum[lid].split()
                    wline[2] = '{:.4f}'.format(float(wline[2]) + e * dE)
                    out.writelines('\n'+ ' '.join(wline) + '\n')
                    out.writelines('\n'.join(refs[lid + 2:]))
                    with open(wdir + 'batch/batch_table.txt', 'a') as btable:
                        btable.writelines(' '.join(['jump_frequencies_' + str(k), str(lid), wline[2], '\n']))

    logger.info("Creating files for {} calculations in batch folder".format(k+1))


def change_base(arr: np.ndarray, crystal: Crystal, inv: bool = False, matrix: bool = False) -> np.ndarray:
    """Change vector or matrix from orthonormal base to supercell base or viceversa"""
    # Select correct crystal matrix depending on "inv" variable (if True, pass from crystal to orthonormal base with invvectors)
    if inv:
        M = crystal.get_invvectors()
        Minv = crystal.get_primvectors()
    else:
        M = crystal.get_primvectors()
        Minv = crystal.get_invvectors()
    # Check if passed array is a vector or a matrix
    if matrix:
        return np.dot(Minv, np.dot(arr, M))
    else:
        return np.dot(Minv, arr)


def check_connectivity(configuration: Configuration, kira: float, crystal: Crystal) -> bool:
    """Check if components in a configuration form a connected cluster, i.e., there exists a linking path whose branches are shorter than the kinetic range"""
    n_comp = len(configuration.get_defects())  # Number of components in the configuration
    # The components are numbered from 0 to n_comp-1.
    # Two lists are defined: non_visited_list and node_list
    # The exploration starts from the first atom in the component list (which is then always part of the cluster)
    # At each step, node_list contains the last visited connected nodes, from each the distance to the remaining nodes is to be evaluated
    connected_list = [0]
    node_list = [0]
    distancelist = [0]
    # When a component is found to be connected to the first atom, it is removed from non_visited_list
    non_visited_list = [i for i in range(1,n_comp)]
    # Starting loop until connectivity has been verified
    isConnected = None
    while isConnected == None:
        non_visited_connected = [False for _ in range(len(non_visited_list))]
        for x in node_list:  # For each of the last visited nodes
            position_x = configuration.get_defect_position(x)
            for idx_y, y in enumerate(non_visited_list):  # For each of the non visited nodes
                # If distance xy is smaller than the kinetic range, add y to the list of new found connected nodes
                distancelist.append(distance(vect1=position_x, vect2=configuration.get_defect_position(y), crystal=crystal))
                if distancelist[-1] <= (kira * (1+tol)):
                    non_visited_connected[idx_y] = True
                else:
                    del distancelist[-1]
        # Now update the list of last visited nodes (marked as True in non_visited_connected), and the list of nodes that are still outside the cluster (False in non_visited_connected)
        node_list = [i for (i, j) in zip(non_visited_list, non_visited_connected) if j]
        connected_list += node_list
        non_visited_list = [i for (i, j) in zip(non_visited_list, non_visited_connected) if not j]
        # The exploration ends when: a) the non_visited_list is empty (connected cluster), or b) at a given iteration, the node_list is empty (non connected cluster)
        if non_visited_list == []:
            isConnected = True
        elif node_list == []:
            isConnected = False
    return [isConnected, connected_list, non_visited_list, int(max(distancelist)*100)/100]


def check_configuration_consistence(label: str, component_list: list, all_defect_list: list, crystal: Crystal) -> bool:
    """Check consistence of a configuration, given its label. The consistence criteria are the ones defined in check_subconfiguration_consistence"""
    spec_list = [comp.get_species() for comp in component_list]
    config = name2conf(name=label, all_defect_list=all_defect_list, species=spec_list)
    position_list = [config.get_defect_position(d) for d in range(0, len(component_list))]
    return check_subconfiguration_consistence(species_list=spec_list, position_list=position_list, all_defect_list=all_defect_list, crystal=crystal)  # do not pass component_list because we don't need to check for correct number of components


def check_subconfiguration_consistence(species_list: list, position_list: list, all_defect_list: list, crystal: Crystal, component_list: list = []) -> bool:
    """Check that list of species and corresponding positions is consistent with the system defined in the input.
    This entails checking on: 1) correct number of components (only if component_list has been passed) 2) sublattices,
    3) no overlapping on same sites, 4) special case of "-1" permission."""
    # Check consistency of input (length of lists)
    if len(species_list) != len(position_list):
        produce_error_and_quit("In check_subconfiguration_consistence - Species list and position list have different lengths.")
    # Check if we have the correct number of components (only if the full component list has been passed)
    if len(component_list) > 0:
        component_species_list = [comp.get_species().get_name() for comp in component_list]
        for spec in species_list:
            if spec.get_name() != 'bulk':
                if spec.get_name() in component_species_list:
                    component_species_list.remove(spec.get_name())  # remove used component from list
                else:
                    return False
    # Check if species sit on the correct sublattices
    def_list = [] # list of defects occupied in this configuration
    perm_list = []  # list of permissions for each found defect
    for spec, pos in zip(species_list, position_list):
        if spec.get_name() != 'bulk':  # skip bulk species, assuming it is not a problem if user inputs a wrong lattice site
            # Transform position into a defect+translation pair (but translation is not needed)
            defect, _ = vect2deftrans(vec=pos, all_defect_list=all_defect_list)
            if defect.get_sudefect() not in spec.get_defects():
                return False
            else:  # locate and add corresponding permission
                perm_list.append((defect.get_sudefect(), spec.get_permissions()[spec.get_defects().index(defect.get_sudefect())]))
        else:  # for bulk species, add an artificial permission (just to maintain the correct species order in perm_list)
            perm_list.append((None, 1))
    if len(position_list) == 1:  # simply check that there are no negative permissions
        if not perm_list[0][1] > 0:
            return False
    else:
        okmatrix = np.zeros((len(position_list), len(position_list)))  # checking that species with a negative permission do have another species to share the site
        for i, j in combinations(range(len(position_list)), r=2):  # all possible combinations of species pairs (using indices)
            # (when using "combinations" instead of "permutations", order does not count)
            # Compute distance between species i and j
            dist = distance(vect1=position_list[i], vect2=position_list[j], crystal=crystal)
            # In order to be accepted, the distance between two species must be larger than the max distance between dumbbell atoms (defined as maxdbdist * db_displ)
            # unless at least one of the permissions is negative, in which case there must be a match between the negative permission and the other species
            if dist < maxdbdist * db_displ:
                if perm_list[i][1] > 0 and perm_list[j][1] > 0:  # if both permissions i and j are positive -> there cannot be superposition of defects
                    return False
                else:
                    # if the negative permission does not match the other species (and defects must match)
                    if not ((perm_list[i][0] == perm_list[j][0]) and (
                                (perm_list[i][1] < 0 and np.abs(perm_list[i][1]) == species_list[j].get_index()) or
                                (perm_list[j][1] < 0 and np.abs(perm_list[j][1]) == species_list[i].get_index()))):
                        return False
                    else:
                        # both species are very close to each other, occupy similar defects and one of them has a negative permission
                        # just need to check the "alignment" to be sure
                        if not are_equals(a=perm_list[i][0].get_align_dist(), b=dist):
                            return False
                        else:
                            okmatrix[i,j] = 1
                            okmatrix[j,i] = 1
        for k in range(len(position_list)):
            if perm_list[k][1] < 0:
                if np.sum(okmatrix[k,:]) == 0: # species k has a negative permission but no one on the same site
                    return False
    # If all checks were successful at this point, the configuration is acceptable!
    return True

def convergence_analysis(kiraloop: list, maxkira: dict, Wnum, Cnum, z, z0, L0, prec, JP, strain, temp, ndir, nsp, T, KL, Td, dir, refLij) -> None:
    """Looping on kira loop to compute transport coefficients for various kinetic ranges"""
    logger.info("         Performing automatic convergence study...")
    # Numerical evaluation of maxkira
    for key in maxkira:
        maxkira[key].append(0) #maxkira[key][3] =  partition function for kira=key
        maxkira[key].append(0)  # maxkira[key][4] =  partition function z0 for kira=key
        for a in maxkira[key][1]:
            maxkira[key][3] += Cnum[a,0]*maxkira[key][1][a]
            maxkira[key][4] += maxkira[key][1][a]
        maxkira[key].append({}) #maxkira[key][5] = dictionnary with L0 coefficients
        for a in maxkira[key][2]:
            maxkira[key][5][a] = 0
            for key2 in maxkira[key][2][a]:
                maxkira[key][5][a] += np.array(maxkira[key][2][a][key2](strain))*Wnum[key2,0]

    for direction in range(ndir): # initializing the output files
        with open(dir+'S{:.4f}_T{:.0f}_D{:.0f}.dat'.format(strain*100, temp, direction), 'w') as output:
            output.writelines(("#{:^19s} {:^9s} {:^12s}" + (" {:^15s}" * (int(2+nsp*(nsp+1)+0.5*nsp*(nsp-1)))) + "\n").format(
                '1) KineticRange[a]', '2) T[K]', '3) Strain[%]', '4) Z', '5) Z0',
                *[str(6 + b + sum([(nsp - e) for e in range(1, a + 1)])) + ') L_' + str(a) + str(b) for a in range(0, nsp) for b in range(a, nsp)],
                *[str(6 + int(0.5*nsp*(nsp+1)) + b + sum([(nsp - e) for e in range(1, a + 1)])) + ') L0_' + str(a) + str(b) for a in range(0, nsp) for b in range(a, nsp)],
                *[str(5 + 2*int(0.5*nsp*(nsp+1)) + b + sum([(nsp-1-e) for e in range(1, a + 1)])) + ') RE_' + str(a) + str(b) for a in range(0, nsp) for b in range(a+1, nsp)]))
    # quantities to be modified
    z2 = copy.deepcopy(z)
    z02 = copy.deepcopy(z0)
    L02 = copy.deepcopy(L0)
    T2 = copy.deepcopy(T)
    KL2 = copy.deepcopy(KL)
    Td2 = copy.deepcopy(Td)
    for a in kiraloop: # loop over the values we are going to study
        Lij0 = np.zeros((ndir, nsp, nsp), dtype=float)
        toremove = [] # kinetic interactions to be removed
        keytoremove = [] # dictionary keys to remove to be more efficient (works because kiraloop is sorted in reverse order)
        for key in maxkira:
            if key > a+0.01: # the corresponding configurations are now outside of the cluster
                z2 -= maxkira[key][3] # removing from the partition function
                z02 -= maxkira[key][4] # removing from the z0 partition function
                for key2 in maxkira[key][5]: # loop on final configuration distances
                    L02 -= maxkira[key][5][key2]  # removing the jump frequency from L0mat
                toremove += maxkira[key][0]
                keytoremove.append(key)

        toremove = sorted(toremove, reverse=True) # kinetic interactions to remove from KL2 and T2 (and Td2)
        for k in toremove: # removing columns from KL2 which is a list of list of list (direction, species, kinetic interaction)
            for direction in range(len(KL2)):
                for sp in range(nsp):
                    del KL2[direction][sp][k-1]
            del Td2[k-1]

        if prec is not None: # if precision tag is used the format of T2
            T2 = np.array(T2.tolist(), dtype='object')
            for k in toremove: # removing lines/columns from T2
                T2 = np.delete(T2, (k-1), axis=0)
                T2 = np.delete(T2, (k-1), axis=1)
            T2 = mp.matrix(T2)
            Lij = solve_prec(Tnum=T2, Klambda=KL2, Ninter=T2.rows, Nspec=nsp, Ndir=ndir, Pref=JP/z2, L0 =L02)
            for direction in range(ndir):
                Lij0[direction] = np.array(((JP/z2)*mp.matrix(L02[direction])).tolist(), dtype=float)
        else: # performing calculation with default precision using scipy
            T2 = T2.tolil()
            for k in toremove:  # removing lines/columns from T2
                T2.rows = np.delete(T2.rows, k-1)
                T2.data = np.delete(T2.data, k-1)
                T2._shape = (T2._shape[0] - 1, T2._shape[1])
                T2 = T2.transpose()
                T2.rows = np.delete(T2.rows, k-1)
                T2.data = np.delete(T2.data, k-1)
                T2._shape = (T2._shape[0] - 1, T2._shape[1])
            T2 = T2.transpose().tocsc()
            Lij, _, _ = solve_def(Tnum=T2, Klambda=KL2, Ndir=ndir, Nspec=nsp, Pref=JP/z2, L0=L02, Site=[])
            for direction in range(ndir):
                Lij0[direction] = float(JP/z2)*(np.array(L02[direction], dtype=float))

        for direction in range(ndir):
            if not are_equal_arrays(A=refLij[direction], B=0*refLij[direction]):
                with open(dir + 'S{:.4f}_T{:.0f}_D{:.0f}.dat'.format(strain*100, temp, direction), 'a') as output:
                    output.writelines("{:^20.3f} {:^9.1f} {:^+12.6f} {:^15.6E} {:^15.1f}".format(a, temp, strain*100, float(z2), float(z02)))
                    for s1 in range(nsp):
                        for s2 in range(s1, nsp):
                            output.writelines(" {:^+15.6E}".format(Lij[direction][s1,s2]))
                    for s1 in range(nsp):
                        for s2 in range(s1, nsp):
                            output.writelines(" {:^+15.6E}".format(Lij0[direction][s1,s2]))
                    for s1 in range(nsp):
                        for s2 in range(s1+1, nsp):
                            output.writelines(" {:^+15.6E}".format(float(z2/z)*Lij[direction][s1,s2]/refLij[direction][s1,s2]-1))

                    output.writelines("\n")
        # cleaning maxkira for next iteration
        for key in keytoremove:
            del maxkira[key] # removing keys in maxkira
        for key in maxkira: # renumbering the remaing kinetic interactions in maxkira
            for k in range(len(maxkira[key][0])-1, -1, -1):
                maxkira[key][0][k] -= len([b for b in toremove if b < maxkira[key][0][k]])
    return None

def dataset_to_defects(dataset: dict, crystal: Crystal, symop_list: list) -> list:
    n_defects = int(dataset['uniquepos'][0])
    defect_list = []  # list of all symmetry unique defects
    all_defect_list = []  # list of all defects, including symmetric ones
    dindex = 0
    doubledefects = []
    for i in range(n_defects):
        sublattice = np.array([evalinput(e) for e in dataset['uniquepos'][i * 4 + 2:i * 4 + 5]], dtype=float)
        sublattice_nod = np.array([evalinputd0(e) for e in dataset['uniquepos'][i * 4 + 2:i * 4 + 5]], dtype=float)
        if dataset['uniquepos'][i * 4 + 1] is 'o':  # convert positions to supercell base if necessary
            sublattice = change_base(arr=sublattice, crystal=crystal)
            sublattice_nod = change_base(arr=sublattice_nod, crystal=crystal)
        sublattice = apply_pbc(vec=sublattice)  # apply boundary conditions
        sublattice_nod = apply_pbc(vec=sublattice_nod)
        defect_list.append(Defect(crystal=crystal, sublattice=sublattice, index=dindex, sublattice_nod=sublattice_nod))  # add to list of symmetry-unique defects
        defect_list[i].find_symeq(symop_list)  # find all symmetry equivalents of this defect
        defect_list[i].set_align_dist()
        doubles = False
        # Check that this defect is not equivalent to a previously found defect
        for new in defect_list[i].get_symeq():
            for iold, old in enumerate(defect_list[:-1]):
                if (np.allclose(new.get_sublattice(), old.get_sublattice(), atol=tol, rtol=tol)):
                    doubles = True
                    logger.info('!! Defect [ {} {:+8.5f} {:+8.5f} {:+8.5f} ] is ignored because symmetrically equivalent to defect {}.'.format(dataset['uniquepos'][i*4+1], *[evalinput(e) for e in dataset['uniquepos'][i*4+2:i*4+5]], old.get_index()))
                    doubledefects.append(i)
                    break
            if doubles:
                break
        if not doubles:
            all_defect_list += defect_list[i].get_symeq()
            dindex += len(defect_list[i].get_symeq())
            defect_list[i].get_info()
    doubledefects=list(reversed(sorted(doubledefects)))
    for i in doubledefects:
        del defect_list[i]
    return [defect_list, all_defect_list, doubledefects]


def dataset_to_jumps(dataset: dict, crystal: Crystal, symop_list: list, all_defect_list: list, species_list: list, sym_unique: bool, component_list: list) -> list:
    su_jump_list = []  # list of symmetry unique jump objects
    jump_list = []  # list of all possible jump objects
    jumpmech2 = (''.join(str(e) + ' ' for e in dataset['jumpmech'])).split(sep='%%')[1:]  # '%%' symbol separates jump mechanisms
    for i in range(len(jumpmech2)-1,-1,-1):  # Checking the number of species for this jump
        tmpcomp = [a.get_species().get_index() for a in component_list] # non-bulk species in cluster
        for sp in [int(b) for b in [a.split()[0] for a in jumpmech2[i].split(sep='>')[1:]] if b !='0']: # non bulk species in jump constraints
            if sp in tmpcomp:
                del tmpcomp[tmpcomp.index(sp)]
            else:
                logger.info("!! Jump {} is not taken into account because it requires components that are not part of the cluster".format(jumpmech2[i].split(sep=' ')[2]))
                del jumpmech2[i]
                break
    n_jumps = len(jumpmech2)
    logger.info("  Found {} jump mechanisms".format(n_jumps))
    for i in range(0, n_jumps):  # Loop on jump mechanisms
        jumpmech3 = jumpmech2[i].split(sep=' ')[1:]
        su_jump_list.append(JumpMech(index=i, name=jumpmech3[1]))  # List of symmetry unique jump mechanisms
        # Check user-input constraints for this jump
        tvec = np.array([0, 0, 0], dtype=float)
        tmplist = []  # temporary constraint list
        for j in range(0, int(jumpmech3[0])):  # Loop on constraints for this jump
            # Check for correct constraint format in input file
            if jumpmech3[7 + j * 10] != '>':
                produce_error_and_quit("In dataset_to_jumps - Problem in the format of constraints {} of jump {}".format(j+1, i+1))
            # Check if there is a transmutation reaction (i.e., one of the atoms changes species during the jump)
            if jumpmech3[3 + j * 10] != jumpmech3[8 + j * 10]:
                produce_error_and_quit("In dataset_to_jumps - There is a transmutation on constraint {} of jump {}, which is not handled by the current version.".format(str(j), jumpmech3[1]))
            # If everything looks fine, add this constraint to the temporary list
            tmplist.append(jumpmech3[2 + j * 10:12 + j * 10])
        jumpmech3 = jumpmech3[0:2] + flat_list(sorted(tmplist, key=lambda x: -int(x[1])))  # sort constraints by species in decreasing order (from highest index to 0)
        # Add constraints to jump object
        for j in range(0, int(jumpmech3[0])):  # Loop on constraints for this jump
            # Initial species and position
            inispec = species_list[int(jumpmech3[3 + j * 10]) - 1]
            inipos = np.array([evalinput(e) for e in jumpmech3[4 + j * 10:7 + j * 10]], dtype=float)
            inipos_nod = np.array([evalinputd0(e) for e in jumpmech3[4 + j * 10:7 + j * 10]], dtype=float)
            if jumpmech3[2 + j * 10] == 'o':  # convert to supercell basis if necessary
                inipos = change_base(arr=inipos, crystal=crystal)
                inipos_nod = change_base(arr=inipos_nod, crystal=crystal)
            # Translate every atom so that the initial position of the atom in the first constraint is in the initial supercell
            if j == 0:
                tvec = -np.floor(inipos + tol_pbc * np.array([1, 1, 1]))
            inipos += tvec
            inipos_nod +=tvec
            # Final species and position
            finspec = species_list[int(jumpmech3[8 + j * 10]) - 1]
            finpos = np.array([evalinput(e) for e in jumpmech3[9 + j * 10:12 + j * 10]], dtype=float)
            finpos_nod = np.array([evalinputd0(e) for e in jumpmech3[9 + j * 10:12 + j * 10]], dtype=float)
            if jumpmech3[2 + j * 10] == 'o':  # convert to supercell basis if necessary
                finpos = change_base(arr=finpos, crystal=crystal)
                finpos_nod = change_base(arr=finpos_nod, crystal=crystal)
            finpos += tvec  # translating (as above)
            finpos_nod += tvec  # translating (as above)
            # Add constraint object to this jump object
            su_jump_list[i].add_constraint(JumpConstraint(inispecies=inispec, iniposition=copy.copy(inipos), iniposition_nod=copy.copy(inipos_nod), finspecies=finspec, finposition=copy.copy(finpos), finposition_nod=copy.copy(finpos_nod)))
        # Check jump, making sure that species sit on correct sublattices and there are no overlapping components on the same site (unless permission -1 is present)
        if not su_jump_list[i].check_jump(all_defect_list=all_defect_list, component_list=component_list, crystal=crystal):
            produce_error_and_quit("In dataset_to_jumps - Jump {} is not consistent with sublattice permissions and/or site occupancy.".format(su_jump_list[i].get_name()))
        # Search and assign defect type and translation to each constraint
        for cons in su_jump_list[i].get_constraints():
            cons.vect2deftrans_constraint(all_defect_list=all_defect_list)
        # Check that this jump is not symmetrically equivalent to other jumps (but do not remove it!)
        for poteq in jump_list:
            if su_jump_list[i] == poteq:
                logger.info("!! Jumps {} and {} are symmetrically equivalent.".format(str(i), str(poteq.get_index())))
        jump_list += su_jump_list[i].find_symeqs(symops=symop_list, all_defect_list=all_defect_list)
    if sym_unique:
        return su_jump_list
    else:
        return jump_list


def dataset_to_species(dataset: dict, defect_list: list, doubledefects: list = []) -> list:
    n_species = int(dataset['species'][0])
    n_defects = int(dataset['uniquepos'][0])  # n_defects is not taken from len(defect_list) because doubledefects have already been removed from that list
    n_components = 0
    species_list = []  # list of species
    component_list = []  # list of components
    for i in range(0, n_species):
        # For each species, create the list of defects that are allowed, based on user input
        tmp_permissions = dataset['species'][i * (2 + n_defects) + 2:i * (2 + n_defects) + (2 + n_defects)]
        if len(tmp_permissions) != n_defects:  # wrong list of permissions in user input file!
            produce_error_and_quit("In dataset_to_species - The number of permissions for each species does not match the number of defects in UNIQUEPOS!\n" \
                                   "Correct input for each species: << X  0/1 0/1 ... spec_name >>,\n" \
                                   "where X is the number of components for that species, and 0/1 marks if the species can occupy the corresponding position in UNIQUEPOS.\n" \
                                   "Check that the amount of 0/1 flags matches the amount of defects defined in UNIQUEPOS.")
        for idef in doubledefects:
            del tmp_permissions[idef]
        tmp_bool = np.array(tmp_permissions, dtype=int).astype(bool)  # OBS! anything that is not 0 returns True!
        tmp_defectlist = [k for (k, v) in zip(defect_list, tmp_bool) if v]  # selection of items in defect_list corresponding to a "True" value in tmp_bool
        tmp_reduced_permissions = np.array([k for (k, v) in zip(tmp_permissions, tmp_bool) if v], dtype=int)
        tmp_name = dataset['species'][i * (2 + n_defects) + (2 + n_defects)].lower()
        # Check that there is at least one positive permission, otherwise give warning
        if len(tmp_reduced_permissions) == 0:
            logging.info("WARNING! Species {} is not permitted on any sublattice. Is this what you meant doing?".format(tmp_name))
        # Check species name (name 'bulk' is forbidden because it is assigned to the matrix atoms)
        if tmp_name == 'bulk':
            produce_error_and_quit("Name 'bulk' for a species is forbidden. Choose another name, and keep in mind that you don't need to define the matrix atoms as a species in the input file.")
        # Append new species to species list
        species_list.append(Species(index=i + 1, name=tmp_name, defects=tmp_defectlist, permissions=[int(p) for p in tmp_reduced_permissions]))
        species_list[-1].get_info()
        # Check negative permissions (-n, where n is the species that has to be on the same site in order to activate permission)
        # Create adequate number of components for each species
        for j in range(0, int(dataset['species'][i * (2 + n_defects) + 1])):
            component_list.append(Component(species=species_list[i], index=n_components))
            component_list[n_components].get_info()
            n_components += 1
    # Check for bulkspecies tag (species requiring special treatment to avoid double counting of symmetric configurations)
    if dataset['bulkspecies'] is not None:
        for bs in dataset['bulkspecies']:
            species_list[int(bs)-1].set_bulk()
    # Create bulk (matrix) species (for jump constraints)
    species_list.append(Species(index=0, name="bulk", defects=[], permissions=[]))
    #species_list[-1].set_bulk()
    return [species_list, component_list]


def distance(vect1: np.ndarray, vect2: np.ndarray, crystal: Crystal) -> float:
    """Returns the distance between vectors 1 and 2"""
    # First the difference between both vectors is transformed into the orthonormal basis
    ortho_dist = change_base(arr=vect2 - vect1, crystal=crystal, inv=True)
    # In strain calculations, ortho_dist contains the 'e' variable, but we take the distance at e=0
    if ortho_dist.dtype == 'O':
        ortho_dist = np.array([x.subs('e',0) for x in ortho_dist], dtype=float)
    return np.linalg.norm(ortho_dist)


def evalinputS(number: str) -> object: # for symbolic strain
    return eval(number.replace('sqrt', 'sym.sqrt'))


def evalinputd0(number: str) -> float:
    return float(eval(number.replace("d", str(0)).replace('sqrt', 'np.sqrt')))


def evalinput(number: str) -> float:
    return float(eval(number.replace("d", str(db_displ)).replace('sqrt', 'np.sqrt')))


def find_cpg_symmetries(crystal: Crystal, symop_list: list, cpg: np.ndarray, inhom: bool = False) -> list:
    """Find subset of symmetry operations that maintain the symmetry of the cpg (for kinetic interactions)"""
    logger.info("Searching for symmetry operations valid for the cpg...")
    # CPG needs to be written in supercell format
    cpg = sym.matrix2numpy(sym.Matrix(change_base(arr=cpg, crystal=crystal)).subs(e, strainval), dtype='float')
    # Loop over symmetry operation list
    # Symmetry operation is kept if: when applied to cpg, it returns the cpg or its exact inverse (the latter only in case cpg is perp. to mirror plane)
    symop_cpg_list = []
    for symop in symop_list:
        if (not inhom) or (inhom and np.dot(symop.get_translation(), cpg)==0):
            R = symop.get_rotation()   # rotation matrix
            new_cpg = np.dot(R, cpg)  # It should not be necessary to add the translation in this case
            if are_equal_arrays(new_cpg, cpg): # check if the new cpg is the same, or the exact inverse in case cpg is perp. to mirror plane
                symop_cpg_list.append(symop)
                symop.set_cpgsign(1)
            elif are_equal_arrays(new_cpg, -cpg):
                symop_cpg_list.append(symop)
                symop.set_cpgsign(-1)
    # Find identity matrix and move it to the beginning of the list
    for idx, s in enumerate(symop_cpg_list):
        if are_equal_arrays(s.get_rotation(), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
            identity = symop_cpg_list[idx]
            del symop_cpg_list[idx]
            symop_cpg_list = [identity] + symop_cpg_list
            break
    return symop_cpg_list


def find_symmetry_operations(crystal: Crystal) -> list:
    """Gives a list of all valid symmetry operations (SymOp class) of a given crystal"""
    # Search for all possible rotation matrices is done with a recursive function that creates all possible 3x3 matrices with (-1,0,1)
    # by updating the list of symop as a function attribute
    def build_unitary_matrices(ix, args_list) -> None:
        if ix <= 9:  # Call function recursively until 9 indices are defined
            for i in range(-1,2):
                new_args_list = args_list.copy()
                new_args_list.append(i)  # a new index is added to args_list
                build_unitary_matrices(ix+1, new_args_list)
        else:  # Now that we have 9 indices, check if matrix can be a symmetry operation (determinant = 1 and 6-idempotent)
            Nmat = np.array(args_list, dtype=int).reshape((3,3))  # Building a 3x3 matrix from the 9 indices
            determ = np.linalg.det(Nmat)
            if are_equals(np.fabs(determ), 1):  # if determinant is 1 or -1
                for j in range(1, 7):  # check if 6-idempotent
                    Nmat2 = np.linalg.matrix_power(Nmat, j)    # Nmat2 = np.matrix(Nmat) ** j
                    if are_equal_arrays(Nmat2, np.identity(3)):
                        build_unitary_matrices.symop_list.append(SymOp(rotation=Nmat))  # Store found matrix

    # Here the build_unitary_matrices recursive function is called
    build_unitary_matrices.symop_list = []  # initialization of the symop_list as function attribute, which is updated inside the function
    build_unitary_matrices(ix=1, args_list=[])
    symop_list = build_unitary_matrices.symop_list

    # Now check if At * A = id, where A = C * M * C^-1, and keep only matrices fulfilling this criterion
    # (loop is done backward so that matrices can be removed safely without interfering with the loop)
    C = crystal.get_primvectorsnum()
    Cinv = crystal.get_invvectorsnum()
    for j in range(len(symop_list)-1, -1, -1 ):
        M = symop_list[j].get_rotation()
        A = np.dot(C, np.dot(M, Cinv))  # C*M*C^-1
        if not are_equal_arrays(np.dot(np.transpose(A), A), np.identity(3)):  # Is At*A - id == 0 ?
            del symop_list[j]   # matrix is removed

    # Modify matrices for 2D or 1D crystals
    if crystal.get_dim() <= 2:
        for j in range(len(symop_list) - 1, -1, -1):
            A = symop_list[j].get_rotation()
            A[2,:] = np.array([0,0,1])
            A[:,2] = np.array([0,0,1])
            symop_list[j].set_rotation(A)
    if crystal.get_dim() == 1:
        for j in range(len(symop_list) - 1, -1, -1):
            A = symop_list[j].get_rotation()
            A[1,:] = np.array([0,1,0])
            A[:,1] = np.array([0,1,0])
            symop_list[j].set_rotation(A)

    # Finally, check for identical matrices and remove them (backward loop for the same reason as above)
    for j in range(len(symop_list)-1, -1, -1):
        for k in range(len(symop_list)-1, j, -1):
            if are_equal_arrays(symop_list[k].get_rotation(), symop_list[j].get_rotation()):
                del symop_list[k]
    # We have symmetry matrices for the lattice but not yet for the crystal. Need to look for fractional translations
    SGop = 0  # number of space group operations
    PGop = 0 # number of point group operations
    basis = crystal.get_basislist()
    if len(basis) > 1:
        basis = np.array(sorted([a for a in basis], key=lambda x: [x[0], x[1], x[2], x[3]]))
        eqbasis = apply_symmetry_operations(vector_list=[a[1:4] for a in basis], symop_list=symop_list, unique=False, applytrans=False)
        symtoremove=[] # indexes of symmetry operations to be removed if any
        for idx, eqpositions in enumerate(eqbasis):  # loop on symmetry operations
            for id2, pos in enumerate(eqpositions):  # loop on basis atoms
                #apply periodic boundary conditions
                eqpositions[id2] = apply_pbc(np.array(pos))
            # find possible translations on the symmetry equivalent of the first atom
            # this symmetry equivalent of the first basis atom must coincide with one of the basis atoms
            # hence the possible translations; same goes for each and every symmetry operations
            possible_trans = []
            for pos in basis:
                if pos[0]==basis[0][0]:
                    possible_trans.append(pos[1:4]-eqpositions[0])
            found = False
            for trans in possible_trans:
                translated_symeq = []
                for pos in eqpositions:
                    translated_symeq.append(apply_pbc(pos+trans))
                if are_equal_arrays(np.array(sorted([np.array([basis[i][0]]+list(a)) for i,a in enumerate(translated_symeq)], key=lambda x:[x[0], x[1], x[2], x[3]])), basis):
                    found = True
                    if symop_list[idx].get_translation() is None:
                        symop_list[idx].set_translation(trans)
                    else:
                        symop_list.append(SymOp(rotation=symop_list[idx].get_rotation()))
                        symop_list[-1].set_translation(trans)
                    if are_equal_arrays(trans, np.array([0,0,0])):
                        PGop += 1
                    else:
                        SGop += 1
            if not found:
                symtoremove.append(idx)
        symtoremove.sort(reverse=True)
        for idx in symtoremove:
            del symop_list[idx]
    else:
        for idx in range(len(symop_list)):
            PGop += 1
            symop_list[idx].set_translation(np.array([0,0,0]))
    logger.info("  Found {} symmetry operations ({} point group op. and {} space groups op.)".format(len(symop_list),PGop,SGop))
    return symop_list


def find_possible_permutations(specs: list, species_list):
    # Creating permutations list "name_perms" for configuration names
    # specs is a list of species for each component
    # species_list is the list of all species in the system, the last one being bulk
    tmp = []
    for isp, sp in enumerate(species_list[:-1]):  # loop over each species, except the last one which is bulk
        indices = []
        for i, x in enumerate(specs):
            if x == sp:
                indices.append(i)
        tmp.append(list(permutations(indices)))
    name_perms = list(product(*tmp))
    for id, variation in enumerate(name_perms):
        name_perms[id] = flat_list(variation)
    return name_perms


def flat_list(listoflist) -> list:
    """Takes a list of lists and turns it into a simple list"""
    flat_list = [inner for outer in listoflist for inner in outer]
    return flat_list


def id_jump_freq(config1: Configuration, config2: Configuration, jumpmech: JumpMech, freq_list: dict, symop_list: list, name_perms: list, all_defect_list: list, Nfreq: int):
    """Analyzes the jump between config1 and config2 via a jump mechanism to find the corresponding jump frequency"""
    tmp = [config1.get_thermoint(), config2.get_thermoint()]
    dissofreq = False  # by default the jump is not a dissociation jump
    tmp.sort()
    name = str(tmp[0])+'_'+str(tmp[1])+'_'+str(jumpmech.get_index())
    if config1.get_label()[0] == 'd' or config2.get_label()[0] == 'd':
        dissofreq = True
        name = 'd'+name
    beyondThra = config1.get_beyond() or config2.get_beyond()
    newfreq = False # turns to true if a new jump frequency is found
    if freq_list.get(name) is None:
        newfreq = True # this jump frequency does not exist, we need to create a new one
        w = JumpFrequency(number=Nfreq+1, config_ini=config1, config_fin=config2, jump=jumpmech, symop_list=symop_list,
                          dissofreq=dissofreq, name_perms=name_perms, all_defect_list=all_defect_list, beyond=(config1.get_beyond() or config2.get_beyond()))
    else:
        tmp = [config1.get_label(), config2.get_label()] # both configuration names
        tmp.sort() # sorted names by alphanumerical order
        tmp =''.join(tmp) # joined names, as in jump frequency symmetry equivalents
        found = False # turns to True when the jump frequency is found
        if dissofreq or beyondThra: # no need to check symmetry for kinetically or thermodynamically dissociate configurations
            w = freq_list[name][0]
        else:
            for oldfreq in freq_list[name]: # all jump frequency that are between similar initial and
                                            # final configuration and with the same migration mechanism
                if oldfreq.get_symeq().get(tmp) is not None:
                    found = True #this jump frequency already exists
                    w = oldfreq
                    break
            if not found:  # this jump frequency does not exist, we need to create a new one
                newfreq = True
                w = JumpFrequency(number=Nfreq+1, config_ini=config1, config_fin=config2, jump=jumpmech,
                                  symop_list=symop_list, dissofreq=dissofreq, name_perms=name_perms,
                                  all_defect_list=all_defect_list, beyond=(config1.get_beyond() or config2.get_beyond()))
    return [w, newfreq, name] #jump frequency object (w), whether it is new or not and its name in the dictionary freq_list


def name2conf(name: str, all_defect_list: list, species: list) -> Configuration:
    tmpname = name.replace('c', '').replace('d', '').split(sep='|')[0].split(sep='_')
    translations=[0, 0, 0] + [int(a) for a in tmpname[-1].replace('m','p-').split(sep='p')[1:]]
    defects = [all_defect_list[int(a)] for a in tmpname[:-1]]
    conf = Configuration(defects=copy.copy(defects), translations=copy.copy(translations), label=name)
    conf.set_species(species)
    return conf


def printxyz(direc: str, input: list, thconfig_list: dict, freq_list: dict, n_components: int, crystal: Crystal, component_list: list, name_perms: list) -> None:
    if not os.path.isdir(direc+'CONFIGURATIONS'):
        os.makedirs(direc+'CONFIGURATIONS')
    k = 0
    for conf in thconfig_list:
        with open(direc+'CONFIGURATIONS/conf_{}.dat'.format(thconfig_list[conf].get_thermoint()), 'w') as output:
            limitp = [-1, -1, -1]
            limitm = [0, 0, 0]
            if len(input) == 2:
                if input[1] == 'wrap':
                    for b in range(3):
                        limitp[b] = 1+int(np.max(a=[thconfig_list[conf].get_translations()[3*e+b] for e in range(n_components)]))
                        limitm[b] = -1+int(np.min(a=[thconfig_list[conf].get_translations()[3*e+b] for e in range(n_components)]))
                else:
                    try:
                        n = int(float(input[1]))
                        limitp = [n, n, n]
                        limitm = [-n, -n, -n]
                    except ValueError:
                        logger.info("!! Second item in PRINTXYZ tag must be 'wrap' or an integer")
            output.writelines('{:.0f}\n'.format(n_components+(limitp[0]+1-limitm[0])*(limitp[1]+1-limitm[1])*(limitp[2]+1-limitm[2])*len(crystal.get_basislist())))
            if input[0] == 'o':  # orthonormal basis
                cell = "".join('{:8.4f} {:8.4f} {:8.4f}\n {:8.4f} {:8.4f} {:8.4f}\n {:8.4f} {:8.4f} {:8.4f}'.format(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            elif input[0] == 's':  # supercell basis
                cell = "".join(['{:8.4f} {:8.4f} {:8.4f}\n '.format(*(crystal.get_primvectors()[b]*(limitp[b]-2*limitm[b]))) for b in range(3)])
                output.writelines(cell)
            else:
                logger.info('!! Problem in the PRINTXYZ tag; must be s (supercell basis) or o (orthonormal basis)')
            #output.writelines("Lattice=\"{}\" Properties=species:S:1:pos:R:3 Time=0.0\n".format(cell))
            output.writelines("\n")
            if input[0] == 'o':  # orthonormal basis
                for idx in range(n_components):
                    output.writelines('{:12s} {:8.4f} {:8.4f} {:8.4f}\n'.format(component_list[idx].get_species().get_name(),*change_base(arr=thconfig_list[conf].get_defect_position(def_idx=idx),crystal=crystal, inv=True)))
            elif input[0] == 's':  # supercell basis
                for idx in range(n_components):
                    output.writelines('{:12s} {:8.4f} {:8.4f} {:8.4f}\n'.format(component_list[idx].get_species().get_name(),*thconfig_list[conf].get_defect_position(def_idx=idx)))
            for superx in range(limitm[0], limitp[0]+1):
                for supery in range(limitm[1], limitp[1]+1):
                    for superz in range(limitm[2], limitp[2]+1):
                        trans = np.array([superx, supery, superz])
                        for bulk in crystal.get_basislist():
                            if input[0] == 'o':  # orthonormal basis
                                output.writelines('{:12s} {:8.4f} {:8.4f} {:8.4f}\n'.format('Bulk{:.0f}'.format(bulk[0]),*change_base(arr=(bulk[1:4]+trans), crystal=crystal, inv=True)))
                            elif input[0] == 's':  # supercell basis
                                output.writelines('{:12s} {:8.4f} {:8.4f} {:8.4f}\n'.format('Bulk{:.0f}'.format(bulk[0]),*(bulk[1:4]+trans)))
        if k == 0:
            k = 1
            if not os.path.isdir(direc+'JUMP_FREQUENCIES'):
                os.makedirs(direc+'JUMP_FREQUENCIES')
        for freq0 in freq_list:
            for freq in  freq_list[freq0]:
                with open(direc+'JUMP_FREQUENCIES/freq_{}.dat'.format(freq.get_number()), 'w') as output:
                    if input[0] == 's': # print crystal vector only when using supercell basis
                        for b in range(3):
                            output.writelines('{:8.4f} {:8.4f} {:8.4f}\n'.format(*crystal.get_primvectors()[b]))
                    limitp = [-1, -1, -1]
                    limitm = [0, 0, 0]
                    ini_position, fin_position = untranslatedfinal(jumpmech=freq.get_jump(), component_list=component_list, name_perms=name_perms,
                                                finconf=freq.get_config_fin(), iniconf=freq.get_config_ini(), fnum=freq.get_number())
                    if len(input) == 2:
                        if input[1] == 'wrap':  # must be changed if we change the output
                            for b in range(3):
                                limitp[b] = int(np.max([np.floor(flat_list(flat_list([ini_position, fin_position]))[3*e+b]+tol_pbc) for e in range(2*n_components)]))
                                limitm[b] = int(np.min([np.floor(flat_list(flat_list([ini_position, fin_position]))[3*e+b]+tol_pbc) for e in range(2*n_components)]))
                        else:
                            try:
                                n = int(float(input[1]))
                                limitp = [n, n, n]
                                limitm = [-n, -n, -n]
                            except ValueError:
                                logger.info("!! Second item in PRINTXYZ tag must be 'wrap' or an integer")
                    output.writelines('{:.0f}\n'.format(n_components + (limitp[0]+1-limitm[0])*(limitp[1]+1-limitm[1])*(limitp[2]+1-limitm[2])*len(crystal.get_basislist())))
                    output.writelines('\n')
                    # Print coordinates
                    if input[0] == 'o':  # orthonormal basis
                        for idx in range(n_components):
                            output.writelines('{:12s} {:8.4f} {:8.4f} {:8.4f}  > {:8.4f} {:8.4f} {:8.4f}\n'.format(component_list[idx].get_species().get_name(),
                                *change_base(arr=ini_position[idx], crystal=crystal,inv=True),
                                *change_base(arr=fin_position[idx], crystal=crystal, inv=True)))
                    elif input[0] == 's': # supercell basis
                        for idx in range(n_components):
                            output.writelines('{:12s} {:8.4f} {:8.4f} {:8.4f}  > {:8.4f} {:8.4f} {:8.4f}\n'.format(component_list[idx].get_species().get_name(),
                                *ini_position[idx], *fin_position[idx]))
                    for superx in range(limitm[0], limitp[0]+1):
                        for supery in range(limitm[1], limitp[1]+1):
                            for superz in range(limitm[2], limitp[2]+1):
                                trans = np.array([superx, supery, superz])
                                for bulk in crystal.get_basislist():
                                    if input[0] == 'o':  # orthonormal basis
                                        output.writelines('{:12s} {:8.4f} {:8.4f} {:8.4f}  > {:8.4f} {:8.4f} {:8.4f}\n'.format('Bulk{:.0f}'.format(bulk[0]),*change_base(arr=(bulk[1:4]+trans), crystal=crystal, inv=True), *change_base(arr=(bulk[1:4]+trans), crystal=crystal, inv=True)))
                                    elif input[0] == 's':  # supercell basis
                                        output.writelines('{:12s} {:8.4f} {:8.4f} {:8.4f}  > {:8.4f} {:8.4f} {:8.4f}\n'.format('Bulk{:.0f}'.format(bulk[0]),*(bulk[1:4]+trans), *(bulk[1:4]+trans)))


def print_license_notice() -> None:
    """Print license notice at the beginning of each log file"""
    license_notice = "KineCluE - Copyright (C) 2018 CEA, École Nationale Supérieure des Mines de Saint-Étienne\n" \
                     "This program comes with ABSOLUTELY NO WARRANTY; for details, see the license terms.\n" \
                     "This is free software, and you are welcome to redistribute it\n" \
                     "under certain conditions; see the license terms for details.\n" \
                     "You are required to cite the following paper when using KineCluE or part of KineCluE:\n" \
                     "T. Schuler, L. Messina and M. Nastar, Computational Materials Science (2019) [doi: https://doi.org/10.1016.j.commatsci.2019.109191]"
    logger.info(license_notice)


def produce_error_and_quit(error_message: str) -> None:
    logger.info("ERROR! " + error_message)
    raise Exception("ERROR! " + error_message)


def produce_strain_input(dataset: dict, input_file: str) -> dict:
    """This function reads the input_dataset dictionary and updates it to include
    new defects and jumps arising from the broken symmetry due to strain. It returns
    the updated dataset and creates a "_strained" input file that is saved in the output
    directory. The crystals (deformed and undeformed) are loaded from the saved pkl files."""

    logger.info('\n ------------------------ Now applying strain ------------------------- ')
    crystal_name = dataset['crystal'][0]  # user-input crystal name
    directory = dataset['directory'][0]
    # Load crystal files
    file_name_undef = directory + 'crystal_' + crystal_name + '.pkl'
    file_name_def = directory + 'crystal_' + crystal_name + '_strained' + '.pkl'
    if os.path.exists(file_name_undef) and os.path.exists(file_name_def):
        crystal_undef, symop_undef = pickle.load(open(file_name_undef, 'rb'))
        with sym.evaluate(False):
            crystal_def, symop_def = pickle.load(open(file_name_def, 'rb'))
    else:
        produce_error_and_quit("In produce_strain_input - Crystal file not found.")
    # Create list of defects in perfect crystal
    logger.info("Creating list of defect symmetry equivalents in unstrained crystal...")
    [defect_list, all_defect_list, sptoremove] = dataset_to_defects(dataset=dataset, crystal=crystal_undef, symop_list=symop_undef)
    # removing species for double defects
    tmp_species_dataset=[dataset['species'][0]]
    i = 1
    for k in range(int(dataset['species'][0])):
        tmp_species_dataset.append(dataset['species'][i])
        i+=1
        for j in range(int(dataset['uniquepos'][0])):
            if not j in sptoremove:
                tmp_species_dataset.append(dataset['species'][i])
            i+=1
        tmp_species_dataset.append(dataset['species'][i])
        i+=1
    dataset['species'] = tmp_species_dataset
    n_defects = len(defect_list)
    # Check defect symmetry break with deformation
    # Here we loop over each item in defect list (unique defects)
    # In each defect, we look at the symmetry equivalents and perform on each of the latter the symmetry operations
    # of the strained crystal. For each, if by symmetry we have re-found at least one of the already listed symmetry
    # equivalents, the symmetry has not been broken. But if we do not find any, the symmetry has been broken and a
    # new unique defect is added to broken_defect_list.
    logger.info("Searching for defect broken symmetries under strain...")
    broken_defect_list = []  # List of symmetry unique defects in the deformed crystal and its corresponding original defect label
    for idx, defect in enumerate(defect_list):
        broken_defect_list.append([defect, idx])  # [ [defect1, Index1], [defect2, Index2], etc.. ]
        tmp_symeq_list = [defect]
        for defsym in defect.get_symeq():
            tmp_symeq_list.append(defsym)
            defsym.find_symeq(symop_def)
            double = False
            for new in defsym.get_symeq():
                for old in tmp_symeq_list[:-1]:
                    if np.allclose(new.get_sublattice(), old.get_sublattice(), atol=tol, rtol=tol):
                        double = True
                        break
                if double:
                    break
            if not double:
                if not (np.allclose(defsym.get_sublattice(), defect.get_sublattice(), atol=tol, rtol=tol)): #&#WHY??
                    broken_defect_list.append([defsym, idx])
                    logger.info(
                        "  !!  Broken symmetry for Defect {}: adding defect on sublattice {} (ortho no def {}).".format(
                            defect.get_index(), defsym.get_sublattice(),
                            change_base(arr=defsym.get_sublattice(), crystal=crystal_undef, inv=True)))
    n_broken_defects = len(broken_defect_list)
    # Update dataset[uniquepos] (need to do it here otherwise it gives bugs in dataset_to_species)
    dataset['uniquepos'] = [str(len(broken_defect_list))] + ([None] * (4 * len(broken_defect_list)))
    for idx, defect in enumerate(broken_defect_list):  # !! Each item in broken_defect_list is a list of 2 items: [0] defect; [1] Original defect index
        dataset['uniquepos'][1 + 4 * idx:5 + 4 * idx] = ["s"] + [str(sub) for sub in defect[0].get_sublattice()]
    del defect, defsym, tmp_symeq_list, new, old, double, idx
    # Update species permissions in dataset, taking into account the new broken_defect_list
    logger.info("Analyzing species and their sublattice occupancies in unstrained crystal...")
    n_species = int(dataset["species"][0])
    tmp_species_dataset = [str(n_species)]  # the updated dataset["species"] is re-written in this list
    for i in range(0, n_species):
        tmp_species_dataset.append(dataset["species"][1 + i * ( 2 + n_defects ) ])
        for defect in broken_defect_list:  # add permission for each defect
            tmp_species_dataset.append(dataset["species"][2 + i * ( 2 + n_defects ) + defect[1]])  # defect[1] contains the defect type according to the original defect_list
        tmp_species_dataset.append(dataset["species"][( 2 + n_defects ) + i * ( 2 + n_defects )])
    dataset["species"] = list(tmp_species_dataset)
    del tmp_species_dataset, i, defect
    # Create species list with new updated species (to be used to identify the jumps)
    [species_list, component_list] = dataset_to_species(dataset=dataset, defect_list=[defect[0] for defect in broken_defect_list]) # doubledefects is not needed because dataset["species"] has been already rewritten properly
    # Create list of jumps in perfect crystal
    logger.info("Analyzing jump mechanisms in unstrained crystal...")
    su_jump_list = dataset_to_jumps(dataset=dataset, crystal=crystal_undef, symop_list=symop_undef, all_defect_list=all_defect_list, species_list=species_list, sym_unique=True, component_list=component_list)
    # Apply the same algorithm as broken_defect_list to find the inclusive list of broken jumps
    logger.info("Searching for jump broken symmetries under strain...")
    broken_jump_list = []  # List of symmetry unique jumps in the deformed crystal
    broken_jump_name_list = []  # List oj jump names to avoid doubles
    for jump in su_jump_list:
        broken_jump_list.append(jump)
        broken_jump_name_list.append(jump.get_name())
        tmp_jump_list = [jump]
        for j, jumpsym in enumerate(jump.find_symeqs(symops=symop_undef, print=False, all_defect_list=all_defect_list)):
            tmp_jump_list.append(jumpsym)
            new_jump_symeq_list = jumpsym.find_symeqs(symops=symop_def, print=False, all_defect_list=all_defect_list)
            double = False
            for new in new_jump_symeq_list:
                for old in tmp_jump_list[:-1]:
                    if new.get_label() == old.get_label():
                        double = True
                        break
                if double:
                    break
            if not double:
                if not jumpsym.get_label() == jump.get_label():
                    tmp_name = jumpsym.get_name()
                    broken_jump_name_list.append(tmp_name)
                    n_doubles = broken_jump_name_list.count(tmp_name)
                    if n_doubles > 1:
                        jumpsym.set_name(tmp_name + "_" + str(n_doubles))
                    broken_jump_list.append(jumpsym)
                    # Search for corresponding rotation matrix
                    for isym in range(len(symop_undef)):
                        if jump.find_symeqs(symops=[symop_undef[isym]], print=False, all_defect_list=all_defect_list)[0].get_label() == jumpsym.get_label():
                            found_sym = isym
                            break
                    logger.info("  !!  Broken symmetry found for Jump {}".format(tmp_name) + " (symmetry matrix: " + "".join([str(s) for s in change_base(arr=symop_undef[found_sym].get_rotation(), crystal=crystal_undef, inv=True, matrix=True)]) + ")")

    del jump, tmp_jump_list, jumpsym, new_jump_symeq_list, double, new, old
    # Now we are done with processing, we just need to update the dataset and write the strained input file
    # Read the original input file (to copy and modify it in the new input file)
    if os.path.exists(input_file):
        input_string_original = open(input_file, "r").read()
    else:
        produce_error_and_quit("!! In produce_strained_input - Input file {} not found".format(input_file))
    # Remove #-comments from input file
    input_string_original = "\n".join([string.split(sep='#')[0] for string in input_string_original.split(sep='\n') if len(string.split(sep='#')[0]) > 0])
    # Update dataset and build up the new input file
    new_input_lines = []
    new_input_lines.append("# File for strained calculations automatically created with produce_strained_input on {}\n".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    new_input_lines.append(input_string_original.split("&")[0])  # print user comments on top of input file
    d = sym.Symbol('d')
    for ipt_item in input_string_original.split("&")[1:]:
        # New list of defects from broken_defect_list (dataset[uniquepos] was already updated above)
        if "uniquepos" in ipt_item.split()[0].lower():
            dataset["uniquepos"] = dataset["uniquepos"][:1]
            new_input_lines.append("& UNIQUEPOS {}\n".format(len(broken_defect_list)))
            for idx, defect in enumerate(broken_defect_list):  # !! Each item in broken_defect_list is a list of 2 items: [0] defect; [1] Original defect index
                pos = [str(a.evalf(4)).replace(" ","").replace(".0*d","*d") for a in d*(defect[0].get_sublattice()-defect[0].get_sublattice_nod())/db_displ+defect[0].get_sublattice_nod()]
                new_input_lines.append("s {} {} {} \n".format(*pos))
                dataset["uniquepos"].append("s")
                [dataset["uniquepos"].append(str(a)) for a in pos]
        # Update list of species with permissions related to the new extended defect list broken_defect_list
        elif "species" == ipt_item.split()[0].lower():
            new_input_lines.append("& SPECIES {}\n".format(n_species))
            for i in range(0, n_species):
                new_input_lines.append(" ".join([s for s in dataset["species"][1+i*(2+n_broken_defects):3+n_broken_defects+i*(2+n_broken_defects)]]) + "\n")
        # New list of jumps from broken_jump_list
        elif "jumpmech" == ipt_item.split()[0].lower():  # print new list of jumps
            new_input_lines.append("& JUMPMECH\n")
            tmp_jumpmech_dataset = []  # the new dataset entry for jumpmech is re-written in this list
            for jump in broken_jump_list:
                tmp_jumpmech_dataset += ( ["%%"] + [str(len(jump.get_constraints()))] + [jump.get_name()] )
                new_input_lines.append("%% {} {}\n".format(len(jump.get_constraints()), jump.get_name()))
                for constr in jump.get_constraints():
                    posini = [str(a.evalf(4)).replace(" ","").replace(".0*d","*d") for a in d*(constr.get_iniposition()-constr.get_iniposition_nod())/db_displ + constr.get_iniposition_nod()]
                    posfin = [str(a.evalf(4)).replace(" ","").replace(".0*d","*d") for a in d*(constr.get_finposition()-constr.get_finposition_nod())/db_displ + constr.get_finposition_nod()]
                    tmp_jumpmech_dataset += (["s", str(constr.get_inispecies().get_index())] + posini
                                            +[ ">", str(constr.get_finspecies().get_index())] + posfin)
                    new_input_lines.append("s {:.0f} {} {} {} > {:.0f} {} {} {}".format(constr.get_inispecies().get_index(),*posini, constr.get_finspecies().get_index(), *posfin) +
                                           " # o {:.0f} {:.4f} {:.4f} {:.4f} > {:.0f} {:.4f} {:.4f} {:.4f}\n".format(constr.get_inispecies().get_index(), *change_base(arr=constr.get_iniposition(), crystal=crystal_undef, inv=True),
                                                                                             constr.get_finspecies().get_index(), *change_base(arr=constr.get_finposition(), crystal=crystal_undef, inv=True)))
            dataset['jumpmech'] = list(tmp_jumpmech_dataset)
        # Transform basis atoms in supercell coordinates
        elif "basis" in ipt_item.split()[0].lower():
            if dataset['basis'][0] == 's': # don't do anything if basis is already in supercell coordinates
                new_input_lines.append("&"+ipt_item)
            else:
                dataset["basis"][0] = "s"
                n_basis_atoms = int(dataset['basis'][1])
                new_input_lines.append("& BASIS s {:.0f}\n".format(n_basis_atoms))
                for i in range(n_basis_atoms):
                    tmp_pos = change_base(arr=np.array(dataset['basis'][3+i*4:6+i*4], dtype=float), crystal=crystal_undef)
                    dataset["basis"][3+i*4:6+i*4] = [str(pos) for pos in tmp_pos]
                    new_input_lines.append("{:.0f} ".format(dataset["basis"][2+i*4])+" ".join(["{}".format(pos) for pos in tmp_pos]) + "\n")
        # Transform iniconf in supercell coordinates
        elif "iniconf" in ipt_item.split()[0].lower():
            n_iniconf_comp = int(dataset["iniconf"][0])
            new_input_lines.append("& INICONF {}\n".format(n_iniconf_comp))
            for i in range(0, n_iniconf_comp):
                tmp_pos_nod = np.array([evalinputd0(e) for e in dataset["iniconf"][3 + i * 5:6 + i * 5]], dtype=float)
                tmp_pos_onlyd = np.array([evalinput(e)-evalinputd0(e) for e in dataset["iniconf"][3+i*5:6+i*5]], dtype=float)
                if dataset["iniconf"][1+i*5] == "o":  # change coordinates to supercell basis and update dataset
                    tmp_pos_nod = change_base(arr=tmp_pos_nod, crystal=crystal_undef)
                    tmp_pos_onlyd = change_base(arr=tmp_pos_onlyd, crystal=crystal_undef)
                    dataset["iniconf"][1+i*5:6+i*5] = ["s", dataset["iniconf"][2+i*5]] + [str(pos.evalf(4)).replace(" ","").replace(".0*d","*d")  for pos in (tmp_pos_nod + d*tmp_pos_onlyd/db_displ)]
                new_input_lines.append("s {} ".format(dataset["iniconf"][2+i*5]) + " ".join([str(pos.evalf(4)).replace(" ","").replace(".0*d","*d")  for pos in (tmp_pos_nod + d*tmp_pos_onlyd/db_displ)]) + "\n")
        # Transform basis into orthonormal coordinates
        elif "cpg" in ipt_item.split()[0].lower():
            if dataset["cpg"][0] == "o":
                new_input_lines.append("&" + ipt_item)  # don't do anything, keep it orthonormal
            else:  # change it from supercell (unstrained) to orthonormal coordinates
                tmp_cpg = change_base(arr=np.array(dataset["cpg"][1:4], dtype=float), crystal=crystal_undef, inv=True)
                new_input_lines.append("& CPG o {} {} {}\n".format(*tmp_cpg))
                dataset["cpg"][0:4] = ["o"] + [str(a) for a in tmp_cpg]
        elif "normal" in ipt_item.split()[0].lower():
            if dataset["normal"][0] == "o":
                new_input_lines.append("&" + ipt_item)  # don't do anything, keep it orthonormal
            else:  # change it from supercell (unstrained) to orthonormal coordinates
                tmp_cpg = change_base(arr=np.array(dataset["normal"][1:4], dtype=float), crystal=crystal_undef, inv=True)
                new_input_lines.append("& NORMAL o {} {} {}\n".format(*tmp_cpg))
                dataset["normal"][0:4] = ["o"] + [str(a) for a in tmp_cpg]
        else:  # all other lines of the file are re-printed unchanged
            new_input_lines.append("&" + ipt_item)
    del d
    # Write the new input file
    if ".txt" in input_file:
        output_name = directory + input_file.split("/")[-1].replace(".txt","") + "_strained.txt"
    else:
        output_name = directory + input_file.split("/")[-1] + "_strained.txt"
    with open(output_name, "w") as new_input:
        new_input.writelines(new_input_lines)
    logger.info(' ------------------------ Done applying strain -------------------------\n')
    return dataset


def proj(vect: np.ndarray, crystal: Crystal, direc: list) -> list:
    vect = change_base(arr=vect, crystal=crystal, inv=True) # vector in the orthonormal base
    return [np.dot(e, vect) for e in direc]


def rotate_matrix(matrix: np.ndarray, symop: SymOp) -> np.ndarray:
    """Apply a symmetry rotation to a matrix (for instance, for elastic dipole symmetries)"""
    R = symop.get_rotation()
    Rinv = np.linalg.inv(R)
    return np.dot(np.dot(R, matrix), Rinv)


def set_confname(defects: list, translations: list, perms: list, species: list, disso: bool=False, thermoint: int=-1) -> str:
    """Uses defects and translations lists to create a configuration name,
    accounting for permutations when there are several components of a given species.
    If bulkspecies_list is not void, the bulkspecies defects are modified so that the first component is always positive."""
    # defects is a list of defect objects
    # trans must be a list of np.arrays with 3 components each
    # disso is set to true if configuration is dissociated
    # Work on a copy of the "defects" list (to make sure we are not modifying the elements somewhere else in the code)
    defect_list = list(defects)
    if not disso:
        defect_list = apply_bulkspecies(defects=defect_list, translations=translations, species=species)
    # Now, analyze defects and translations to define the confname
    name_list = []
    swap_dict = {}
    for order in perms:
        tmp_def = [defect_list[i] for i in order]
        tmp_trans = flat_list([list(translations[i]-translations[order[0]]) for i in order])
        if disso:
            name = 'd' + ''.join(str(e) + '_' for e in [f.get_index() for f in tmp_def])
        else:
            name = 'c' + ''.join(str(e) + '_' for e in [f.get_index() for f in tmp_def])
        name += ''.join('p' + str(int(round(e))) for e in tmp_trans[3:])
        name = name.replace('p-', 'm')
        if disso:
            name += '$'+str(thermoint)
        name_list.append(name)
        swap_dict[name] = [tmp_def, tmp_trans]
    name_list.sort()
    return [name_list[0], swap_dict[name_list[0]][0], swap_dict[name_list[0]][1]]


def set_interaction_classes(conf: Configuration, thermo_index: float, kinetic_index1: float, symops: list, symops_cpg: list, all_defect_list: list, name_perms: list) -> list:
    """Analyzes conf (Configuration objects) to find symmetrically equivalent configurations
       in terms of kinetic interaction classes and thermodynamic interaction classes.
       Creates all these configuration in a dictionary"""
    if conf.get_thermoint() is not None:
        produce_error_and_quit("In set_interaction_classes - Configuration {} is not new.. this is odd.".format(conf.get_label()))
    thermo_index += 1  # increasing thermodynamic interaction index
    kinetic_index1 += 1  # increasing kinetic interaction index
    threp_list = []  # list of thermodynamic interactions instances (configurations).
    kirep_list = []  # list of kinetic interactions instances (configurations). We will search for one equation for each item in this list
    null_kirep_list = []  # list of kinetic interactions with zero contribution
    config_list = {}  # dictionary of all configurations
    # Look for all symmetry equivalent configurations (stored in symeqs)
    symeqs = apply_configuration_symmetry_operations(config=conf, symops=symops, ncpgops=len(symops_cpg), all_defect_list=all_defect_list, create=True, name_perms=name_perms)
    # symeqs[0] = kinetic symmetry equivalents (cpg_symeqconf)
    # symeqs[1] = thermodynamic symmetry equivalents (symeqconf)
    # symeqs[2] = kinetic interaction sign (sign_cpg)
    # symeqs[3] = configuration object list (created_confs)
    # symeqs[4] = list of symmetry operation indices (uniquesymops)
    it = -1
    # Store all kinetic symmetry equivalents, and assign them the same thermodynamic interaction label and same kinetic one
    for cidx in range(0, len(symeqs[0])):
        config_list[symeqs[0][cidx]]=symeqs[3][cidx]  # appending a new configuration to the dictionary
        it += 1
        config_list[symeqs[0][cidx]].set_thermoint(thermo_index)  # assign configuration to the same thermodynamic interaction class
        config_list[symeqs[0][cidx]].set_kinint1(symeqs[2][it]*kinetic_index1)  # assign configuration to the same kinetic interaction class
    # Store all thermodynamic symmetry equivalents (that are not kinetically equivalent), and assign them the same thermo int. label
    for cidx in range(0, len(symeqs[1])):
        config_list[symeqs[1][cidx]] = symeqs[3][cidx+len(symeqs[0])]  # appending a new configuration to the dictionary
        config_list[symeqs[1][cidx]].set_thermoint(thermo_index)  # assign configuration to the same thermodynamic interaction class
    # Append all configurations (both kinetic and thermodynamic equivalents) to threp_list: [representative config, [list of all configs]]
    threp_list.append([config_list[symeqs[0][0]], [config_list[e] for e in symeqs[0]+symeqs[1]]])
    # Check for null interactions (symmetric interaction with respect to cpg)
    if symeqs[2][0] != 0: # if interaction is not null
        kirep_list.append([config_list[symeqs[0][0]], len(symeqs[0])])  # kirep_list contains: [representative config, multiplicity]
        threp_list[-1][0].set_kineticinter(kinetic_index1)  # this appends the kinetic index to the list of kinetic indices for this configuration
    else:  # zero -> append to list of null kinetic interactions
        null_kirep_list.append([config_list[symeqs[0][0]], len(symeqs[0])])  # [representative config, multiplicity]
        kinetic_index1 -= 1  # decrease kinetic interaction index
    # Keep only list containing thermodynamic equivalents that are not kinetically equivalents
    symeqs = symeqs[1]
    # Assign these thermodynamic equivalents to their kinetic interaction class
    while len(symeqs) > 0:
        # Apply cpg symmetry operations
        new_symeqs = apply_configuration_symmetry_operations(config=config_list[symeqs[0]], symops=symops_cpg, all_defect_list=all_defect_list, ncpgops=len(symops_cpg), name_perms=name_perms)
        if new_symeqs[2][0] != 0:  # if interaction is not null
            kinetic_index1 += 1
            kirep_list.append([config_list[symeqs[0]], len(new_symeqs[0])])
            threp_list[-1][0].set_kineticinter(kinetic_index1)  # this appends the kinetic index to the list of kinetic indices for this configuration
        else:  # null kinetic interaction
            null_kirep_list.append([config_list[symeqs[0]], len(new_symeqs[0])])
        # Assign each configuration to the same kinetic interaction class
        it = -1
        for eq in new_symeqs[0]:
            it += 1
            config_list[eq].set_kinint1(new_symeqs[2][it]*kinetic_index1)  # assign kinetic interaction class
            # Remove configuration from symeqs list, because it has already been treated
            for idx in range(0, len(symeqs)):
                if eq == symeqs[idx]:
                    del symeqs[idx]
                    break
    # Return a list of lists: configurations (dictionary), thermodynamic and kinetic instances lists, updated thermodynamic and kinetic indices
    return [config_list, threp_list, kirep_list, thermo_index, kinetic_index1, null_kirep_list]


def sensitivity_study(KL: list, L, dL0: list, dKL: list, dT: list, temp: int, derS: list, Sterms: list, W: list, dir: str, uniqueJF: list, directions: list):
    """performs the sensitivity study on transport coefficients to identify the most relevant jump frequencies"""
    logger.info("         Performing sensitivity analysis...")
    gradL = np.zeros((len(directions), W.shape[0], KL[0].shape[0], KL[0].shape[0]), dtype=float) #initializing gradient of transport coefficient in jump frequency space
    NgradL = np.zeros((len(directions), KL[0].shape[0], KL[0].shape[0]), dtype=float)
    mainfreq = np.zeros((len(directions), W.shape[0], KL[0].shape[0], KL[0].shape[0]), dtype=float)
    Vnum = L(KL[0].transpose()) # n-body interaction values
    if len(Sterms) != 0: # correction terms for site interactions if needed
        dtau = []
        dgam = []
        for wi in range(W.shape[0]):
            dtau.append(derS[wi][2]-derS[wi][0].transpose()*Sterms[3]+Sterms[3].transpose()*(dT[wi]*Sterms[3]-derS[wi][0]))
            dgam.append([derS[wi][1][d]-dKL[wi][d]*Sterms[3]+KL[d]*L(dT[wi]*Sterms[3]-derS[wi][0]) for d in directions])
    for d in directions: # loop on directions
        for wi in range(W.shape[0]): # loop on jump frequencies
            gradL[d, wi] = dL0[wi][d] + KL[d]*L(dKL[wi][0].transpose() - dT[wi]*Vnum) + dKL[wi][d]*Vnum
            if len(Sterms) != 0:
                gradL[d, wi] += dgam[wi][d]*Sterms[2] + Sterms[0][d]*np.linalg.solve(a=Sterms[1], b=dgam[wi][0].transpose()-dtau[wi]*Sterms[2])

        for sp1 in range(KL[0].shape[0]):
            for sp2 in range(KL[0].shape[0]):
                NgradL[d, sp1, sp2] = np.linalg.norm(gradL[d, :, sp1, sp2])
                for wi in range(W.shape[0]):  # loop on jump frequencies
                    mainfreq[d, wi, sp1, sp2] = gradL[d, wi, sp1, sp2] / NgradL[d, sp1, sp2]

    with open(dir+'SENSITIVITY/IF_{}.dat'.format(temp), 'w') as output: # writing results to file IF=important frequencies
        k = 1
        output.writelines('# 1:W') # writing comment line
        for d in directions:
            for sp1 in range(KL[0].shape[0]):
                for sp2 in range(sp1, KL[0].shape[0]):
                    k +=1
                    output.writelines(' {:3.0f}:L{:.0f}{:.0f}_{:.0f}'.format(k, sp1, sp2,d))
        output.writelines('\n')
        for wi in range(0, W.shape[0]):
            output.writelines('{:5.0f}'.format(uniqueJF[wi]))
            for d in directions:
                for sp1 in range(0, KL[0].shape[0]):
                    for sp2 in  range(sp1, KL[0].shape[0]):
                        output.writelines('{:10.4f}'.format(mainfreq[d,wi,sp1,sp2]))
            output.writelines('\n')

def solve_prec(Tnum, Klambda, Ninter, Nspec, Ndir, Pref, L0, Site: list = []):
    Klambda = [mp.matrix(Klambda[direc]) for direc in range(Ndir)]
    Vnum = mp.matrix(Ninter, Nspec)  # initialization at zero
    L = mp.cholesky(A=-Tnum)  #lower triangular matrix
    Vnum = subs_forwardbackward(L=L, X=Vnum, B=Klambda[0].T)
    if len(Site)==3: # Site contains Dmap; Dlambda; DissContrib
        Site[0] = mp.matrix(Site[0])
        Site[2] = mp.matrix(Site[2])
        Site[1] = [mp.matrix(x) for x in Site[1]]
        TD = mp.matrix(Ninter, Site[0].cols)  # initialization at zero
        TD = subs_forwardbackward(L=L, X=TD, B=Site[0])
        tau = Site[2]-Site[0].T * TD
        if np.linalg.matrix_rank(np.array(tau.tolist(), dtype=float)) == tau.rows: # matrix is invertible
            gamma = [Site[1][d] - Klambda[d] * TD for d in range(Ndir)]
            delta = subs_forwardbackward(L=mp.cholesky(A=-tau), X=mp.matrix(tau.rows, Nspec), B=gamma[0].T)
            Scorr = [gamma[direc] * delta for direc in range(Ndir)]
        else:
            Scorr = [mp.matrix(Nspec) for _ in range(Ndir)]
    else:
        Scorr = [mp.matrix(Nspec) for _ in range(Ndir)]
    return [np.array((Pref*(mp.matrix(L0[direc]) + mp.matrix(Klambda[direc])*Vnum + 0*Scorr[direc])).tolist(), dtype=float) for direc in range(Ndir)]

def solve_def(Tnum, Klambda, Ndir, Nspec, Pref, L0, Site: list = [], sensi: bool = False):
    Sterms = [] # numerical values for gamma, tau, delta and T*D, eventually needed for sensitivity study
    Klambda = np.array(Klambda, dtype=float)
    L = linalg.factorized(Tnum) # LU factorization; can be turned to Cholesky with the CHOLMOD package
    Vnum = L(Klambda[0].transpose())
    #Vnum = linalg.spsolve(A=Tnum, b=np.array(KLfunc(strain, Wnum), dtype=float)[0].transpose()) # old solve not saving the factorization
    if len(Site)==3: # Site contains Dmap; Dlambda; DissContrib
        TD = L(Site[0])
        tau = Site[2]-np.dot(Site[0].transpose(), TD)
        if np.linalg.matrix_rank(tau) == tau.shape[0]: # matrix is invertible
            gamma = [Site[1][direc]-np.dot(Klambda[direc], TD) for direc in range(Ndir)]
            delta = np.linalg.solve(a=tau, b=gamma[0].transpose())
            Scorr = [np.dot(gamma[direc], delta) for direc in range(Ndir)]
            if sensi:
                Sterms = [gamma, tau, delta, TD]
        else:
            Scorr = [np.zeros((Nspec, Nspec)) for _ in range(Ndir)]
    else:
        Scorr = [np.zeros((Nspec, Nspec)) for _ in range(Ndir)]
    #logger.info("Vnum = {}".format(delta))
    return [float(Pref)*(np.array(L0[direc], dtype=float) + np.dot(Klambda[direc], Vnum) + Scorr[direc]) for direc in range(Ndir)], L, Sterms

def subs_forwardbackward(L, X, B):
    # performs the forward and backward subsitution to solve LL.TX=B
    # X must be initialized at zero with proper dimensions
    # we are in fact solving for -B because T is negative def pos
    y = mp.matrix(X.rows, X.cols)  # initialization at zero
    for i in range(X.rows):  # forward substitution
        y[i, :] = -(B[i, :] + L[i, :] * y) / L[i, i]
    for i in range(X.rows - 1, -1, -1):  # backward substitution
        X[i, :] = (y[i, :] - L[:, i].T * X) / L[i, i]
    return X

def trans_and_check_unicity(veclist: list, symop_idx_list: list, perms: list, species_list: list, dbdist: float,
                            all_defect_list: list, bulkspecies_list: dict, bulkspecies_symdefects: dict):
    # Translating each defects such that the first one of each configuration is in the "000" supercell
    for idx, vectors in enumerate(veclist):
        veclist[idx] = np.array(vectors)
        # Apply bulkspecies if needed
        if len(bulkspecies_list) > 0:
            defects = []
            translations = []
            for c in veclist[idx]:
                [d, t] = vect2deftrans(vec=c, all_defect_list=all_defect_list)
                defects.append(d)
                translations.append(t)
            for d, sp in enumerate(species_list):
                if bulkspecies_list.get(sp) is not None:  # this species should be applied the bulkspecies feature
                    distances = [distance(vect1=veclist[idx][d], vect2=vect2, crystal=defects[0].get_crystal()) for
                                 vidx, vect2 in enumerate(veclist[idx]) if vidx != d if species_list[vidx].get_name() != 'bulk']
                    if np.all(distances > dbdist):  # this dumbbel is isolated from the others
                        defects[d] = bulkspecies_symdefects[defects[d]]
            for d, t in enumerate(translations):
                veclist[idx][d] = t + defects[d].get_sublattice()

    # Check unicity
    newveclist = []
    symop_index = []
    for sidx, vectors in zip(symop_idx_list, veclist):
        found = False
        for foundvectors in newveclist:
            for order in perms:
                t = [foundvectors[0] - vectors[order][0]]*len(vectors)
                if np.all([abs(t[0][k]-round(t[0][k]))<tol for k in range(0,3)]): # translations between similar sublattice
                    if are_equal_arrays(A=foundvectors, B=vectors[order]+t):
                        found = True
                        break
            if found:
                break
        if not found:
            newveclist.append(vectors[perms[0]])
            symop_index.append(sidx)
    return newveclist, symop_index


def untranslatedfinal(jumpmech, component_list: list, iniconf, finconf, name_perms: list, fnum=0, init=False):
    ini_position = np.array([iniconf.get_defect_position(def_idx=idx) for idx in range(len(component_list))], dtype=float)
    if (not init) and (iniconf.get_beyond() or finconf.get_beyond()):
        return ini_position, np.array([finconf.get_defect_position(def_idx=idx) for idx in range(len(component_list))], dtype=float)

    jumpvec = []
    specs = [k.get_species().get_index() for k in component_list]
    for cons in jumpmech.get_constraints():
        if cons.get_inispecies().get_index() != 0:
            jumpvec.append([cons.get_inispecies().get_index(), cons.get_finposition() - cons.get_iniposition()])
            for i, s in enumerate(specs):
                if s == cons.get_inispecies().get_index():
                    del specs[i]
                    break
    for s in specs:
        jumpvec.append([s, np.array([0., 0., 0.])])
    jumpvec = sorted(jumpvec, key=lambda x:x[0])

    ljumpvec = [[jumpvec[a] for a in b] for b in name_perms]
    ldiff = [np.array([finconf.get_defect_position(a) for a in b])-ini_position for b in name_perms]
    for diff in ldiff:
        for j in ljumpvec:
            tall = diff - np.array([a[1] for a in j])
            for k in range(len(tall) - 1, -1, -1):
                tall[k] -= tall[0]
            if sum(sum(np.abs(tall))) < 1e-8:
                return ini_position, ini_position+np.array([a[1] for a in j])
    produce_error_and_quit("Could not find the untranslated final position for jump frequency {}".format(fnum))


def vect2deftrans(vec: np.ndarray, all_defect_list: list) -> list:
    """ Takes a vector and transforms it into a defect index and an integer translation vector, written as a 4-item list."""
    trans = np.floor(vec + tol_pbc)
    vec2 = vec - trans  # do not modify vec, otherwise it will mess up the values in the upper-level functions
    found = False
    for defect in all_defect_list:
        if are_equal_arrays(defect.get_sublattice(), vec2):
            return [defect, trans]
    if not found:
        produce_error_and_quit("In vect2deftrans, problem in identifying defect for vector {}. Please check coordinates in the input file.".format(vec))

