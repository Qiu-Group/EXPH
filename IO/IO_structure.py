#!/usr/bin/env python3
# Author: Bowen Hou
# Date: 07/2022
# A general class for input file reading and writting
import numpy as np
import re

class POSCAR_IO():
    def __init__(self, style=False, filename=False, atom=False, lattice=False):
        """
        :param style: "qe", "vasp", "direct"
        :param vasp_input,qe_input:  POSCAR or pw_input
        :param atom: {"C": np.array([..,[a1,a2,a3],..])}
        :param lattice: np.array([[...],[x,y,z],[...]])
        """
        if style == "direct":
            self.atom = atom
            self.lattice = lattice
        elif style == "vasp" or style == "VASP":
            self.filename = filename
            self.atom, self.lattice = self.__get_poscar_vasp()
        elif style == "qe" or style == "QE":
            self.filename = filename
            self.atom, self.lattice = self.__get_poscar_qe()
        else:
            self.atom = ''
            self.lattice = ''

    def __get_poscar_qe(self):
        #todo: add function:  ansgrom2bohr / bohr2angstrom
        f = open(self.filename)
        lines = f.readlines()
        for line in lines:
            if "nat" in line:
                line_no_space = line.replace(' ','')
                index_temp = line_no_space.index('nat=')+4
                line_no_space_trimed = line_no_space[index_temp:]
                n_atom = int(re.findall('\d+',line_no_space_trimed)[0])
                break

        for index in range(len(lines)):
            if "CELL_PARAMETERS" in lines[index]:
                lattice = np.vstack(
                    (np.array(list(map(float, lines[index + 1].split()))),
                     np.array(list(map(float, lines[index + 2].split()))),
                     np.array(list(map(float, lines[index + 3].split()))),)
                )
                break

        atom_dic = {}
        for index in range(len(lines)):
            if "ATOMIC_POSITIONS" in lines[index]:
                for i in range(1, n_atom + 1):
                    line = lines[index + i]
                    atom_name = line.split()[0]
                    atom_position = list(map(float, line.split()[1:]))
                    if atom_name in atom_dic:
                        atom_dic[atom_name] += [atom_position]
                    else:
                        atom_dic[atom_name] = [atom_position]
        for i in atom_dic:
            atom_dic[i] = np.array(atom_dic[i])
        return atom_dic, lattice

    def __get_poscar_vasp(self):
        """
        :param poscar_name: POSCAR file (VASP taste)
        :return: atom_dic = {'atom1':[(atom1_1_x, atom_1_y, atom_1_z), (atom1_2_x, atom_2_y, atom_2_z), ...], ...}
                 lattice = [(a1,b1,c1),(a2,b2,c2),(a3,b3.c3)]
        """
        poscar_data_raw = open(self.filename, 'r')
        poscar_data = poscar_data_raw.readlines()    #processed poscar data
        atom_name = poscar_data[5].split()
        atom_number = poscar_data[6].split()
        [atom_information, position, atom_dic] = [[], [] , {}]
        lattice =  []
        for ia in range(3):
            temp5 = poscar_data[ia+2].split()
            temp6 = []
            for i in temp5:
                temp6 = temp6 + [float(i)]
            lattice = lattice + [tuple(temp6)]
        for i in range(len(atom_name)):   #[(atom1,number),...]
            temp = [(atom_name[i], int(atom_number[i]))]
            atom_information = atom_information + temp
        for ii in range(len(poscar_data)-8): #position = [(a1,b1,c1),...]
            temp1 = []
            temp = poscar_data[ii+8].split()
            for j in range(len(temp)):
                temp1 = temp1 + [float(temp[j])]
            position = position + [temp1]
        # position = np.array(position)
        for jj in range(len(atom_name)):
            [temp2, temp3, index] = [[], [], 0] #temp2 is name of atom; temp3 is corresponding section of such atom positions
            temp2 = atom_information[jj][0]
            for ij in range(jj+1):
                index = index + atom_information[ij][1]
            index0 = index - atom_information[jj][1]
            temp3 = position[index0: index]
            atom_dic[temp2] = np.array(temp3)
        lattice = np.array(lattice)
        return atom_dic, lattice

    def atom_num(self):
        """
        :return: number of each atom {"atom1":10...}
        """
        atom_num = {}
        for i in self.atom:
            atom_num[i] = self.atom[i].shape[0]
        return atom_num

    def atom_total_num(self):
        """
        :return: number of all atoms in this cell
        """
        count = 0
        for i in self.atom:
            count += self.atom[i].shape[0]
        return count

    def atom_species(self):
        return self.atom.keys()

    def write_poscar_vasp(self, output_file="POSCAR_new"):
        """
            :param atom_dic: {'atom1':[(atom1_1_x, atom_1_y, atom_1_z), (atom1_2_x, atom_2_y, atom_2_z), ...], ...}
            :param lattice: [(a1,b1,c1),(a2,b2,c2),(a3,b3.c3)]
            :return: write a new poscar file: POSCAR_new
            """
        with open(output_file, 'w') as f:
            f.write('generated by bw\n1.0000000000\n')
        with open(output_file, 'a') as f:
            for i in self.lattice:
                f.write('%.15f %.15f %.15f \n' % (i[0], i[1], i[2]))
            for i in self.atom:
                f.write('%s ' % i)
            f.write('\n')
            for i in self.atom:
                f.write('%s ' % (len(self.atom[i])))
            f.write('\nDirect\n')
            for i in self.atom:
                for ii in range(len(self.atom[i])):
                    f.write('%.15f %.15f %.15f \n' % (self.atom[i][ii][0], self.atom[i][ii][1], self.atom[i][ii][2]))
