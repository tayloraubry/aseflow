#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:15:27 2022

@author: taubry
"""

import os, sys
import argparse
import glob
import pandas as pd


def check_converged(filename):
    LM = False
    IM = False
    EM = False

    with open(filename, encoding='latin-1') as f:
        for line in reversed(f.readlines()):
            if 'LatticeMinimize: Converged' in line:
                LM = True
            elif 'IonicMinimize: Converged' in line:
                IM = True
            elif 'ElecMinimize: Converged' in line:
                EM = True
                break

    return pd.Series({'LM': LM, 'IM': IM, 'EM': EM})


def get_energies(filename):
    G = None
    F = None
    E = None

    with open(filename, encoding='latin-1') as f:
        for line in reversed(f.readlines()):
            if ('G =' in line) and (len(line) > 4) and (G is None):
                G = float(line.rstrip().split()[-1])
            elif 'F =' in line and F is None:
                F = float(line.rstrip().split()[-1])
            elif 'Etot =' in line and E is None:
                E = float(line.rstrip().split()[-1])
                break

    return pd.Series({'E': E, 'F': F, 'G': G})


def get_nelectrons(filename):
    nE = None

    with open(filename, encoding='latin-1') as f:
        for line in reversed(f.readlines()):
            if 'nElectrons' in line:
                splitline = line.split()
                i = splitline.index('nElectrons:') + 1
                nE = float(splitline[i])
                break

    return pd.Series({'nE': nE})


def get_vib_components(filename):
    ZPE = None
    Evib = None
    TSvib = None
    Avib = None

    with open(filename, encoding='latin-1') as f:
        for line in reversed(f.readlines()):
            if 'Avib:' in line and Avib is None:
                Avib = float(line.rstrip().split()[-1])
            elif 'TSvib:' in line and TSvib is None:
                TSvib = float(line.rstrip().split()[-1])
            elif 'Evib:' in line and Evib is None:
                Evib = float(line.rstrip().split()[-1])
            elif 'ZPE' in line and ZPE is None:
                ZPE = float(line.rstrip().split()[-1])
                break

    return pd.Series({'ZPE': ZPE, 'Evib': Evib, 'TSvib': TSvib, 'Avib': Avib})


def get_ionic_steps(filename):
    nsteps = None

    with open(filename, encoding='latin-1') as f:
        for line in reversed(f.readlines()):
            if 'IonicMinimize: Iter:' in line:
                parts = line.split()
                nsteps = int(parts[parts.index('Iter:') + 1])
                break

    return pd.Series({'IonicSteps': nsteps})


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compile energies from selected directories'
    )
    parser.add_argument(
        '-n', default='out', type=str,
        help='Structure name to include, may use *'
    )
    parser.add_argument(
        '-d', default=[], type=str, nargs='+',
        help='Directories to include, may use *'
    )
    parser.add_argument(
        '-sds', default='', help='Subdirectory structure, if any'
    )
    parser.add_argument(
        '-vibfile', default=None, type=str,
        help='Filename for vibrations.out'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    filename = args.n
    dirs = args.d
    subdir = args.sds
    vibfile = args.vibfile

    topdir = os.getcwd()
    series_list = []

    for directory in dirs:
        os.chdir(directory + '/' + str(subdir) + '/')

        name = pd.Series({'Calculation': directory})
        converged = check_converged(filename)
        energies = get_energies(filename)
        nelec = get_nelectrons(filename)
        ionic_steps = get_ionic_steps(filename)

        if vibfile is not None:
            vib_components = get_vib_components(vibfile)
            series = pd.concat(
                [name, converged, energies, vib_components, nelec, ionic_steps],
                axis=0
            )
        else:
            series = pd.concat(
                [name, converged, energies, nelec, ionic_steps],
                axis=0
            )

        os.chdir(topdir)
        series_list.append(series)

    df = pd.concat(series_list, axis=1).T
    df.to_csv('energies.csv', index=False)

    with pd.option_context('display.max_colwidth', 20,
                           'display.precision', 8):
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
