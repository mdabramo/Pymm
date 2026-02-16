"""
PyMM with Parallel Processing Support

This is a modified version of pymm.py that supports parallel processing of MD trajectories.
Usage:
    python pymm_parallel_main.py run_pmm [options] --n-workers 4
    
The --n-workers option controls how many parallel processes to use.
"""

from pymm.conversions import au2eV
from pymm.eigvecs import get_ext, eig_corr, eig_corr_tot, eig_hist
from pymm.inputs import read_raw_matrix
from pymm.pmm import pmm
from pmm_parallel import pmm_parallel
from pymm.spectra import calc_pert_matrix, calc_abs
from pymm.free_en import calc_dA, calc_dA_mean
from datetime import datetime
import getpass
import logging
import sys
from timeit import default_timer as timer
from argparse import ArgumentParser, FileType
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def main():
    '''Program with CLI to perform MD-PMM calculations with parallel support.'''
    parser = ArgumentParser(prog='PyMM-Parallel',
                            description='Perform MD-PMM calculation with parallel processing.')
    subparsers = parser.add_subparsers(dest='command')

    # Perform MD-PMM calculation.
    parser_pmm = subparsers.add_parser('run_pmm',
                                       help='Perform MD-PMM calculation.')
    parser_pmm.add_argument('-g', '--ref-geom', action='store', type=str,
                        help='QC reference (QM) geometry filename')
    parser_pmm.add_argument('-gu', '--geom-units', choices=['angstrom', 'bohr',
                        'nm'], help='specify units used in the reference (default: angstrom)'
                        'geometry', default='angstrom')
    parser_pmm.add_argument('-dm', '--dip-matrix', action='store', type=str,
                        help='QC unperturbed electric dipole moment matrix '
                        'filename')
    parser_pmm.add_argument('-e', '--energies', action='store', type=str,
                        help='QC unperturbed electronic states energies '
                        'filename')
    parser_pmm.add_argument('-ch', '--charges', action='store', type=str,
                        default=False, help='file with the QC atomic '
                        'charges for each unperturbed electronic state '
                        '(default=False)')
    parser_pmm.add_argument('-traj', '--trajectory-path', action='store',
                        type=str, help='XTC file of the MD simulation '
                        'trajectory')
    parser_pmm.add_argument('-top', '--topology-path', action='store', type=str,
                        help='TPR file of the MD simulation')
    parser_pmm.add_argument('-q', '--qc-charge', action='store', default=0,
                        type=int, help='total QC charge (default=0)')
    parser_pmm.add_argument('-nm', '--mm-indexes', action='store', type=str,
                        help='indexes of the QC in the MM trajectory')
    parser_pmm.add_argument('-nq', '--qm-indexes', action='store', type=str,
                        default=False, help='indexes of the portion of the '
                        'reference (QM) geometry to be considered in the '
                        'MD-PMM calculation')
    parser_pmm.add_argument('--match', action='store_true',
                            help='Reorder the QC reference geometry '
                            'to match the atoms order in the MD simulation')
    parser_pmm.add_argument('-o', '--output', action='store', type=str,
                        default='pymm', help='job name used to name the '
                        'output files (default: pymm)')
    parser_pmm.add_argument('--pre-diag', action='store_true',
                            help='Request the perturbed Hamiltonian matrix '
                            'diagonal elements')
    parser_pmm.add_argument('-n', '--n-workers', action='store', type=int,
                        default=1, help='Number of parallel workers to use '
                        '(default=1, no parallelization). Recommended: 4-8')

    # Calculate absorption spectra - keep same as original
    parser_abs = subparsers.add_parser('calc_abs',
                                       help='Calculate absorption spectra')
    parser_abs.add_argument('-dm', '--dip-matrix', action='store', type=str,
                            help='QC unperturbed electric dipole moment matrix '
                            'filename')
    parser_abs.add_argument('-el', '--eigvals', action='store', type=str,
                            default='pymm_eigvals.dat', help='Perturbed '
                            'eigenvalues trajectory (default: pymm_eigvals.dat)')
    parser_abs.add_argument('-ev', '--eigvecs', action='store', type=str,
                            default='pymm_eigvecs.npy', help='Perturbed '
                            'eigenvectors trajectory (default: pymm_eigvecs.npy)')
    parser_abs.add_argument('-sigma', action='store', type=float,
                            default=0.034, help='Square root of the variance of the gaussian '
                            'broadening applied to the signal (expressed in eV). Default: 0.034 eV.')
    parser_abs.add_argument('-xtr', '--extra-range', action='store', type=float,
                            default=0.005, help='Additional value to add to the'
                            'frequency range considered for the spectrum '
                            'calculation')
    parser_abs.add_argument('-ot', '--output', action='store', type=str,
                            default='abs_spectrum',
                            help='Calculated absorption spectra names '
                            '(default: abs_spectrum)')

    # Calculate free energy - keep same as original
    parser_dA = subparsers.add_parser('free_en',
                                     help='Calculate free energy')
    parser_dA.add_argument('-T', '--temperature', action='store', type=float,
                           default=298., help='Temperature of the system '
                           'during the simulation')
    parser_dA.add_argument('-eii', '--en_in_in', action='store', type=str,
                           help='Filename of the MD-PMM energies trajectory '
                           'for the initial state in the initial ensemble')
    parser_dA.add_argument('-efi', '--en_fin_in', action='store', type=str,
                           help='Filename of the MD-PMM energies trajectory '
                           'for the final state in the initial ensemble')
    parser_dA.add_argument('-eif', '--en_in_fin', action='store', type=str,
                           help='Filename of the MD-PMM energies trajectory '
                           'for the initial state in the final ensemble')
    parser_dA.add_argument('-eff', '--en_fin_fin', action='store', type=str,
                           help='Filename of the MD-PMM energies trajectory '
                           'for the final state in the final ensemble')
    parser_dA.add_argument('-col', action='store', type=int, default=0,
                           help='Column (or electronic state) to consider '
                           'from the energies trajectory file')
    parser_dA.add_argument('-o', '--output', action='store', type=str,
                            default='dA_mean.dat',
                            help='Calculated free energy differences'
                            '(default: dA_mean.dat)')

    # Eigenvectors analysis - keep same as original
    parser_eig = subparsers.add_parser('eig', help='Analyse the perturbed '
                                       'eigenvectors')
    parser_eig.add_argument('-i', '--input', action='store', type=str,
                            default='pymm_eigvecs.npy',
                            help='Perturbed eigenvectors trajectory provided '
                            'as an npy file')
    parser_eig.add_argument('-first', action='store', type=int, default=0,
                            help='First unperturbed state to consider')
    parser_eig.add_argument('-last', action='store', type=int,
                            help='Last unperturbed state to consider.')
    parser_eig.add_argument('-state', action='store', type=int,
                            help='Perturbed state to consider.')
    parser_eig.add_argument('-dpi', action='store', type=int, default=300,
                            help='Dpi of the saved image (default=300).')
    parser_eig.add_argument('-bins', action='store', type=int, default=20,
                            help='Number of bins for histograms (default=20).')
    parser_eig.add_argument('-oc', '--output_corr', action='store', type=str,
                            const='eig_corr.png', nargs='?',
                            help='Calculate correlation plot')
    parser_eig.add_argument('-ot', '--output_tot', action='store', type=str,
                            const='eig_corr_tot.png', nargs='?',
                            help='Calculate total correlation plot')
    parser_eig.add_argument('-oh', '--output_hist', action='store', type=str,
                            const='eig_hist.png', nargs='?',
                            help='Calculate histogram')

    args = parser.parse_args()

    if args.command == 'run_pmm':
        start = timer()
        
        # Choose parallel or serial version based on n_workers
        if args.n_workers > 1:
            logging.info(f'\nUsing PARALLEL mode with {args.n_workers} workers\n')
            pmm_parallel(args, n_workers=args.n_workers)
        else:
            logging.info('\nUsing SERIAL mode (single worker)\n')
            pmm(args)
        
        end = timer()
        
        logging.info(f'\n ** Total wall time: {end - start:.2f} seconds\n')

    elif args.command == 'calc_abs':
        # Same as original - absorption spectrum calculation
        start = timer()
        
        log_format = '%(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                          format=log_format)
        
        logging.info('\n'
                     '==========================================================================\n'
                     '|                                                                        |\n'
                     '|                     Absorption Spectra Calculation                     |\n'
                     '|                                                                        |\n'
                     '==========================================================================\n')

        dip_matrix_unp = read_raw_matrix(args.dip_matrix)
        eigvals = np.loadtxt(args.eigvals)
        eigvecs = np.load(args.eigvecs)

        logging.info(' * Number of frames = {}'.format(eigvals.shape[0]))
        logging.info(' * Number of electronic states = {}\n'.format(eigvals.shape[1]))

        logging.info(' * Sigma = {} eV\n'.format(args.sigma))

        dip_matrix_per = calc_pert_matrix(eigvecs, dip_matrix_unp)
        freq_min, freq_max, freq_abs_mean = calc_abs(eigvals, dip_matrix_per,
                                                      args.sigma, args.extra_range)

        out_fn = '{}.dat'.format(args.output)
        out = np.zeros((freq_abs_mean.shape[0], 2))
        out[:, 0] = np.linspace(freq_min, freq_max, freq_abs_mean.shape[0])
        out[:, 1] = freq_abs_mean

        np.savetxt(out_fn, out, header='Frequency (eV)   Absorption (a.u.)')

        logging.info('==========================================================================\n'
                     '|                               OUTPUT                                   |\n'
                     '==========================================================================\n'
                     '\n'
                     f' * Calculated absorption spectra saved in: {out_fn}\n'
                     '\n'
                     '==========================================================================\n')

        end = timer()
        logging.info(f'\n ** Total execution time: {end - start:.2f} seconds\n')

    elif args.command == 'free_en':
        # Same as original - free energy calculation
        start = timer()
        
        log_format = '%(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                          format=log_format)

        logging.info('==========================================================================\n'
                     '|                                                                        |\n'
                     '|                     Free Energy Calculation                            |\n'
                     '|                                                                        |\n'
                     '==========================================================================\n')

        col = args.col

        if (args.en_in_in and args.en_fin_in) and (args.en_in_fin and args.en_fin_fin):
            logging.info(' * Free energy calculation using both ensembles.\n')
            
            en_in_in = np.loadtxt(args.en_in_in)[:,col]
            en_fin_in = np.loadtxt(args.en_fin_in)[:,col]
            en_in_fin = np.loadtxt(args.en_in_fin)[:,col]
            en_fin_fin = np.loadtxt(args.en_fin_fin)[:,col]

            if en_in_in.shape[0] == en_in_fin.shape[0] == en_fin_in.shape[0] == en_fin_fin.shape[0]:
                logging.info(f' * Number of frames = {en_in_in.shape[0]}\n')
                dA = calc_dA_mean(args.temperature, en_in_in, en_fin_in,
                                  en_in_fin, en_fin_fin)
            else:
                logging.error(' ! Trajectories have different lengths\n')
                raise IOError

        elif (args.en_in_in and args.en_fin_in) and not args.en_in_fin and not args.en_fin_fin:
            logging.info(' * Free energy using initial ensemble only.\n')
            logging.warning(' ! Consider using both ensembles for better accuracy.\n')
            
            en_in_in = np.loadtxt(args.en_in_in)[:,col]
            en_fin_in = np.loadtxt(args.en_fin_in)[:,col]

            if en_in_in.shape[0] == en_fin_in.shape[0]:
                logging.info(f' * Number of frames = {en_in_in.shape[0]}\n')
                dA = calc_dA(args.temperature, en_in_in, en_fin_in, state='ini')
            else:
                logging.error(' ! Trajectories have different lengths\n')
                raise IOError

        else:
            raise IOError('Input files provided incorrectly.')

        logging.info('==========================================================================\n'
                     '|                              RESULTS                                   |\n'
                     '==========================================================================\n'
                     f'\n * Free Energy Change: {dA} J/mol\n'
                     '\n'
                     '==========================================================================\n')

        end = timer()
        logging.info(f'\n ** Total execution time: {end - start:.2f} seconds\n')

    elif args.command == 'eig':
        # Same as original - eigenvector analysis
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                          format='%(message)s')

        logging.info('==========================================================================\n'
                     '|                     Eigenvectors Analysis                              |\n'
                     '==========================================================================\n')

        eigvecs = np.load(args.input)

        logging.info(f' * Number of frames = {eigvecs.shape[0]}')
        logging.info(f' * Number of electronic states = {eigvecs.shape[1]}\n')

        plt.rcParams['axes.linewidth'] = 1.3
        plt.rcParams['font.size'] = 18
        plt.rcParams['legend.fontsize'] = 15

        if args.output_corr is not None:
            logging.info(f' * Generating correlation plot...\n')
            eig_corr(eigvecs, args.first, args.last, args.state,
                     args.output_corr, args.dpi, args.bins)

        if args.output_tot is not None:
            logging.info(f' * Generating total correlation plot...\n')
            eig_corr_tot(eigvecs, args.state, args.output_tot,
                         args.dpi, args.bins)

        if args.output_hist is not None:
            logging.info(f' * Generating histogram...\n')
            eig_hist(eigvecs, args.state, args.output_hist, args.dpi)

        logging.info('\n'
                     '==========================================================================\n'
                     '|                             FINISHED                                   |\n'
                     '==========================================================================\n')

    return 0


if __name__ == '__main__':
    sys.exit(main())
