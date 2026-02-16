'''Parallelized version of MD-PMM calculations using multiprocessing'''

from pymm.inputs import read_pmm_inputs, write_geom
import itertools
import logging
import sys
from timeit import default_timer as timer
from argparse import ArgumentParser, FileType
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd
import numpy as np
from numba import njit, prange
from scipy import linalg
import pymm.conversions as conv
from pymm.conversions import Bohr2Ang
import multiprocessing as mp
from functools import partial

# Import all the helper functions from the original pmm module
from pymm.pmm import (
    convert2Universe, bynum2indexes, split_qc_solv, cut_qc,
    order_mass, match_qc, calc_el_field_pot_atom, calc_el_field_pot_qc,
    rotate_dip_matrix, calc_pmm_matrix
)


def process_frame_chunk(start_idx, end_idx, traj_path, top_path, mm_indexes, 
                       qc_ref_geom, qc_energies, qc_dip_matrix, qc_charges,
                       qc_charge, qc_ch_switch, pre_diag_flag):
    """
    Process a chunk of frames from the trajectory.
    
    Parameters:
        start_idx (int): Starting frame index
        end_idx (int): Ending frame index (exclusive)
        traj_path (str): Path to trajectory file
        top_path (str): Path to topology file
        mm_indexes (str): QC atom indexes in MM trajectory
        qc_ref_geom (np.ndarray): Reference QC geometry
        qc_energies (np.ndarray): QC electronic state energies
        qc_dip_matrix (np.ndarray): Electric dipole matrix
        qc_charges (np.ndarray): QC atomic charges
        qc_charge (int): Total QC charge
        qc_ch_switch (bool): Whether to use atomic charges
        pre_diag_flag (bool): Whether to calculate pre-diagonalized elements
        
    Returns:
        tuple: (eigvals, eigvecs, pre_diags) for this chunk
    """
    # Load trajectory for this worker
    mm_traj = mda.Universe(top_path, traj_path)
    
    # Split into QC and solvent
    qc_traj, solv_traj = split_qc_solv(mm_traj, mm_indexes)
    
    # Create reference Universe
    qc_ref = convert2Universe(qc_ref_geom)
    qc_ref.atoms.positions -= qc_ref.atoms.center_of_mass()
    
    # Calculate number of frames in this chunk
    n_frames = end_idx - start_idx
    n_states = qc_energies.shape[0]
    
    # Initialize arrays for this chunk
    eigvals = np.zeros((n_frames, n_states))
    eigvecs = np.zeros((n_frames, n_states, n_states))
    
    if pre_diag_flag:
        pre_diags = np.zeros_like(eigvals)
    else:
        pre_diags = None
    
    # Process frames in this chunk
    for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
        # Move to the specific frame
        mm_traj.trajectory[global_idx]
        
        # Calculate electric field and potential
        if qc_ch_switch:
            el_field, potential = calc_el_field_pot_atom(
                solv_traj.atoms.positions,
                solv_traj.atoms.charges,
                qc_traj.atoms.positions,
                qc_traj.atoms.center_of_mass(),
                qc_charges
            )
        else:
            el_field, potential = calc_el_field_pot_qc(
                solv_traj.atoms.positions,
                solv_traj.atoms.charges,
                qc_traj.atoms.positions,
                qc_traj.atoms.center_of_mass(),
                q_tot=qc_charge
            )
        
        # Rotate dipole matrix
        rot_dip_matrix, rot_matrix = rotate_dip_matrix(
            qc_dip_matrix, qc_traj, qc_ref
        )
        
        # Calculate PMM matrix
        pmm_matrix = calc_pmm_matrix(
            qc_energies, rot_dip_matrix, el_field, potential, qc_ch_switch
        )
        
        # Diagonalize
        eigval, eigvec = linalg.eigh(pmm_matrix)
        
        eigvals[local_idx, :] = eigval
        eigvecs[local_idx, :, :] = eigvec
        
        if pre_diag_flag:
            pre_diags[local_idx, :] = pmm_matrix.diagonal()
    
    return eigvals, eigvecs, pre_diags


def pmm_parallel(cmdline, n_workers=4):
    """
    Main parallelized PMM calculation function.
    
    Parameters:
        cmdline: Command line arguments (same as original pmm function)
        n_workers (int): Number of parallel workers to use
        
    Returns:
        int: 0 on success
    """
    start = timer()
    
    # Set up logging (same as original)
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    
    logging.info('\n'
                 '==========================================================================\n'
                 '|                                                                        |\n'
                 '|                          PyMM (Parallel)                               |\n'
                 '|                                                                        |\n'
                 '|                 Perform MD-PMM calculations using                      |\n'
                 '|                     {} parallel workers                                 |\n'.format(n_workers) +
                 '|                                                                        |\n'
                 '==========================================================================\n')
    
    # Read inputs (same as original)
    qc, mm_traj = read_pmm_inputs(cmdline)
    
    # Apply geometry unit conversions
    if cmdline.geom_units.lower() == 'bohr':
        qc.geom[:, 1:] *= Bohr2Ang
    elif cmdline.geom_units.lower() == 'nm':
        qc.geom[:, 1:] *= 10
    
    # Check if charges are provided
    qc_ch_switch = isinstance(cmdline.charges, str)
    
    # Handle QM indexes if provided
    if cmdline.qm_indexes:
        new_indexes = bynum2indexes(cmdline.qm_indexes)
        qc.geom = qc.geom[new_indexes, :]
        
        if qc_ch_switch:
            tot_indexes = [i for i in range(qc.charges.shape[1])]
            comp_indexes = [i for i in tot_indexes if i not in new_indexes]
            exc_charge = qc.charges[:,comp_indexes].sum(axis=1)
            qc.charges = qc.charges[:, new_indexes]
            qc.charges = qc.charges + (exc_charge/qc.geom.shape[0])[:,None]
    
    qc_ref = convert2Universe(qc.geom)
    qc_ref.atoms.positions -= qc_ref.atoms.center_of_mass()
    
    # Split trajectory
    qc_traj, solv_traj = split_qc_solv(mm_traj, cmdline.mm_indexes)
    
    # Match QC atoms order if requested
    if cmdline.match:
        logging.info(' * Reordering reference QC geometry to match MD simulation...\n')
        new_indices = match_qc(order_mass(qc_traj, qc_ref), qc_traj, qc_ref)
        qc_ref.atoms.masses = qc_ref.atoms.masses[new_indices]
        qc_ref.atoms.positions = qc_ref.atoms.positions[new_indices,:]
        qc.geom = qc.geom[new_indices,:]
        if qc_ch_switch:
            qc.charges = qc.charges[:,new_indices]
    
    # Log system information
    n_qc_atoms = qc.geom.shape[0]
    n_qc_states = qc.energies.shape[0]
    n_frames = mm_traj.trajectory.n_frames
    
    logging.info('\n'
                 '==========================================================================\n'
                 '|                          System Information                            |\n'
                 '==========================================================================\n')
    logging.info(f' * Number of QC atoms: {n_qc_atoms}')
    logging.info(f' * Number of electronic states: {n_qc_states}')
    logging.info(f' * Total frames in trajectory: {n_frames}')
    logging.info(f' * Number of parallel workers: {n_workers}')
    logging.info(f' * Frames per worker: ~{n_frames // n_workers}\n')
    
    # Determine frame chunks for each worker
    frames_per_worker = n_frames // n_workers
    frame_chunks = []
    
    for i in range(n_workers):
        start_idx = i * frames_per_worker
        if i == n_workers - 1:
            # Last worker takes any remaining frames
            end_idx = n_frames
        else:
            end_idx = (i + 1) * frames_per_worker
        frame_chunks.append((start_idx, end_idx))
    
    logging.info(' * Frame distribution across workers:')
    for i, (start, end) in enumerate(frame_chunks):
        logging.info(f'   Worker {i+1}: frames {start} to {end-1} ({end-start} frames)')
    logging.info('\n')
    
    logging.info('==========================================================================\n'
                 '|                      Starting Parallel Calculation                     |\n'
                 '==========================================================================\n')
    
    # Create partial function with fixed parameters
    process_func = partial(
        process_frame_chunk,
        traj_path=cmdline.trajectory_path,
        top_path=cmdline.topology_path,
        mm_indexes=cmdline.mm_indexes,
        qc_ref_geom=qc.geom,
        qc_energies=qc.energies,
        qc_dip_matrix=qc.dip_matrix,
        qc_charges=qc.charges if qc_ch_switch else None,
        qc_charge=cmdline.qc_charge,
        qc_ch_switch=qc_ch_switch,
        pre_diag_flag=cmdline.pre_diag
    )
    
    calc_start = timer()
    
    # Process chunks in parallel
    with mp.Pool(processes=n_workers) as pool:
        results = pool.starmap(process_func, [(start, end) for start, end in frame_chunks])
    
    calc_end = timer()
    
    logging.info(f' * Parallel calculation completed in {calc_end - calc_start:.2f} seconds\n')
    
    # Combine results from all workers
    logging.info(' * Combining results from all workers...\n')
    
    eigvals_list = [r[0] for r in results]
    eigvecs_list = [r[1] for r in results]
    
    eigvals = np.vstack(eigvals_list)
    eigvecs = np.vstack(eigvecs_list)
    
    if cmdline.pre_diag:
        pre_diags_list = [r[2] for r in results]
        pre_diags = np.vstack(pre_diags_list)
    
    # Save results
    np.savetxt('{}_eigvals.dat'.format(cmdline.output), eigvals,
               header='Perturbed QC energies:')
    np.save('{}_eigvecs'.format(cmdline.output), eigvecs)
    
    if cmdline.pre_diag:
        np.savetxt('{}_pre_diags.dat'.format(cmdline.output), pre_diags,
                   header='Perturbed Hamiltonian matrix diagonal elements:')
    
    xyz_fn = '{}_qc_geom.xyz'.format(cmdline.output)
    write_geom(xyz_fn, qc.geom)
    
    end = timer()
    
    logging.info('==========================================================================\n'
                 '|                               OUTPUT                                   |\n'
                 '==========================================================================\n'
                 '\n'
                 ' * Results have been saved in:\n' +
                 ' ** Perturbed energies: {}_eigvals.dat\n'.format(cmdline.output) +
                 ' ** Eigenvectors: {}_eigvecs.npy\n'.format(cmdline.output) +
                 ' ** QC reference geometry: {}\n'.format(xyz_fn) +
                 '\n'
                 ' * Total execution time: {:.2f} seconds\n'.format(end - start) +
                 ' * Calculation time: {:.2f} seconds\n'.format(calc_end - calc_start) +
                 '\n'
                 '==========================================================================\n')
    
    return 0


if __name__ == '__main__':
    # Example of how to use this in your main script
    print("This is a module to be imported. See pymm_parallel_main.py for usage.")
