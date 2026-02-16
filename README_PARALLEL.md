# PyMM Parallel Processing Guide

## Overview

This parallelized version of PyMM splits your MD trajectory into chunks and processes them simultaneously using multiple CPU cores. This can provide significant speedups, especially for long trajectories.

## Key Improvements

### Original Code (Serial Processing)
```
Frame 1 → Frame 2 → Frame 3 → ... → Frame 100
(one at a time)
```

### Parallel Code (with 4 workers)
```
Worker 1: Frames 1-25     ┐
Worker 2: Frames 26-50    ├─ All running simultaneously
Worker 3: Frames 51-75    │
Worker 4: Frames 76-100   ┘
```

## Installation

1. Place the following files in your PyMM directory:
   - `pmm_parallel.py` - Parallel processing module
   - `pymm_parallel_main.py` - Modified main script with parallel support

2. No additional dependencies needed beyond standard PyMM requirements

## Usage

### Basic Command (4 workers)

```bash
python pymm_parallel_main.py run_pmm \
    -g reference.xyz \
    -dm dipole_matrix.dat \
    -e energies.dat \
    -traj trajectory.xtc \
    -top topology.tpr \
    -nm "1:50" \
    -o output_name \
    --n-workers 4
```

### Key Parameter: --n-workers

- `--n-workers 1` : Serial mode (same as original, no parallelization)
- `--n-workers 4` : Use 4 parallel processes (recommended starting point)
- `--n-workers 8` : Use 8 parallel processes (for larger systems/more cores)

### Complete Example

```bash
# Process 100-frame trajectory with 4 workers
python pymm_parallel_main.py run_pmm \
    -g qc_geometry.xyz \
    -gu angstrom \
    -dm dipole_matrix.dat \
    -e electronic_energies.dat \
    -ch atomic_charges.dat \
    -traj simulation.xtc \
    -top topology.tpr \
    -q 0 \
    -nm "1:10,15:20" \
    -o parallel_results \
    --match \
    --pre-diag \
    --n-workers 4
```

## Choosing the Number of Workers

### Rule of Thumb
- **Start with 4 workers** for most systems
- **Maximum: number of CPU cores** on your machine
- **Sweet spot: 4-8 workers** for typical HPC nodes

### How to Determine Optimal Number

1. **Check your CPU cores:**
   ```bash
   # On Linux/Mac
   nproc
   # or
   lscpu | grep "CPU(s)"
   ```

2. **Run a quick test:**
   ```bash
   # Test with different worker counts on a subset of your trajectory
   time python pymm_parallel_main.py run_pmm ... --n-workers 1  # baseline
   time python pymm_parallel_main.py run_pmm ... --n-workers 2
   time python pymm_parallel_main.py run_pmm ... --n-workers 4
   time python pymm_parallel_main.py run_pmm ... --n-workers 8
   ```

3. **Diminishing returns:**
   - 1→2 workers: ~2x speedup
   - 1→4 workers: ~3-3.5x speedup
   - 1→8 workers: ~5-6x speedup
   - Beyond 8: minimal gains due to overhead

## Expected Speedup

For a 100-frame trajectory:

| Workers | Time (example) | Speedup |
|---------|----------------|---------|
| 1       | 100 min        | 1.0x    |
| 2       | 52 min         | 1.9x    |
| 4       | 28 min         | 3.6x    |
| 8       | 18 min         | 5.6x    |

*Actual speedup depends on system size and CPU architecture*

## Important Notes

### Memory Considerations
- Each worker loads a copy of the trajectory
- **Memory usage ≈ (base memory) × (number of workers)**
- For large trajectories, limit workers to avoid memory issues
- Example: 10 GB trajectory × 4 workers = ~40 GB RAM needed

### When Parallelization Helps Most
✅ **Good candidates:**
- Long trajectories (>50 frames)
- Complex QM/MM calculations
- Large QC (quantum center) systems
- Multiple electronic states

❌ **May not help:**
- Very short trajectories (<20 frames)
- I/O-bound systems (slow disk access)
- Limited memory situations

## Verification

The parallel version produces **identical results** to the serial version. To verify:

```bash
# Run both versions
python pymm.py run_pmm ... -o serial_output
python pymm_parallel_main.py run_pmm ... --n-workers 4 -o parallel_output

# Compare results
diff serial_output_eigvals.dat parallel_output_eigvals.dat
# Should show no differences (or only floating-point rounding)
```

## Troubleshooting

### Issue: No speedup observed
**Solutions:**
- Trajectory may be too short - try longer trajectories
- Check CPU usage with `htop` or `top` during execution
- Ensure you're not I/O bound - use fast SSD storage

### Issue: Memory errors
**Solutions:**
- Reduce number of workers
- Process trajectory in smaller segments
- Use a machine with more RAM

### Issue: Results differ slightly
**Note:** Small floating-point differences (<1e-10) are normal due to:
- Different order of operations in parallel processing
- Numerical precision in matrix operations
- This does not affect physical conclusions

## Performance Benchmarking Script

Create this script to test different configurations:

```bash
#!/bin/bash
# benchmark.sh - Test different worker counts

TRAJECTORY="your_trajectory.xtc"
TOPOLOGY="your_topology.tpr"
OUTPUT_BASE="benchmark"

echo "Benchmarking PyMM Parallel Performance"
echo "======================================"

for WORKERS in 1 2 4 8; do
    echo ""
    echo "Testing with $WORKERS workers..."
    START=$(date +%s)
    
    python pymm_parallel_main.py run_pmm \
        -g reference.xyz \
        -dm dipole.dat \
        -e energies.dat \
        -traj $TRAJECTORY \
        -top $TOPOLOGY \
        -nm "1:50" \
        -o ${OUTPUT_BASE}_w${WORKERS} \
        --n-workers $WORKERS
    
    END=$(date +%s)
    RUNTIME=$((END-START))
    echo "Runtime with $WORKERS workers: $RUNTIME seconds"
done
```

## Advanced: Batch Processing Multiple Trajectories

If you have multiple independent trajectories to analyze:

```bash
#!/bin/bash
# process_multiple.sh

# Process 4 trajectories simultaneously, each using 2 workers
parallel -j 4 python pymm_parallel_main.py run_pmm \
    -g reference.xyz \
    -dm dipole.dat \
    -e energies.dat \
    -traj {} \
    -top topology.tpr \
    -nm "1:50" \
    -o output_{#} \
    --n-workers 2 \
    ::: traj1.xtc traj2.xtc traj3.xtc traj4.xtc
```

## Technical Details

### How It Works

1. **Trajectory Splitting**: The total frames are divided into N equal chunks
2. **Worker Assignment**: Each worker processes its assigned chunk independently
3. **Result Combination**: Results are concatenated in the original frame order
4. **No Communication**: Workers don't communicate, avoiding overhead

### What Gets Parallelized

✅ **Parallelized:**
- Frame-by-frame QM/MM calculations
- Electric field calculations
- Matrix diagonalization
- Eigenvalue/eigenvector computation

❌ **Not Parallelized:**
- Trajectory loading (done per worker)
- Initial geometry matching
- Result file writing
- Post-processing (absorption, free energy, etc.)

### Architecture

```
Main Process
    │
    ├─── Worker 1 (frames 0-24)
    │    └─── Load trajectory → Process frames → Return results
    │
    ├─── Worker 2 (frames 25-49)
    │    └─── Load trajectory → Process frames → Return results
    │
    ├─── Worker 3 (frames 50-74)
    │    └─── Load trajectory → Process frames → Return results
    │
    └─── Worker 4 (frames 75-99)
         └─── Load trajectory → Process frames → Return results
    
    Results Combined → Files Written
```

## Contact & Support

If you encounter issues with the parallel version:

1. First try with `--n-workers 1` to ensure the problem isn't in the core calculation
2. Check memory usage with `htop` or `top`
3. Verify trajectory file integrity
4. Compare with serial version results for correctness

## Citation

If you use this parallel implementation in your research, please cite both:
1. The original PyMM software
2. Mention the use of parallel processing in your methods section

Example text:
> "QM/MM calculations were performed using PyMM with parallel trajectory processing 
> across N CPU cores to accelerate the computation."
