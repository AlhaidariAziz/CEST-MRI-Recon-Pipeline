#!/bin/bash -l

#SBATCH --export=NONE

#SBATCH --ntasks=2
#SBATCH --gres=gpu:a100:1 # a100 GPU
##SBATCH --gres=gpu:1 # just any GPU
##SBATCH --gres=gpu:a100:1 -C a100_80 # supposed to be for a100 (80GB) GPU
#SBATCH -p a100
##SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=LLR_pars_lamda_12072026
#SBATCH --mail-type=END
#SBATCH --mail-user=ab.alhaidari@yahoo.com

unset SLURM_EXPORT_ENV

module unload python
# module load cuda/12.4.1
module load python/3.12-conda
# conda init bash
eval "$(conda shell.bash hook)"
conda activate LLRv3
module load cuda/12.4.1

DATA_DIR='/home/hpc/iwbi/iwbi112h/LLR codebooks/lab_llr_alhaidari/scripts/'

mkdir $TMPDIR/$SLURM_JOB_ID


free -h 

echo "Running on $(hostname)"
echo "Start time: $(date)"

##Original Work

# python "${DATA_DIR}mps.py" --space ksp # space = Hyb || ksp

# python "${DATA_DIR}HybVolCS.py" --i 30 --r 1e-4
# # echo "End time: $(date)"


# =======================================================================
# Cartesian  LLRs variations  
# =======================================================================

# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.15
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.08
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 30  --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.001
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 30  --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.0001
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 10 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 50 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 70 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 100 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 1 1
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 3 3
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 7 7
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 9 9
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 11 11
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 13 13
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 15 15
# echo "End time: $(date)"

# =======================================================================
# Cartesian (y-shift)  LLRs variations (Regs + blk size +splits) 
# =======================================================================


# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_3x2_yshift_True.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 5 5
# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_3x2_yshift_True.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.001 --blk_shape 1 5 5
# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_3x2_yshift_True.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.0001 --blk_shape 1 5 5
# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_3x2_yshift_True.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.00001 --blk_shape 1 5 5
# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_3x2_yshift_True.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.000001 --blk_shape 1 5 5
# echo "End time: $(date)"

# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_3x2_yshift_True.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.000001 --blk_shape 1 1 1
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_3x2_yshift_True.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.000001 --blk_shape 1 3 3
# echo "End time: $(date)"

# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_3x2_yshift_True.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.000001 --blk_shape 1 3 3 --splits=4
# echo "End time: $(date)"

# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_3x2_yshift_True.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.08 --blk_shape 1 5 5 --splits=8
# echo "End time: $(date)"

# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_3x2_yshift_True.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.15 --blk_shape 1 5 5 --splits=8
# echo "End time: $(date)"





# =======================================================================
# =======================================================================
# Cartesian  sense 
# =======================================================================

# python "${DATA_DIR}HybVolSENSE.py" --i 30  --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.001
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 30  --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_3x2_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.001
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 30  --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_3x3_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.001
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 30  --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_4x3_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.001
# echo "End time: $(date)"


# =======================================================================
# Cartesian  sense  variations
# =======================================================================



# python "${DATA_DIR}HybVolSENSE.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.15
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.08
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 30  --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.001
# # echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 30  --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_3x2_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.001
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 30  --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_4x2_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.001
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 30  --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_3x3_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.001
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 30  --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_4x3_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.001
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 30  --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.0001
# echo "End time: $(date)"

# python "${DATA_DIR}HybVolSENSE.py" --i 10 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 50 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 70 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 100 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_Cart_3x1_yshift_False.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 
# echo "End time: $(date)"

#CAIP Ssense Recons 
# python "${DATA_DIR}HybVolSENSE.py" --i 30 --r 0.001 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_APT_C6/kdat_2D_R26_C58_C6.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R26_C58_C6.h5"
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 30 --r 0.001 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_APT_C12/kdat_2D_R26_C58_C12.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R26_C58_C12.h5"
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolSENSE.py" --i 30 --r 0.001 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_APT_C16/kdat_2D_R26_C58_C16.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R26_C58_C16.h5"
# echo "End time: $(date)"


# python "${DATA_DIR}HybVolLLR.py" --i 30 --r 1e-4
# echo "End time: $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 30 --r 1e-3
# echo "End time (HybVolLLR.py --i 30 --r 1e-3): $(date)"
# python "${DATA_DIR}HybVolLLR.py" --i 30 --r 1e-1
# echo "End time (HybVolLLR.py --i 30 --r 1e-1): $(date)"
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 1 1 --i 30
# echo "End time (HybVolLLR.py --blk_shape 1 1 1 --i 30): $(date)"
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 3 3 --i 30
# echo "End time (HybVolLLR.py --blk_shape 1 3 3 --i 30): $(date)"
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 7 7 --i 30
# echo "End time (HybVolLLR.py --blk_shape 1 7 7 --i 30): $(date)"

# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 6 6 --i 50
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 1 1 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 2 2 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 4 4 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 5 5 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 6 6 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 8 8 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 10 10 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolLLR.py" --blk_shape 1 12 12 --i 30 --r 1e-2
# python "${DATA_DIR}HybVolGRAPPA.py" 
# python "${DATA_DIR}mdGRAPPA.py"
# python "${DATA_DIR}ifft_RO.py"
# python "${DATA_DIR}HybVolLLR.py" --i 30 --blk_shape 1 5 5 --r 1e-3
# python "${DATA_DIR}HybVolLLR.py" --i 30 --blk_shape 1 8 8



# ==========================================================
# DC-SYS LLR fine Tuneing
# ==========================================================


    # ++++++++++++++++++++++++++++++++
    # # Blk-size
    # ++++++++++++++++++++++++++++++++

# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_5x4_sym_yshift_True_dc_16.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 1 1 --splits=2
# echo "End time: $(date)"

# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_5x4_sym_yshift_True_dc_16.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 3 3 --splits=2
# echo "End time: $(date)"

# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_5x4_sym_yshift_True_dc_16.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 7 7 --splits=2
# echo "End time: $(date)"

# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_5x4_sym_yshift_True_dc_16.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 9 9 --splits=2
# echo "End time: $(date)"

# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_5x4_sym_yshift_True_dc_16.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 11 11 --splits=2
# echo "End time: $(date)"

# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_5x4_sym_yshift_True_dc_16.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 13 13 --splits=2
# echo "End time: $(date)"

# python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_5x4_sym_yshift_True_dc_16.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.01 --blk_shape 1 15 15 --splits=2
# echo "End time: $(date)"

    # ++++++++++++++++++++++++++++++++
    # # lamda
    # ++++++++++++++++++++++++++++++++

python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_5x4_sym_yshift_True_dc_16.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.05 --blk_shape 1 5 5 --splits=2
echo "End time: $(date)"

python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_5x4_sym_yshift_True_dc_16.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.03 --blk_shape 1 5 5 --splits=2
echo "End time: $(date)"

python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_5x4_sym_yshift_True_dc_16.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.009 --blk_shape 1 5 5 --splits=2
echo "End time: $(date)"

python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_5x4_sym_yshift_True_dc_16.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.007 --blk_shape 1 5 5 --splits=2
echo "End time: $(date)"

python "${DATA_DIR}HybVolLLR.py" --i 30 --data "/home/vault/iwbi/iwbi112h/CEST_DATA_06052026/CEST_GRAPPA3_3Shot/kdat_2D_R34_C52_3Shot_us_CAIP_5x4_sym_yshift_True_dc_16.h5" --mps "mps_c_0.9_t0.05_w_36_kw_6_sp_ksp_3D_R34_C52_3Shot.h5" --r 0.005 --blk_shape 1 5 5 --splits=2
echo "End time: $(date)"


rm -r $TMPDIR/$SLURM_JOB_ID




echo "End time (script): $(date)"
