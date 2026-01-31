#!/bin/bash
#
# run the tas resolution simulation
# @author Tobias Weber <tweber@ill.fr>
# @date march-2021
# @license GPLv3
#

# number of neutrons
num_neutrons=1e8

# compile and run the simulation
run_compilers=1
run_simulations=1
use_mpi=1

# number of processes
hw_processes=$(nproc)
let num_processes=${hw_processes}/2+1

# tools
MCSTAS_COMP=$(which mcstas)
MCSTAS_RUN=$(which mcrun)
MPI_COMP=$(which mpicc)
MPI_RUN=$(which mpirun)

INSTR_FILE="tas.instr"
C_FILE="${INSTR_FILE%.instr}.c"
BIN_FILE="${INSTR_FILE%.instr}.bin"
OUT_DIR="results"


# use default number of processes if automatic determination failed
if [ "${num_processes}" = "" ]; then
	num_processes=4
fi


# check if the compilers could be found
if [ "${MCSTAS_COMP}" = "" ] || [ "${MPI_COMP}" = "" ] || [ "${MPI_RUN}" = "" ]; then
	echo -e "McStas or MPI compilers could not be found."
	exit -1
fi


# compile the simulation
if [ "$run_compilers" != 0 ]; then
	# cleanup previous files
	rm -fv ${C_FILE}
	rm -fv ${BIN_FILE}

	echo -e "\n================================================================================"
	echo -e "Compiling ${INSTR_FILE} -> ${C_FILE}."
	echo -e "================================================================================"
	if ! ${MCSTAS_COMP} --verbose -o ${C_FILE} ${INSTR_FILE}; then
		echo -e "Failed compiling instrument file."
		exit -1
	fi

	echo -e "\n================================================================================"
	echo -e "Compiling ${C_FILE} -> ${BIN_FILE}."
	echo -e "================================================================================"
	if ! ${MPI_COMP} -march=native -O2 -time -DUSE_MPI -o ${BIN_FILE} ${C_FILE} -lm; then
		echo -e "Failed compiling C file."
		exit -1
	fi
fi


# run the simulation
if [ "$run_simulations" != 0 ]; then
	# cleanup output directory
	rm -rfv ${OUT_DIR}

	echo -e "\n================================================================================"
	echo -e "Running simulation ${BIN_FILE}, directory: ${OUT_DIR}, number of processes: ${num_processes}."
	echo -e "================================================================================"

	if [ "$use_mpi" != 0 ]; then
		if ! ${MPI_RUN} --use-hwthread-cpus -v -np ${num_processes} ${BIN_FILE} \
			--ncount=${num_neutrons} --format=McStas --dir=${OUT_DIR} \
			src_lam=4.5
		then
			echo -e "Simulation failed."
			exit -1
		fi
	else
		# non-mpi call:
		if ! ${MCSTAS_RUN} --ncount=${num_neutrons} --dir=${OUT_DIR} ${INSTR_FILE}
		then
			echo -e "Simulation failed."
			exit -1
		fi
	fi

	# show results
	python3 ../calc_cov.py "${OUT_DIR}/reso.dat"
	#mcresplot "${OUT_DIR}/reso.dat"
fi
