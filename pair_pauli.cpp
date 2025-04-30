/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "pair_pauli.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

const double PairPauli::CC=4.0;

/* ---------------------------------------------------------------------- */

PairPauli::PairPauli(LAMMPS *lmp) : Pair(lmp)
{
	respa_enable = 0;
	//born_matrix_enable = 1;
	born_matrix_enable = 0;
	writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairPauli::~PairPauli()
{
	if (copymode) return;

	if (allocated) {
		memory->destroy(setflag);
		memory->destroy(cutsq);
		memory->destroy(cut);
		memory->destroy(z);
		memory->destroy(r);
		memory->destroy(zz);
		memory->destroy(rr);
	}
}

/* ---------------------------------------------------------------------- */

void PairPauli::compute(int eflag, int vflag)
{
	//energy
	ev_init(eflag, vflag);
	//atom properties - global
	double **x = atom->x;
	double **f = atom->f;
	const int *type = atom->type;
	const int nlocal = atom->nlocal;
	const int newton_pair = force->newton_pair;
	//neighbors - ith atom
	const int inum = list->inum;
	const int* ilist = list->ilist;
	const int* numneigh = list->numneigh;
	int** firstneigh = list->firstneigh;
	
	//loop over all owned atoms
	for (int ii = 0; ii < inum; ++ii) {
		const int i = ilist[ii];
		const double xtmp = x[i][0];
		const double ytmp = x[i][1];
		const double ztmp = x[i][2];
		const int itype = type[i];
		const int* jlist = firstneigh[i];
		const int jnum = numneigh[i];
		//loop over all nearest neighbors
		for (int jj = 0; jj < jnum; ++jj) {
			int j = jlist[jj];
			j &= NEIGHMASK;
			//compute distance
			const double delx = xtmp - x[j][0];
			const double dely = ytmp - x[j][1];
			const double delz = ztmp - x[j][2];
			const double rsq = delx * delx + dely * dely + delz * delz;
			const int jtype = type[j];
			//compute energy/force
			if (rsq < cutsq[itype][jtype]) {
				//force constant
				const double dr=sqrt(rsq);
				const double fexp=exp(-CC*(dr-rr[itype][jtype])/rr[itype][jtype]);
				const double fpair = zz[itype][jtype]*fexp*CC/(rr[itype][jtype]*dr);
				//add to force
				f[i][0] += delx * fpair;
				f[i][1] += dely * fpair;
				f[i][2] += delz * fpair;
				if (newton_pair || j < nlocal) {
					f[j][0] -= delx * fpair;
					f[j][1] -= dely * fpair;
					f[j][2] -= delz * fpair;
				}
				//add to energy
				double evdwl = 0.0;
				if (eflag) evdwl=zz[itype][jtype]*fexp;
				if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
			}
		}
	}
	
	//compute virial
	if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairPauli::allocate()
{
	allocated = 1;
	const int n = atom->ntypes + 1;
	
	memory->create(setflag, n, n, "pair:setflag");
	for (int i = 1; i < n; ++i){
		for (int j = i; j < n; ++j){
			setflag[i][j] = 0;
		}
	}
	
	memory->create(cutsq, n, n, "pair:cutsq");
	memory->create(cut, n, n, "pair:cut");
	memory->create(z, n, n, "pair:z");
	memory->create(r, n, n, "pair:r");
	memory->create(zz, n, n, "pair:zz");
	memory->create(rr, n, n, "pair:rr");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairPauli::settings(int narg, char **arg)
{
	if (narg != 1) error->all(FLERR, "Illegal pair_style command");
	
	cut_global = utils::numeric(FLERR, arg[0], false, lmp);
	
	// reset cutoffs that have been explicitly set
	if (allocated) {
		const int ntypes=atom->ntypes;
		for (int i = 1; i <= ntypes; ++i){
			for (int j = i; j <= ntypes; ++j){
				if (setflag[i][j]) cut[i][j] = cut_global;
			}
		}
	}
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairPauli::coeff(int narg, char **arg)
{
	if (narg < 4 || narg > 5) error->all(FLERR, "Incorrect args for pair coefficients");
	if (!allocated) allocate();

	int ilo, ihi, jlo, jhi;
	utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
	utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

	const double z_one = utils::numeric(FLERR, arg[2], false, lmp);
	const double r_one = utils::numeric(FLERR, arg[3], false, lmp);

	double cut_one = cut_global;
	if (narg == 5) cut_one = utils::numeric(FLERR, arg[4], false, lmp);

	int count = 0;
	for (int i = ilo; i <= ihi; ++i) {
		for (int j = MAX(jlo, i); j <= jhi; ++j) {
			z[i][j] = z_one;
			r[i][j] = r_one;
			cut[i][j] = cut_one;
			setflag[i][j] = 1;
			++count;
		}
	}

	if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairPauli::init_style()
{
	// request regular list
	int list_style = NeighConst::REQ_DEFAULT;
	neighbor->add_request(this, list_style);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairPauli::init_one(int i, int j)
{
	if (setflag[i][j] == 0) {
		zz[i][j]=sqrt(z[i][i]*z[j][j]);
		zz[j][i]=zz[i][j];
		rr[i][j]=0.5*(r[i][i]+r[j][j]);
		rr[j][i]=rr[i][j];
		cut[i][j]=sqrt(cut[i][i]*cut[j][j]);
		cut[j][i]=cut[i][j];
	}
	
	return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairPauli::write_restart(FILE *fp)
{
	write_restart_settings(fp);
	
	for (int i = 1; i <= atom->ntypes; ++i){
		for (int j = i; j <= atom->ntypes; ++j) {
			fwrite(&setflag[i][j], sizeof(int), 1, fp);
			if (setflag[i][j]) {
				fwrite(&z[i][j], sizeof(double), 1, fp);
				fwrite(&r[i][j], sizeof(double), 1, fp);
				fwrite(&cut[i][j], sizeof(double), 1, fp);
			}
		}
	}
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairPauli::read_restart(FILE *fp)
{
	read_restart_settings(fp);
	allocate();
	
	int me = comm->me;
	for (int i = 1; i <= atom->ntypes; ++i){
		for (int j = i; j <= atom->ntypes; ++j) {
			if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
			MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
			if (setflag[i][j]) {
				if (me == 0) {
					utils::sfread(FLERR, &z[i][j], sizeof(double), 1, fp, nullptr, error);
					utils::sfread(FLERR, &r[i][j], sizeof(double), 1, fp, nullptr, error);
					utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
				}
				MPI_Bcast(&z[i][j], 1, MPI_DOUBLE, 0, world);
				MPI_Bcast(&r[i][j], 1, MPI_DOUBLE, 0, world);
				MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
			}
		}
	}
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairPauli::write_restart_settings(FILE *fp)
{
	fwrite(&cut_global, sizeof(double), 1, fp);
	fwrite(&mix_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairPauli::read_restart_settings(FILE *fp)
{
	const int me = comm->me;
	if (me == 0) {
		utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
		utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
	}
	MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
	MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairPauli::write_data(FILE *fp)
{
	for (int i = 1; i <= atom->ntypes; ++i){
		fprintf(fp, "%d %g %g\n", i, z[i][i], r[i][i]);
	}
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairPauli::write_data_all(FILE *fp)
{
	for (int i = 1; i <= atom->ntypes; ++i){
		for (int j = i; j <= atom->ntypes; ++j){
			fprintf(fp, "%d %d %g %g %g\n", i, j, z[i][j], r[i][j], cut[i][j]);
		}
	}
}

/* ---------------------------------------------------------------------- */

double PairPauli::single(int i, int j, int itype, int jtype, double rsq, double factor_coul, double factor_lj, double &fforce)
{
	const double dr=sqrt(rsq);
	const double fexp=exp(-CC*(dr-rr[itype][jtype]));
	fforce=zz[itype][jtype]*fexp*CC/(rr[itype][jtype]*dr);
	return zz[itype][jtype]*fexp;
}

/* ---------------------------------------------------------------------- */

void PairPauli::born_matrix(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                            double /*factor_coul*/, double factor_lj, double &dupair,
                            double &du2pair)
{
	/*
	double rinv, r2inv, r6inv, du, du2;

	r2inv = 1.0 / rsq;
	rinv = sqrt(r2inv);
	r6inv = r2inv * r2inv * r2inv;

	// Reminder: lj1 = 48*e*s^12, lj2 = 24*e*s^6
	// so dupair = -forcelj/r = -fforce*r (forcelj from single method)

	du = r6inv * rinv * (lj2[itype][jtype] - lj1[itype][jtype] * r6inv);
	du2 = r6inv * r2inv * (13 * lj1[itype][jtype] * r6inv - 7 * lj2[itype][jtype]);

	dupair = factor_lj * du;
	du2pair = factor_lj * du2;
	*/
}

/* ---------------------------------------------------------------------- */

void *PairPauli::extract(const char *str, int &dim)
{
	dim = 2;
	if (strcmp(str, "z") == 0) return (void *) z;
	if (strcmp(str, "r") == 0) return (void *) r;
	return nullptr;
}
