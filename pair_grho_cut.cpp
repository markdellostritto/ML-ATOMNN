/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "pair_grho_cut.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <iostream>
using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairGRhoCut::PairGRhoCut(LAMMPS *lmp) : Pair(lmp)
{
  ewaldflag = pppmflag = 0;
  born_matrix_enable = 0;
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairGRhoCut::~PairGRhoCut()
{
  if (copymode) return;
  if (allocated) {
    //standard
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    //grho
    memory->destroy(gamma);
    memory->destroy(rgamma);
  }
}

/* ---------------------------------------------------------------------- */

void PairGRhoCut::compute(int eflag, int vflag)
{
  double r2inv, r6inv, forcelj;
  
  const double qqrd2e=force->qqrd2e;
  ev_init(eflag, vflag);

  double *q = atom->q;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  const int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  const int newton_pair = force->newton_pair;

  const int inum = list->inum;
  const int* ilist = list->ilist;
  const int* numneigh = list->numneigh;
  int** firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const double xi = x[i][0];
    const double yi = x[i][1];
    const double zi = x[i][2];
    const double qi = q[i];
    const int itype = type[i];
    const int* jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      const double factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      const double delx = xi - x[j][0];
      const double dely = yi - x[j][1];
      const double delz = zi - x[j][2];
      const double qj = q[j];
      const double dr2 = delx * delx + dely * dely + delz * delz;
      const int jtype = type[j];

      if (dr2 < cutsq[itype][jtype]) {
        const double dr=std::sqrt(dr2);
        
        //coulomb
        const double prefactor=qqrd2e*qi*qj/dr;
        const double ferf=std::erf(rgamma[itype][jtype]*dr);
        double fpair=0.0;
        if(dr>1.0e-8){
          fpair=prefactor/dr2*(
            ferf-2.0/MY_PIS*dr*rgamma[itype][jtype]*std::exp(-0.5*gamma[itype][jtype]*dr2)
          );
          if (factor_coul < 1.0) fpair -= (1.0 - factor_coul) * prefactor;
        }

        //force
        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        //energy
        double ecoul=0.0;
        if (eflag) {
          if(dr>1.0e-8){
            ecoul = prefactor*ferf;
            if (factor_coul < 1.0) ecoul -= (1.0 - factor_coul) * prefactor;
          } else {
            ecoul = qqrd2e*qi*qj*2.0/MY_PIS*rgamma[itype][jtype];
          }
        }

        if (evflag) ev_tally(i, j, nlocal, newton_pair, 0.0, ecoul, fpair, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairGRhoCut::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++){
    for (int j = i; j < n; j++){
       setflag[i][j] = 0;
    }
  }

  //standard
  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(cut, n, n, "pair:cut");
  //cgem
  memory->create(gamma, n, n, "pair:gamma");
  memory->create(rgamma, n, n, "pair:rgamma");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairGRhoCut::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal pair_style command");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  // reset cutoffs that have been explicitly set
  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++){
      for (int j = i; j <= atom->ntypes; j++){
        if (setflag[i][j]) cut[i][j] = cut_global;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairGRhoCut::coeff(int narg, char **arg)
{
  if (narg < 3 || narg > 4) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double gamma_one = utils::numeric(FLERR, arg[2], false, lmp);
  double cut_one = cut_global;
  if (narg == 4) cut_one = utils::numeric(FLERR, arg[3], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      gamma[i][j] = gamma_one;
      rgamma[i][j] = std::sqrt(0.5*gamma_one);
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairGRhoCut::init_style()
{
  // request regular or rRESPA neighbor list
  int list_style = NeighConst::REQ_DEFAULT;
  neighbor->add_request(this, list_style);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairGRhoCut::init_one(int i, int j)
{
  //set parameters (i,j)
  if (setflag[i][j] == 0) {
    //gamma - harmonic average
    gamma[i][j] = 2.0*gamma[i][i]*gamma[j][j]/(gamma[i][i]+gamma[j][j]);
    //arithmetic average
    cut[i][j] = 0.5*(cut[i][i]+cut[j][j]);
  }
  rgamma[i][j] = std::sqrt(0.5*gamma[i][j]);
  cutsq[i][j]=cut[i][j]*cut[i][j];

  //reflection
  gamma[j][i]=gamma[i][j];
  rgamma[j][i]=rgamma[i][j];
  cut[j][i]=cut[i][j];
  cutsq[j][i]=cutsq[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGRhoCut::write_restart(FILE *fp)
{
  write_restart_settings(fp);
  for (int i = 1; i <= atom->ntypes; i++){
    for (int j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&gamma[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGRhoCut::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  const int me = comm->me;
  for (int i = 1; i <= atom->ntypes; i++){
    for (int j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR, &gamma[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&gamma[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGRhoCut::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGRhoCut::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
  }
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairGRhoCut::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g\n", i, gamma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairGRhoCut::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++){
    for (int j = i; j <= atom->ntypes; j++){
      fprintf(fp, "%d %d %g %g\n", i, j, gamma[i][j], cut[i][j]);
    }
  }
}

/* ---------------------------------------------------------------------- */

double PairGRhoCut::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj, double &fforce)
{
  double *q = atom->q;
  const double r=std::sqrt(rsq);
  
  // coulomb
  const double qqrd2e=force->qqrd2e;
  const double prefactor=qqrd2e*q[i]*q[j]/r;
  const double ferf=std::erf(rgamma[itype][jtype]*r);
  double eCoul = prefactor*ferf;
  double fCoul=prefactor/rsq*(
    ferf-2.0/MY_PIS*r*rgamma[itype][jtype]*std::exp(-0.5*gamma[itype][jtype]*rsq)
  );
  if (factor_coul < 1.0){
    eCoul -= (1.0 - factor_coul) * prefactor;
    fCoul -= (1.0 - factor_coul) * prefactor;
  } 
  
  //force
  fforce = fCoul;

  //energy
  return eCoul;
}

/* ---------------------------------------------------------------------- */

void *PairGRhoCut::extract(const char *str, int &dim)
{
  if (strcmp(str, "gamma") == 0) {
    dim = 2;
    return (void *) gamma;
  }
  return nullptr;
}
