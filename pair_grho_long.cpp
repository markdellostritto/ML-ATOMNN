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

#include "pair_grho_long.h"

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

PairGRhoLong::PairGRhoLong(LAMMPS *lmp) : Pair(lmp)
{
  ewaldflag = pppmflag = 1;
  born_matrix_enable = 0;
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairGRhoLong::~PairGRhoLong()
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

void PairGRhoLong::compute(int eflag, int vflag)
{
  // local constants
  const double qqrd2e=force->qqrd2e;

  // total energies
  ev_init(eflag, vflag);

  // atom properties
  double *q = atom->q;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  const int nlocal = atom->nlocal;

  // calculation flags/factors
  double *special_coul = force->special_coul;
  const int newton_pair = force->newton_pair;

  // neighbors
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
        
        // compute energy/force - coulomb
        const double prefactor=qqrd2e*qi*qj/dr;
        const double ferfg=std::erf(rgamma[itype][jtype]*dr);
        const double ferfp=std::erf(g_ewald*dr);
        double fpair=0;
        if(dr>1e-8){
            fpair=prefactor/dr2*(
            (ferfg-ferfp)
            +2.0/MY_PIS*dr*(
              -rgamma[itype][jtype]*std::exp(-0.5*gamma[itype][jtype]*dr2)
              +g_ewald*std::exp(-g_ewald*g_ewald*dr2)
            )
          );
          if (factor_coul < 1.0) fpair -= (1.0 - factor_coul) * prefactor;
        } 
        
        // compute total force
        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        // compute total energy
        double ecoul=0.0;
        if (eflag) {
          if(dr>1e-8){
            ecoul = prefactor*(ferfg-ferfp);
            if (factor_coul < 1.0) ecoul -= (1.0 - factor_coul) * prefactor;
          } else {
            ecoul = 2.0/MY_PIS*qqrd2e*qi*qj*(rgamma[itype][jtype]-g_ewald);
          }
        }

        // tally energy and force
        if (evflag) ev_tally(i, j, nlocal, newton_pair, 0.0, ecoul, fpair, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairGRhoLong::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;
  //flags
  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++){
    for (int j = i; j < n; j++){
       setflag[i][j] = 0;
    }
  }
  //standard
  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(cut, n, n, "pair:cut");
  //grho
  memory->create(gamma, n, n, "pair:gamma");
  memory->create(rgamma, n, n, "pair:rgamma");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairGRhoLong::settings(int narg, char **arg)
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

void PairGRhoLong::coeff(int narg, char **arg)
{
  if (narg < 3 || narg > 4) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double gamma_one = utils::numeric(FLERR, arg[2], false, lmp);
  double cut_one = cut_global;
  if (narg == 7) cut_one = utils::numeric(FLERR, arg[3], false, lmp);
  
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

void PairGRhoLong::init_style()
{
  //check for charge
  if (!atom->q_flag) error->all(FLERR, "Pair style lj/cut/coul/long requires atom attribute q");
  //set neighborlist style
  int list_style = NeighConst::REQ_DEFAULT;
  neighbor->add_request(this, list_style);
  // ensure use of KSpace long-range solver, set g_ewald
  if (force->kspace == nullptr) error->all(FLERR, "Pair style requires a KSpace style");
  g_ewald = force->kspace->g_ewald;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairGRhoLong::init_one(int i, int j)
{
  //set parameters (i,j)
  if (setflag[i][j] == 0) {
    //gamma - harmonic average
    gamma[i][j] = 2.0*gamma[i][i]*gamma[j][j]/(gamma[i][i]+gamma[j][j]);
    //cutoff - arithmetic average
    cut[i][j] = 0.5*(cut[i][i]+cut[j][j]);
  }
  rgamma[i][j] = std::sqrt(0.5*gamma[i][j]);
  cutsq[i][j] = cut[i][j]*cut[i][j];

  //reflection (j,i)
  gamma[j][i]=gamma[i][j];
  rgamma[j][i]=rgamma[i][j];
  cut[j][i]=cut[i][j];
  cutsq[j][i]=cutsq[i][j];

  //return cut
  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGRhoLong::write_restart(FILE *fp)
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

void PairGRhoLong::read_restart(FILE *fp)
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

void PairGRhoLong::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGRhoLong::read_restart_settings(FILE *fp)
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

void PairGRhoLong::write_data(FILE *fp)
{
    for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g\n", i, gamma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairGRhoLong::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++){
    for (int j = i; j <= atom->ntypes; j++){
      fprintf(fp, "%d %d %g %g\n", i, j, gamma[i][j], cut[i][j]);
    }
  }
}

/* ---------------------------------------------------------------------- */

double PairGRhoLong::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj, double &fforce)
{
  double *q = atom->q;
  const double r=std::sqrt(rsq);
  
  // compute energy/force - coulomb
  const double qqrd2e=force->qqrd2e;
  const double prefactor=qqrd2e*q[i]*q[j]/r;
  const double ferfg=std::erf(rgamma[itype][jtype]*r);
  const double ferfp=std::erf(g_ewald*r);
  double eCoul=0;
  double fCoul=0;
  if(r>1e-8){
    eCoul = prefactor*(ferfg-ferfp);
    fCoul=prefactor/rsq*(
      (ferfg-ferfp)
      +2.0/MY_PIS*r*(
        -rgamma[itype][jtype]*std::exp(-0.5*gamma[itype][jtype]*rsq)
        +g_ewald*std::exp(-g_ewald*g_ewald*rsq)
      )
    );
    if (factor_coul < 1.0) {
      eCoul -= (1.0 - factor_coul) * prefactor;
      fCoul -= (1.0 - factor_coul) * prefactor;
    }
  } else {
    eCoul = 2.0/MY_PIS*qqrd2e*q[i]*q[j]*(rgamma[itype][jtype]-g_ewald);
  }
  
  // compute total force
  fforce = fCoul;

  // compute total energy
  return eCoul;
}

/* ---------------------------------------------------------------------- */

void *PairGRhoLong::extract(const char *str, int &dim)
{
  if (strcmp(str, "cut_coul") == 0) {
    dim = 0;
    return (void *) &cut_global;
  }
  if (strcmp(str, "gamma") == 0) {
    dim = 2;
    return (void *) gamma;
  }
  return nullptr;
}
