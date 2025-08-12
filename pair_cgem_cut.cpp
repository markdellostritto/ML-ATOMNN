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

#include "pair_cgem_cut.h"

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

PairCGemCut::PairCGemCut(LAMMPS *lmp) : Pair(lmp)
{
  ewaldflag = pppmflag = 0;
  born_matrix_enable = 0;
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairCGemCut::~PairCGemCut()
{
  if (copymode) return;
  if (allocated) {
    //standard
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    //cgem
    memory->destroy(radius);
    memory->destroy(alphaC);
    memory->destroy(alphaS);
    memory->destroy(aOver);
    memory->destroy(aRep);
    memory->destroy(muS);
    memory->destroy(muC);
    memory->destroy(rmuC);
  }
}

/* ---------------------------------------------------------------------- */

void PairCGemCut::compute(int eflag, int vflag)
{
  double r2inv, r6inv, forcelj;
  
  const double pe=1.0/(2.0*0.05*0.05);

  double evdwl = 0.0;
  double ecoul = 0.0;
  const double qqrd2e=force->qqrd2e;
  ev_init(eflag, vflag);

  double *q = atom->q;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  const int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
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
    const double Zi = std::fabs(q[i]);
    const int itype = type[i];
    const int* jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      const double factor_lj = special_lj[sbmask(j)];
      const double factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      const double delx = xi - x[j][0];
      const double dely = yi - x[j][1];
      const double delz = zi - x[j][2];
      const double qj = q[j];
      const double Zj = std::fabs(q[j]);
      const double dr2 = delx * delx + dely * dely + delz * delz;
      const int jtype = type[j];

      if (dr2 < cutsq[itype][jtype]) {
        const double dr=std::sqrt(dr2);
        
        //coulomb
        const double prefactor=qqrd2e*qi*qj/dr;
        const double ferf=std::erf(rmuC[itype][jtype]*dr);
        double fCoul=0;
        if(dr>1.0e-8){
          fCoul=prefactor/dr2*(
            ferf-2.0/MY_PIS*dr*rmuC[itype][jtype]*std::exp(-muC[itype][jtype]*dr2)
          );
          if (factor_coul < 1.0) fCoul -= (1.0 - factor_coul) * prefactor;
        }

        //overlap
        const double eOver=aOver[itype][jtype]*Zi*Zj*std::exp(-muS[itype][jtype]*dr2)*factor_lj;
        const double fOver=2.0*muS[itype][jtype]*eOver;
        
        //repulsion
        const double eRep=aRep[itype][jtype]*std::exp(-pe*dr)*factor_lj;
        const double fRep=pe*eRep;
        
        //force
        const double fpair = fCoul+fOver+fRep;
        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        //energy
        if (eflag) {
          evdwl = eOver+eRep;
          if(dr>1.0e-8){
            ecoul = prefactor*ferf;
            if (factor_coul < 1.0) ecoul -= (1.0 - factor_coul) * prefactor;
          } else {
            ecoul = qqrd2e*qi*qj*2.0/MY_PIS*rmuC[itype][jtype];
          }
        }

        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, ecoul, fpair, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairCGemCut::allocate()
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
  memory->create(radius, n, n, "pair:radius");
  memory->create(alphaC, n, n, "pair:alphaC");
  memory->create(alphaS, n, n, "pair:alphaS");
  memory->create(aOver, n, n, "pair:aOver");
  memory->create(aRep, n, n, "pair:aRep");
  memory->create(muS, n, n, "pair:muS");
  memory->create(muC, n, n, "pair:muC");
  memory->create(rmuC, n, n, "pair:rmuC");

  for (int i = 1; i < n; i++){
    for (int j = i; j < n; j++){
      radius[i][j] = 0;
      alphaC[i][j] = 0;
      alphaS[i][j] = 0;
    }
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairCGemCut::settings(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR, "Illegal pair_style command");
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  lambdaC = utils::numeric(FLERR, arg[1], false, lmp);
  lambdaS = utils::numeric(FLERR, arg[2], false, lmp);
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

void PairCGemCut::coeff(int narg, char **arg)
{
  if (narg < 5 || narg > 6) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double radius_one = utils::numeric(FLERR, arg[2], false, lmp);
  double aOver_one = utils::numeric(FLERR, arg[3], false, lmp);
  double aRep_one = utils::numeric(FLERR, arg[4], false, lmp);

  double cut_one = cut_global;
  if (narg == 6) cut_one = utils::numeric(FLERR, arg[5], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      radius[i][j] = radius_one;
      alphaC[i][j] = lambdaC/(2.0*radius_one*radius_one);
      alphaS[i][j] = lambdaS/(2.0*radius_one*radius_one);
      aOver[i][j] = aOver_one;
      aRep[i][j] = aRep_one;
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

void PairCGemCut::init_style()
{
  // request regular or rRESPA neighbor list
  int list_style = NeighConst::REQ_DEFAULT;
  neighbor->add_request(this, list_style);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairCGemCut::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    if(aOver[i][j] > 1.0e-8){
      aOver[i][j] = 2.0*(aOver[i][i]*aOver[j][j])/(aOver[i][i]+aOver[j][j]);
    } else aOver[i][j] = 0.0;
    if(aRep[i][j] > 1.0e-8){
      aRep[i][j] = 2.0*(aRep[i][i]*aRep[j][j])/(aRep[i][i]+aRep[j][j]);
    } else aRep[i][j] = 0.0;
    cut[i][j] = 0.5*(cut[i][i]+cut[j][j]);
  }

  aOver[j][i] = aOver[i][j];
  aRep[j][i] = aRep[i][j];
  cut[j][i] = cut[i][j];

  muC[i][j] = alphaC[i][i]*alphaC[j][j]/(alphaC[i][i]+alphaC[j][j]);
  muS[i][j] = alphaS[i][i]*alphaS[j][j]/(alphaS[i][i]+alphaS[j][j]);
  rmuC[i][j] = sqrt(muC[i][j]);

  muS[j][i]=muS[i][j];
  muC[j][i]=muC[i][j];
  rmuC[j][i]=rmuC[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairCGemCut::write_restart(FILE *fp)
{
  write_restart_settings(fp);
  for (int i = 1; i <= atom->ntypes; i++){
    for (int j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&radius[i][j], sizeof(double), 1, fp);
        fwrite(&aOver[i][j], sizeof(double), 1, fp);
        fwrite(&aRep[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairCGemCut::read_restart(FILE *fp)
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
          utils::sfread(FLERR, &radius[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &aOver[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &aRep[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&radius[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&aOver[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&aRep[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairCGemCut::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&lambdaC, sizeof(double), 1, fp);
  fwrite(&lambdaS, sizeof(double), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairCGemCut::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &lambdaC, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &lambdaS, sizeof(double), 1, fp, nullptr, error);
  }
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&lambdaC, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&lambdaS, 1, MPI_DOUBLE, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairCGemCut::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g %g %g\n", i, radius[i][i], aOver[i][i], aRep[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairCGemCut::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++){
    for (int j = i; j <= atom->ntypes; j++){
      fprintf(fp, "%d %d %g %g %g %g\n", i, j, radius[i][j], aOver[i][j], aRep[i][j], cut[i][j]);
    }
  }
}

/* ---------------------------------------------------------------------- */

double PairCGemCut::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj, double &fforce)
{
  double *q = atom->q;
  const double r=std::sqrt(rsq);
  const double pe=1.0/(2.0*0.05*0.05);

  // coulomb
  const double qqrd2e=force->qqrd2e;
  const double prefactor=qqrd2e*q[i]*q[j]/r;
  const double ferf=std::erf(rmuC[itype][jtype]*r);
  double eCoul = prefactor*ferf;
  double fCoul=prefactor/rsq*(
    ferf-2.0/MY_PIS*r*rmuC[itype][jtype]*std::exp(-muC[itype][jtype]*rsq)
  );
  if (factor_coul < 1.0){
    eCoul -= (1.0 - factor_coul) * prefactor;
    fCoul -= (1.0 - factor_coul) * prefactor;
  } 
  
  // overlap
  double eOver=aOver[itype][jtype]*std::fabs(q[i])*std::fabs(q[j])*std::exp(-muS[itype][jtype]*rsq)*factor_lj;
  double fOver=2.0*muS[itype][jtype]*eOver;
  
  // repulsive
  const double eRep=aRep[itype][jtype]*std::exp(-pe*r)*factor_lj;
  const double fRep=pe*eRep;

  //force
  fforce = fCoul+fOver+fRep;

  //energy
  return eCoul+eOver+eRep;
}

/* ---------------------------------------------------------------------- */

void *PairCGemCut::extract(const char *str, int &dim)
{
  if (strcmp(str, "radius") == 0) {
    dim = 2;
    return (void *) radius;
  }
  if (strcmp(str, "aOver") == 0) {
    dim = 2;
    return (void *) aOver;
  }
  if (strcmp(str, "aRep") == 0) {
    dim = 2;
    return (void *) aRep;
  }
  return nullptr;
}
