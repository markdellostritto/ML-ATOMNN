/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(ewald/london,EwaldLondon);
// clang-format on
#else

#ifndef LMP_EWALD_LONDON_H
#define LMP_EWALD_LONDON_H

#include "kspace.h"

namespace LAMMPS_NS {

class EwaldLondon : public KSpace {
 public:
  EwaldLondon(class LAMMPS *);
  ~EwaldLondon() override;
  void init() override;
  void setup() override;
  void settings(int, char **) override;
  void compute(int, int) override;
  double memory_usage() override { return bytes; }

 private:
  double unit[6];
  int first_output;

  int nkvec, nkvec_max, nevec, nevec_max, nbox, sums;
  int peratom_allocate_flag;
  int nmax;
  double bytes;
  double gsqmx;
  double *kenergy, energy_self;
  double *kvirial, virial_self;
  double *energy_self_peratom;
  double *virial_self_peratom;
  struct cvector *ekr_local;
  struct hvector *hvec;
  struct kvector *kvec;

  double *B, volume;
  double bsum,b2sum,b2avg;
  struct complex *cek_local, *cek_global;

  double rms(int, double, bigint, double);
  void reallocate();
  void allocate_peratom();
  void reallocate_atoms();
  void deallocate();
  void deallocate_peratom();
  void coefficients();
  void init_coeffs();
  void init_coeff_sums();
  void init_self();
  void init_self_peratom();
  void compute_ek();
  void compute_force();
  void compute_surface(){};
  void compute_energy();
  void compute_energy_peratom();
  void compute_virial();
  void compute_virial_dipole(){};
  void compute_virial_peratom();
  void compute_slabcorr(){};
  double NewtonSolve(double, double, bigint, double, double);
  double fv(double pre, double g, double rc, double prec);
  double fd(double pre, double g, double rc, double prec);
  double f(double, double, bigint, double, double);
  double derivf(double, double, bigint, double, double);
};

}    // namespace LAMMPS_NS

#endif
#endif
