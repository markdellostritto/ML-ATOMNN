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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(ldamp/long,PairLDampLong);
// clang-format on
#else

#ifndef LMP_PAIR_LDAMP_LONG_H
#define LMP_PAIR_LDAMP_LONG_H

#include "pair.h"

#ifndef PAIR_LDAMP_LONG_PRINT_FUNC
#define PAIR_LDAMP_LONG_PRINT_FUNC 0
#endif

#ifndef LDAMP_A
#define LDAMP_A 6
#endif

namespace LAMMPS_NS {

class PairLDampLong: public Pair {
public:
	PairLDampLong(class LAMMPS *);
	virtual ~PairLDampLong();
	virtual void compute(int, int);
	void options(char **arg, int order);
	void settings(int, char **);
	void coeff(int, char **);
	void init_style();
	double init_one(int, int);
	void write_restart(FILE *);
	void read_restart(FILE *);
	void write_restart_settings(FILE *);
	void read_restart_settings(FILE *);
	void write_data(FILE *);
	void write_data_all(FILE *);
	double single(int, int, int, int, double, double, double, double &);
	void *extract(const char *, int &);
protected:
	//pair parameters
	double ge6_,ge62_,ge66_;
	double rc_;
	double** rvdw_;
	double** rvdw3_;
	double** rvdw6_;
	double** c6_;
	int ewald_order;
	int ewald_off;
	//functions
	virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

*/
