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
   Contributing author: Mark DelloStritto
------------------------------------------------------------------------- */

#include "pair_lj_long.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <iostream>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairLJLong::PairLJLong(LAMMPS *lmp) : Pair(lmp){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::PairLJLong(LAMMPS*):\n";
	writedata = 0;
	ewaldflag = 1;
	dispersionflag = 1;
	epsilon_=nullptr;
	sigma_=nullptr;
	c6_=nullptr;
}

/* ---------------------------------------------------------------------- */

PairLJLong::~PairLJLong(){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::~PairLJLong():\n";
	if (copymode) return;
	if (allocated) {
		//global pair data
		memory->destroy(setflag);
		memory->destroy(cutsq);
		memory->destroy(epsilon_);
		memory->destroy(sigma_);
		memory->destroy(c6_);
	}
}

/* ---------------------------------------------------------------------- */

void PairLJLong::compute(int eflag, int vflag){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::compute(int,int):\n";
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
	
	if (eflag || vflag) ev_setup(eflag,vflag);
	else evflag = vflag_fdotr = 0;
	
	//loop over all owned atoms
	for (int ii=0; ii<inum; ++ii) {
		const int i=ilist[ii];
		const double xtmp=x[i][0];
		const double ytmp=x[i][1];
		const double ztmp=x[i][2];
		const int itype=type[i];
		const int* jlist=firstneigh[i];
		const int jnum=numneigh[i];
		//loop over all nearest neighbors
		for (int jj=0; jj<jnum; ++jj) {
			int j = jlist[jj];
			j&=NEIGHMASK;
			//compute distance
			const double delx=xtmp-x[j][0];
			const double dely=ytmp-x[j][1];
			const double delz=ztmp-x[j][2];
			const double dr2=(delx*delx+dely*dely+delz*delz);
			const int jtype=type[j];
			//check distance
			if(dr2<cutsq[itype][jtype]){
				//compute force term
				const double du2=sigma_[itype][jtype]*sigma_[itype][jtype]/dr2;
				const double du6=du2*du2*du2;
				const double b2=dr2*ge62_;
				const double expf=exp(-b2);
				const double fpair=24.0*epsilon_[itype][jtype]*du6*(2.0*du6-1.0/6.0*expf*(6.0+b2*(6.0+b2*(3.0+b2))))/dr2;
				//compute forces
				f[i][0]+=delx*fpair;
				f[i][1]+=dely*fpair;
				f[i][2]+=delz*fpair;
				if(newton_pair || j<nlocal){
					f[j][0]-=delx*fpair;
					f[j][1]-=dely*fpair;
					f[j][2]-=delz*fpair;
				}
				//compute energy
				double evdwl=0.0;
				if(eflag) evdwl=4.0*epsilon_[itype][jtype]*du6*(du6-expf*(1.0+b2*(1.0+0.5*b2)));
				//tally energy
				if(evflag) ev_tally(i,j,nlocal,newton_pair,evdwl,0.0,fpair,delx,dely,delz);
			}
		}
	}
	
	if(vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJLong::allocate(){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::allocate():\n";
	allocated=1;
	const int n=atom->ntypes+1;
	memory->create(setflag,n,n,"pair:setflag");
	for (int i=1; i<n; i++){
		for (int j=i; j<n; j++){
			setflag[i][j]=0;
		}
	}
	memory->create(cutsq, n, n, "pair:cutsq");
	memory->create(epsilon_, n, n, "pair:epsilon");
	memory->create(sigma_, n, n, "pair:sigma");
	memory->create(c6_, n, n, "pair:c6");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJLong::settings(int narg, char **arg){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::settings(int,char**):\n";
	if (narg != 1) error->all(FLERR, "Illegal pair_style command");
	rc_=utils::numeric(FLERR,arg[0],false,lmp);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJLong::coeff(int narg, char **arg){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::coeff(int,char**):\n";
	if(narg!=4) error->all(FLERR, "Incorrect args for pair coefficients");
	if(!allocated) allocate();
	
	int ilo,ihi,jlo,jhi;
	utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
	utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
	
	const double epsilon=utils::numeric(FLERR,arg[2],false,lmp);
	const double sigma=utils::numeric(FLERR,arg[3],false,lmp);
	
	int count=0;
	for(int i=ilo; i<=ihi; i++){
		for(int j=MAX(jlo,i); j<=jhi; j++){
			epsilon_[i][j]=epsilon;
			sigma_[i][j]=sigma;
			c6_[i][j]=4.0*epsilon_[i][j]*sigma_[i][j]*sigma_[i][j]*sigma_[i][j]*sigma_[i][j]*sigma_[i][j]*sigma_[i][j];
			setflag[i][j]=1;
			count++;
		}
	}

	if(count==0) error->all(FLERR, "Incorrect args for pair coefficients");

}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJLong::init_style(){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::init_style():\n";
	if(atom->tag_enable==0) error->all(FLERR,"Pair style London requires atom IDs");
	
	// ensure use of KSpace long-range solver, set g_ewald for disp
	if (force->kspace == nullptr) error->all(FLERR,"Pair style requires a KSpace style");
	ge6_ = force->kspace->g_ewald_6;
	ge62_ = ge6_*ge6_; //g_ewald_6^2
	
	//==== need a half neighbor list ====
	/*
	const int irequest=neighbor->request(this,instance_me);
	neighbor->requests[irequest]->half=1;//enable half-neighbor list
	neighbor->requests[irequest]->full=0;//disable full-neighbor list
	*/
	
	neighbor->add_request(this);
	
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJLong::init_one(int i, int j){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::init_one(int,int):\n";
	if (setflag[i][j]==0) {
		epsilon_[i][j]=sqrt(epsilon_[i][i]*epsilon_[j][j]);
		sigma_[i][j]=sqrt(sigma_[i][i]*sigma_[j][j]);
		c6_[i][j]=4.0*epsilon_[i][j]*sigma_[i][j]*sigma_[i][j]*sigma_[i][j]*sigma_[i][j]*sigma_[i][j]*sigma_[i][j];
	}
	cutsq[i][j] = rc_*rc_;
	
	epsilon_[j][i]=epsilon_[i][j];
	sigma_[j][i]=sigma_[i][j];
	cutsq[j][i]=cutsq[i][j];
	
	if (tail_flag) {
		
	}
	
	return rc_;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJLong::write_restart(FILE *fp){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::write_restart(FILE*):\n";
	write_restart_settings(fp);
	for(int i=1; i<=atom->ntypes; i++){
		for(int j=i; j<=atom->ntypes; j++) {
			fwrite(&setflag[i][j], sizeof(int), 1, fp);
			if (setflag[i][j]) {
				fwrite(&epsilon_[i][j], sizeof(double), 1, fp);
				fwrite(&sigma_[i][j], sizeof(double), 1, fp);
			}
		}
	}
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJLong::read_restart(FILE *fp){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::read_restart(FILE*):\n";
	read_restart_settings(fp);
	allocate();
	const int me = comm->me;
	for (int i=1; i<=atom->ntypes; i++){
		for (int j=i; j<=atom->ntypes; j++) {
			if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
			MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
			if (setflag[i][j]) {
				if (me == 0) {
					utils::sfread(FLERR,&epsilon_[i][j],sizeof(double),1,fp,nullptr,error);
					utils::sfread(FLERR,&sigma_[i][j],sizeof(double),1,fp,nullptr,error);
				}
				MPI_Bcast(&epsilon_[i][j],1,MPI_DOUBLE,0,world);
				MPI_Bcast(&sigma_[i][j],1,MPI_DOUBLE,0,world);
			}
		}
	}
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJLong::write_restart_settings(FILE *fp){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::write_restart_settings(FILE*):\n";
	fwrite(&rc_, sizeof(double), 1, fp);
	fwrite(&mix_flag, sizeof(int), 1, fp);
	fwrite(&tail_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJLong::read_restart_settings(FILE *fp){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::read_restart_settings(FILE*):\n";
	const int me = comm->me;
	if (me == 0) {
		utils::sfread(FLERR, &rc_, sizeof(double), 1, fp, nullptr, error);
		utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
		utils::sfread(FLERR, &tail_flag, sizeof(int), 1, fp, nullptr, error);
	}
	MPI_Bcast(&rc_, 1, MPI_DOUBLE, 0, world);
	MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
	MPI_Bcast(&tail_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLJLong::write_data(FILE *fp){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::write_data(FILE*):\n";
	for (int i=1; i<=atom->ntypes; i++) fprintf(fp,"%d %g %g\n",i,epsilon_[i][i],sigma_[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLJLong::write_data_all(FILE *fp){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::write_data_all(FILE*):\n";
	for (int i=1; i<=atom->ntypes; i++){
		for (int j=i; j<=atom->ntypes; j++){
			fprintf(fp,"%d %d %g %g\n",i,j,epsilon_[i][j],sigma_[i][j]);
		}
	}
}

/* ---------------------------------------------------------------------- */

double PairLJLong::single(int i, int j, int itype, int jtype, double rsq, double factor_coul, double factor_lj, double &fforce){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::single(int,int,int,int,rsq,double,double,double&):\n";
	const double du2=sigma_[itype][jtype]*sigma_[itype][jtype]/rsq;
	const double du6=du2*du2*du2;
	const double b2=rsq*ge62_;
	const double expf=exp(-b2);
	fforce=24.0*epsilon_[itype][jtype]*du6*(2.0*du6-1.0/6.0*expf*(6.0+b2*(6.0+b2*(3.0+b2))))/rsq;
	return 4.0*epsilon_[itype][jtype]*du6*(du6-expf*(1.0+b2*(1.0+0.5*b2)));
	
}

/* ---------------------------------------------------------------------- */

void *PairLJLong::extract(const char *str, int &dim){
	if(PAIR_LJ_LONG_PRINT_FUNC>0) std::cout<<"PairLJLong::extract(const char*,int&):\n";
	if(strcmp(str,"cut_london")==0){
		dim=1;
		return (void *) &rc_;
	}
	if(strcmp(str,"epsilon")==0){
		dim=2;
		return (void *)epsilon_;
	}
	if(strcmp(str,"sigma")==0){
		dim=2;
		return (void *)sigma_;
	}
	if(strcmp(str,"c6")==0){
		dim=2;
		return (void *)c6_;
	}
	if(strcmp(str,"B")==0){
		dim=2;
		return (void *)c6_;
	}
	return nullptr;
}
