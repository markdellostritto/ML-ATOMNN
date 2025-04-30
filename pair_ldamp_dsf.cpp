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

#include "pair_ldamp_dsf.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
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

PairLDampDSF::PairLDampDSF(LAMMPS *lmp) : Pair(lmp){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::PairLDampDSF(LAMMPS*):\n";
	writedata = 0;
	c6_=nullptr;
	rvdw_=nullptr;
	rvdw6_=nullptr;
	potfr_=nullptr;
	potgr_=nullptr;
	potfRc_=nullptr;
	potgRc_=nullptr;
}

/* ---------------------------------------------------------------------- */

PairLDampDSF::~PairLDampDSF(){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::~PairLDampDSF():\n";
	if (copymode) return;
	if (allocated) {
		//global pair data
		memory->destroy(setflag);
		memory->destroy(cutsq);
		memory->destroy(c6_);
		memory->destroy(rvdw_);
		memory->destroy(rvdw6_);
		memory->destroy(potfr_);
		memory->destroy(potgr_);
		memory->destroy(potfRc_);
		memory->destroy(potgRc_);
	}
}

/* ---------------------------------------------------------------------- */

void PairLDampDSF::compute(int eflag, int vflag){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::compute(int,int):\n";
	//energy
	double evdwl = 0.0;
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
	double etot=0;
	for (int ii=0; ii<inum; ii++) {
		const int i=ilist[ii];
		const double xtmp=x[i][0];
		const double ytmp=x[i][1];
		const double ztmp=x[i][2];
		const int itype=type[i];
		const int* jlist=firstneigh[i];
		const int jnum=numneigh[i];
		//loop over all nearest neighbors
		for (int jj=0; jj<jnum; jj++) {
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
				const double dr=sqrt(dr2);
				const double dr6=dr2*dr2*dr2;
				const double b2=dr2*a2_;
				const double expf=exp(-b2);
				const double den=1.0/(dr6+rvdw6_[itype][jtype]);
				const double potfr_=(
					-1.0*den
					-(-1.0/dr6)
					-expf*(1.0+b2*(1.0+0.5*b2))/dr6
				);
				const double potgr_=-1.0/dr*(
					-6.0*dr6*den*den
					-(-6.0/dr6)
					-expf*(6.0+b2*(6.0+b2*(3.0+b2)))/dr6
				);
				const double fpair=-1.0*c6_[itype][jtype]/dr*(potgr_-potgRc_[itype][jtype]);
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
				if(eflag){
					evdwl=-c6_[itype][jtype]*(potfr_-potfRc_[itype][jtype]-potgRc_[itype][jtype]*(dr-rc_));
				}
				if(eflag) etot+=evdwl;
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

void PairLDampDSF::allocate(){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::allocate():\n";
	allocated=1;
	const int n=atom->ntypes+1;
	memory->create(setflag,n,n,"pair:setflag");
	for (int i=1; i<n; ++i){
		for (int j=i; j<n; ++j){
			setflag[i][j]=0;
		}
	}
	memory->create(cutsq, n, n, "pair:cutsq");
	memory->create(rvdw_, n, n, "pair:rvdw");
	memory->create(rvdw6_, n, n, "pair:rvdw6");
	memory->create(potfr_, n, n, "pair:potfr");
	memory->create(potgr_, n, n, "pair:potgr");
	memory->create(potfRc_, n, n, "pair:potfRc");
	memory->create(potgRc_, n, n, "pair:potgRc");
	memory->create(c6_, n, n, "pair:c6");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLDampDSF::settings(int narg, char **arg){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::settings(int,char**):\n";
	if (narg != 2) error->all(FLERR, "Illegal pair_style command");
	rc_=utils::numeric(FLERR,arg[0],false,lmp);
	a_=utils::numeric(FLERR,arg[1],false,lmp);
	const double rc2_=rc_*rc_;
	rc6_=rc2_*rc2_*rc2_;
	a2_=a_*a_;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLDampDSF::coeff(int narg, char **arg){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::coeff(int,char**):\n";
	if(narg!=4) error->all(FLERR, "Incorrect args for pair coefficients");
	if(!allocated) allocate();
	
	int ilo,ihi,jlo,jhi;
	utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
	utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
	
	const double c6=utils::numeric(FLERR,arg[2],false,lmp);
	const double rvdw=utils::numeric(FLERR,arg[3],false,lmp);
	const double rvdw2=rvdw*rvdw;
	const double rvdw6=rvdw2*rvdw2*rvdw2;
	
	int count=0;
	for(int i=ilo; i<=ihi; i++){
		for(int j=MAX(jlo,i); j<=jhi; j++){
			c6_[i][j]=c6;
			rvdw_[i][j]=rvdw;
			rvdw6_[i][j]=rvdw6;
			setflag[i][j]=1;
			count++;
		}
	}

	if(count==0) error->all(FLERR, "Incorrect args for pair coefficients");

}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLDampDSF::init_style(){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::init_style():\n";
	if(atom->tag_enable==0) error->all(FLERR,"Pair style London requires atom IDs");
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

double PairLDampDSF::init_one(int i, int j){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::init_one(int,int):\n";
	if (setflag[i][j]==0) {
		c6_[i][j]=sqrt(c6_[i][i]*c6_[j][j]);
		const double rvdw=0.5*(rvdw_[i][i]+rvdw_[j][j]);
		const double rvdw2=rvdw*rvdw;
		rvdw_[i][j]=rvdw;
		rvdw6_[i][j]=rvdw2*rvdw2*rvdw2;
		const double bc2=rc_*rc_*a2_;
		const double den=1.0/(rc6_+rvdw6_[i][j]);
		potfRc_[i][j]=(
			-1.0*den
			-(-1.0/rc6_)
			-exp(-bc2)*(1.0+bc2*(1.0+0.5*bc2))/rc6_
		);
		potgRc_[i][j]=-1.0/rc_*(
			-6.0*den*den
			-(-6.0/rc6_)
			-exp(-bc2)*(6.0+bc2*(6.0+bc2*(3.0+bc2)))/rc6_
		);
	}
	
	c6_[j][i]=c6_[i][j];
	rvdw_[j][i]=rvdw_[i][j];
	rvdw6_[j][i]=rvdw6_[i][j];
	potfRc_[j][i]=potfRc_[i][j];
	potgRc_[j][i]=potgRc_[i][j];
	
	if (tail_flag) {
		
	}
	
	return rc_;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLDampDSF::write_restart(FILE *fp){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::write_restart(FILE*):\n";
	write_restart_settings(fp);
	for(int i=1; i<=atom->ntypes; ++i){
		for(int j=i; j<=atom->ntypes; ++j){
			fwrite(&setflag[i][j], sizeof(int), 1, fp);
			if (setflag[i][j]) {
				fwrite(&c6_[i][j], sizeof(double), 1, fp);
				fwrite(&rvdw_[i][j], sizeof(double), 1, fp);
			}
		}
	}
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLDampDSF::read_restart(FILE *fp){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::read_restart(FILE*):\n";
	read_restart_settings(fp);
	allocate();
	const int me = comm->me;
	for (int i=1; i<=atom->ntypes; ++i){
		for (int j=i; j<=atom->ntypes; ++j) {
			if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
			MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
			if (setflag[i][j]) {
				if (me == 0) {
					utils::sfread(FLERR,&c6_[i][j],sizeof(double),1,fp,nullptr,error);
					utils::sfread(FLERR,&rvdw_[i][j],sizeof(double),1,fp,nullptr,error);
				}
				MPI_Bcast(&c6_[i][j],1,MPI_DOUBLE,0,world);
				MPI_Bcast(&rvdw_[i][j],1,MPI_DOUBLE,0,world);
			}
		}
	}
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLDampDSF::write_restart_settings(FILE *fp){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::write_restart_settings(FILE*):\n";
	fwrite(&rc_, sizeof(double), 1, fp);
	fwrite(&a_, sizeof(double), 1, fp);
	fwrite(&mix_flag, sizeof(int), 1, fp);
	fwrite(&tail_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLDampDSF::read_restart_settings(FILE *fp){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::read_restart_settings(FILE*):\n";
	const int me = comm->me;
	if (me == 0) {
		utils::sfread(FLERR, &rc_, sizeof(double), 1, fp, nullptr, error);
		utils::sfread(FLERR, &a_, sizeof(double), 1, fp, nullptr, error);
		utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
		utils::sfread(FLERR, &tail_flag, sizeof(int), 1, fp, nullptr, error);
	}
	MPI_Bcast(&rc_, 1, MPI_DOUBLE, 0, world);
	MPI_Bcast(&a_, 1, MPI_DOUBLE, 0, world);
	MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
	MPI_Bcast(&tail_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLDampDSF::write_data(FILE *fp){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::write_data(FILE*):\n";
	for (int i=1; i<=atom->ntypes; i++) fprintf(fp,"%d %g %g\n",i,c6_[i][i],rvdw_[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLDampDSF::write_data_all(FILE *fp){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::write_data_all(FILE*):\n";
	for (int i=1; i<=atom->ntypes; i++){
		for (int j=i; j<=atom->ntypes; j++){
			fprintf(fp,"%d %d %g %g\n",i,j,c6_[i][j],rvdw_[i][j]);
		}
	}
}

/* ---------------------------------------------------------------------- */

double PairLDampDSF::single(int i, int j, int itype, int jtype, double rsq, double factor_coul, double factor_lj, double &fforce){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::single(int,int,int,int,rsq,double,double,double&):\n";
	const double dr=sqrt(rsq);
	const double dr6=rsq*rsq*rsq;
	const double b2=rsq*a2_;
	const double expf=exp(-b2);
	const double den=1.0/(dr6+rvdw6_[itype][jtype]);
	const double potfr_=(
		-1.0*den
		-(-1.0/dr6)
		-expf*(1.0+b2*(1.0+0.5*b2))/dr6
	);
	const double potgr_=-1.0/dr*(
		-6.0*dr6*den*den
		-(-6.0/dr6)
		-expf*(6.0+b2*(6.0+b2*(3.0+b2)))/dr6
	);
	fforce=-1.0*c6_[itype][jtype]/dr*(potgr_-potgRc_[itype][jtype]);
	return -c6_[itype][jtype]*(potfr_-potfRc_[itype][jtype]-potgRc_[itype][jtype]*(dr-rc_));
}

/* ---------------------------------------------------------------------- */

void *PairLDampDSF::extract(const char *str, int &dim){
	if(PAIR_LDAMPDSF_PRINT_FUNC>0) std::cout<<"PairLDampDSF::extract(const char*,int&):\n";
	dim=2;
	if(strcmp(str,"c6")==0) return (void *)c6_;
	if(strcmp(str,"rvdw")==0) return (void *)rvdw_;
	return nullptr;
}
