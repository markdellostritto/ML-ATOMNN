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

#include "pair_ldamp_cut.h"

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

PairLDampCut::PairLDampCut(LAMMPS *lmp) : Pair(lmp){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::PairLDampCut(LAMMPS*):\n";
	writedata = 0;
	c6_=nullptr;
	rvdw_=nullptr;
}

/* ---------------------------------------------------------------------- */

PairLDampCut::~PairLDampCut(){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::~PairLDampCut():\n";
	if (copymode) return;
	if (allocated) {
		//global pair data
		memory->destroy(setflag);
		memory->destroy(cutsq);
		memory->destroy(c6_);
		memory->destroy(rvdw_);
	}
}

/* ---------------------------------------------------------------------- */

void PairLDampCut::compute(int eflag, int vflag){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::compute(int,int):\n";
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
				const double rij=rvdw_[itype][jtype];
				#if LDAMP_A == 3
					//a=3,b=2
					const double dr=sqrt(dr2);
					const double dr3=dr2*dr;
					const double rij3=rij*rij*rij;
					const double den=1.0/(dr3+rij3);
					const double fpair=-6.0*c6_[itype][jtype]*dr*den*den*den;
				#elif LDAMP_A == 6
					//a=6,b=1
					const double rij2=rij*rij;
					const double rij6=rij2*rij2*rij2;
					const double dr6=dr2*dr2*dr2;
					const double den=1.0/(dr6+rij6);
					const double fpair=-6.0*c6_[itype][jtype]*dr2*dr2*den*den;
				#elif LDAMP_A == 12
					//a=12,b=1/2
					const double rij2=rij*rij;
					const double rij6=rij2*rij2*rij2;
					const double dr6=dr2*dr2*dr2;
					const double den=1.0/sqrt(dr6*dr6+rij6*rij6);
					const double fpair=-6.0*c6_[itype][jtype]*dr2*dr2*dr6*den*den*den;
				#else 
					#error Unsupported LDAMP exponent
				#endif
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
					#if LDAMP_A == 3
						evdwl=-c6_[itype][jtype]*den*den;
					#elif LDAMP_A == 6
						evdwl=-c6_[itype][jtype]*den;
					#elif LDAMP_A == 12
						evdwl=-c6_[itype][jtype]*den;
					#else 
						#error Unsupported LDAMP exponent
					#endif
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

void PairLDampCut::allocate(){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::allocate():\n";
	allocated=1;
	const int n=atom->ntypes+1;
	memory->create(setflag,n,n,"pair:setflag");
	for (int i=1; i<n; ++i){
		for (int j=i; j<n; ++j){
			setflag[i][j]=0;
		}
	}
	memory->create(cutsq, n, n, "pair:cutsq");
	memory->create(c6_, n, n, "pair:c6");
	memory->create(rvdw_, n, n, "pair:rvdw");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLDampCut::settings(int narg, char **arg){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::settings(int,char**):\n";
	if (narg != 1) error->all(FLERR, "Illegal pair_style command");
	rc_=utils::numeric(FLERR,arg[0],false,lmp);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLDampCut::coeff(int narg, char **arg){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::coeff(int,char**):\n";
	if(narg!=4) error->all(FLERR, "Incorrect args for pair coefficients");
	if(!allocated) allocate();
	
	int ilo,ihi,jlo,jhi;
	utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
	utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
	
	const double c6=utils::numeric(FLERR,arg[2],false,lmp);
	const double rvdw=utils::numeric(FLERR,arg[3],false,lmp);
	
	int count=0;
	for(int i=ilo; i<=ihi; i++){
		for(int j=MAX(jlo,i); j<=jhi; j++){
			c6_[i][j]=c6;
			rvdw_[i][j]=rvdw;
			setflag[i][j]=1;
			count++;
		}
	}

	if(count==0) error->all(FLERR, "Incorrect args for pair coefficients");

}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLDampCut::init_style(){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::init_style():\n";
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

double PairLDampCut::init_one(int i, int j){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::init_one(int,int):\n";
	if (setflag[i][j]==0) {
		c6_[i][j]=sqrt(c6_[i][i]*c6_[j][j]);
		rvdw_[i][j]=0.5*(rvdw_[i][i]+rvdw_[j][j]);
	}
	
	c6_[j][i]=c6_[i][j];
	rvdw_[j][i]=rvdw_[i][j];
	
	if (tail_flag) {
		
	}
	
	return rc_;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLDampCut::write_restart(FILE *fp){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::write_restart(FILE*):\n";
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

void PairLDampCut::read_restart(FILE *fp){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::read_restart(FILE*):\n";
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

void PairLDampCut::write_restart_settings(FILE *fp){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::write_restart_settings(FILE*):\n";
	fwrite(&rc_, sizeof(double), 1, fp);
	fwrite(&mix_flag, sizeof(int), 1, fp);
	fwrite(&tail_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLDampCut::read_restart_settings(FILE *fp){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::read_restart_settings(FILE*):\n";
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

void PairLDampCut::write_data(FILE *fp){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::write_data(FILE*):\n";
	for (int i=1; i<=atom->ntypes; i++) fprintf(fp,"%d %g %g\n",i,c6_[i][i],rvdw_[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLDampCut::write_data_all(FILE *fp){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::write_data_all(FILE*):\n";
	for (int i=1; i<=atom->ntypes; i++){
		for (int j=i; j<=atom->ntypes; j++){
			fprintf(fp,"%d %d %g %g\n",i,j,c6_[i][j],rvdw_[i][j]);
		}
	}
}

/* ---------------------------------------------------------------------- */

double PairLDampCut::single(int i, int j, int itype, int jtype, double rsq, double factor_coul, double factor_lj, double &fforce){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::single(int,int,int,int,rsq,double,double,double&):\n";
	const double rij=rvdw_[itype][jtype];
	#if LDAMP_A == 3
		const double dr=sqrt(rsq);
		const double dr3=rsq*dr;
		const double rij3=rij*rij*rij;
		const double den=1.0/(dr3+rij3);
		fforce=-6.0*c6_[itype][jtype]*dr*den*den*den;
		return -c6_[itype][jtype]*den*den;
	#elif LDAMP_A == 6
		//a=6,b=1
		const double rij2=rij*rij;
		const double rij6=rij2*rij2*rij2;
		const double dr6=rsq*rsq*rsq;
		const double den=1.0/(dr6+rij6);
		fforce=-6.0*c6_[itype][jtype]*rsq*rsq*den*den;
		return -c6_[itype][jtype]*den;
	#elif LDAMP_A == 12
		//a=12,b=1/2
		const double rij2=rij*rij;
		const double rij6=rij2*rij2*rij2;
		const double dr6=rsq*rsq*rsq;
		const double den=1.0/sqrt(dr6*dr6+rij6*rij6);
		fforce=-6.0*c6_[itype][jtype]*rsq*rsq*dr6*den*den*den;
		return -c6_[itype][jtype]*den;
	#else 
		#error Unsupported LDAMP exponent
	#endif
}

/* ---------------------------------------------------------------------- */

void *PairLDampCut::extract(const char *str, int &dim){
	if(PAIR_LDAMP_PRINT_FUNC>0) std::cout<<"PairLDampCut::extract(const char*,int&):\n";
	dim=2;
	if(strcmp(str,"c6")==0) return (void *)c6_;
	if(strcmp(str,"rvdw")==0) return (void *)rvdw_;
	return nullptr;
}
