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

#include "pair_ldamp_long.h"

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

PairLDampLong::PairLDampLong(LAMMPS *lmp) : Pair(lmp){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::PairLDampLong(LAMMPS*):\n";
	writedata = 1;
	ewaldflag = 1;
	pppmflag = 1;
	dispersionflag = 1;
	c6_=nullptr;
	rvdw_=nullptr;
	rvdw3_=nullptr;
	rvdw6_=nullptr;
	ewald_order = 0;
	ewald_off = 0;
	mix_flag=GEOMETRIC;//by definition
}

/* ---------------------------------------------------------------------- */

PairLDampLong::~PairLDampLong(){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::~PairLDampLong():\n";
	if (copymode) return;
	if (allocated) {
		//global pair data
		memory->destroy(setflag);
		memory->destroy(cutsq);
		memory->destroy(c6_);
		memory->destroy(rvdw_);
		memory->destroy(rvdw3_);
		memory->destroy(rvdw6_);
	}
}

/* ---------------------------------------------------------------------- */

void PairLDampLong::compute(int eflag, int vflag){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::compute(int,int):\n";
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
				const double dr6=dr2*dr2*dr2;
				const double rij6=rvdw6_[itype][jtype];
				const double b2=dr2*ge62_;
				const double expf=exp(-b2);
				#if LDAMP_A == 3
					//a=3,b=2
					const double dr=sqrt(dr2);
					const double dr3=dr2*dr;
					const double den=1.0/(dr3+rvdw3_[itype][jtype]);
					const double fpair=c6_[itype][jtype]*(
						(6.0-expf*(6.0+b2*(6.0+b2*(3.0+b2))))/(dr6*dr2)
						-6.0*dr*den*den*den
					);
				#elif LDAMP_A == 6
					//a=6,b=1
					const double den=1.0/(dr6+rij6);
					const double fpair=c6_[itype][jtype]*(
						(6.0-expf*(6.0+b2*(6.0+b2*(3.0+b2))))/(dr6*dr2)
						-6.0*dr2*dr2*den*den
					);
				#elif LDAMP_A == 12
					//a=12,b=1/2
					const double den=1.0/sqrt(dr6*dr6+rij6*rij6);
					const double fpair=c6_[itype][jtype]*(
						(6.0-expf*(6.0+b2*(6.0+b2*(3.0+b2))))/(dr6*dr2)
						-6.0*dr2*dr2*dr6*den*den*den
					);
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
						evdwl=c6_[itype][jtype]*(
							(1.0-expf*(1.0+b2*(1.0+0.5*b2)))/dr6
							-den*den
						);
					#elif LDAMP_A == 6
						evdwl=c6_[itype][jtype]*(
							(1.0-expf*(1.0+b2*(1.0+0.5*b2)))/dr6
							-den
						);
					#elif LDAMP_A == 12
						evdwl=c6_[itype][jtype]*(
							(1.0-expf*(1.0+b2*(1.0+0.5*b2)))/dr6
							-den
						);
					#else 
						#error Unsupported LDAMP exponent
					#endif
				}
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

void PairLDampLong::allocate(){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::allocate():\n";
	allocated=1;
	const int n=atom->ntypes+1;
	memory->create(setflag,n,n,"pair:setflag");
	for (int i=1; i<n; i++){
		for (int j=i; j<n; j++){
			setflag[i][j]=0;
		}
	}
	memory->create(cutsq, n, n, "pair:cutsq");
	memory->create(c6_, n, n, "pair:c6");
	memory->create(rvdw_, n, n, "pair:rvdw");
	memory->create(rvdw3_, n, n, "pair:rvdw3");
	memory->create(rvdw6_, n, n, "pair:rvdw6");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

/*
	ewald options read
	copied from lj/long/coul/long
*/
void PairLDampLong::options(char **arg, int order){
	const char *option[] = {"long", "cut", "off", nullptr};
	int i;
	if (!*arg) error->all(FLERR,"Illegal pair_style ldamp/long command");
	for (i=0; option[i]&&strcmp(arg[0], option[i]); ++i);
	switch (i) {
		case 0: ewald_order |= 1<<order; break;
		case 2: ewald_off |= 1<<order; break;
		case 1: break;
		default: error->all(FLERR,"Illegal pair_style ldamp/long command");
	}
}

void PairLDampLong::settings(int narg, char **arg){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::settings(int,char**):\n";
	if (narg != 1) error->all(FLERR, "Illegal pair_style command");
	rc_=utils::numeric(FLERR,arg[0],false,lmp);
	
	//set ewald order, needed for compatibility reasons
	/*
		set the ewald order options
		this includes the "order" and "off" options for ewald simulations
		needed by k-space solvers like ewald/disp
	*/
	//initalize variables
	ewald_order = 0;
	ewald_off = 0;
	//set strings that normally would be read in the input file
	char** args=new char*[2];
	args[0]=new char[10];
	args[1]=new char[10];
	std::strcpy(args[0],"long");//option - kspace - lj
	std::strcpy(args[1],"off");//option - kspace - coul
	//load the options
	options(args,6);//read - option - kspace - lj - order=6 (1/r**6)
	options(args+1,1);//read - option - kspace - coul - order=6 (1/r**1)
	//delete strings
	delete[] args[0];
	delete[] args[1];
	delete[] args;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLDampLong::coeff(int narg, char **arg){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::coeff(int,char**):\n";
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
			rvdw3_[i][j]=rvdw*rvdw*rvdw;
			rvdw6_[i][j]=rvdw3_[i][j]*rvdw3_[i][j];
			setflag[i][j]=1;
			count++;
		}
	}

	if(count==0) error->all(FLERR, "Incorrect args for pair coefficients");

}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLDampLong::init_style(){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::init_style():\n";
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
	
	//neighbor->add_request(this);
	neighbor->add_request(this, NeighConst::REQ_DEFAULT);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLDampLong::init_one(int i, int j){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::init_one(int,int):\n";
	if (setflag[i][j]==0) {
		c6_[i][j]=sqrt(c6_[i][i]*c6_[j][j]);
		rvdw_[i][j]=0.5*(rvdw_[i][i]+rvdw_[j][j]);
	}
	rvdw3_[i][j]=rvdw_[i][j]*rvdw_[i][j]*rvdw_[i][j];
	rvdw6_[i][j]=rvdw3_[i][j]*rvdw3_[i][j];
	cutsq[i][j] = rc_*rc_;
	
	c6_[j][i]=c6_[i][j];
	rvdw_[j][i]=rvdw_[i][j];
	rvdw3_[j][i]=rvdw3_[i][j];
	rvdw6_[j][i]=rvdw6_[i][j];
	cutsq[j][i]=cutsq[i][j];
	
	if (tail_flag) {
		
	}
	
	return rc_;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLDampLong::write_restart(FILE *fp){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::write_restart(FILE*):\n";
	write_restart_settings(fp);
	for(int i=1; i<=atom->ntypes; i++){
		for(int j=i; j<=atom->ntypes; j++) {
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

void PairLDampLong::read_restart(FILE *fp){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::read_restart(FILE*):\n";
	read_restart_settings(fp);
	allocate();
	const int me = comm->me;
	for (int i=1; i<=atom->ntypes; i++){
		for (int j=i; j<=atom->ntypes; j++) {
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

void PairLDampLong::write_restart_settings(FILE *fp){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::write_restart_settings(FILE*):\n";
	fwrite(&rc_, sizeof(double), 1, fp);
	fwrite(&mix_flag, sizeof(int), 1, fp);
	fwrite(&tail_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLDampLong::read_restart_settings(FILE *fp){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::read_restart_settings(FILE*):\n";
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

void PairLDampLong::write_data(FILE *fp){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::write_data(FILE*):\n";
	for (int i=1; i<=atom->ntypes; i++) fprintf(fp,"%d %g %g\n",i,c6_[i][i],rvdw_[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLDampLong::write_data_all(FILE *fp){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::write_data_all(FILE*):\n";
	for (int i=1; i<=atom->ntypes; i++){
		for (int j=i; j<=atom->ntypes; j++){
			fprintf(fp,"%d %d %g %g\n",i,j,c6_[i][j],rvdw_[i][j]);
		}
	}
}

/* ---------------------------------------------------------------------- */

double PairLDampLong::single(int i, int j, int itype, int jtype, double rsq, double factor_coul, double factor_lj, double &fforce){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::single(int,int,int,int,rsq,double,double,double&):\n";
	const double dr6=rsq*rsq*rsq;
	const double b2=rsq*ge62_;
	const double expf=exp(-b2);
	#if LDAMP_A == 3
		//a=3,b=2
		const double dr=sqrt(rsq);
		const double dr3=rsq*dr;
		const double rij3=rvdw3_[itype][jtype];
		const double den=1.0/(dr3+rij3);
		fforce=c6_[itype][jtype]*((6.0-expf*(6.0+b2*(6.0+b2*(3.0+b2))))/(dr6*rsq)-6.0*dr*den*den*den);
		return c6_[itype][jtype]*((1.0-expf*(1.0+b2*(1.0+0.5*b2)))/dr6-den*den);
	#elif LDAMP_A == 6
		//a=6,b=1
		const double rij6=rvdw6_[itype][jtype];
		const double den=1.0/(dr6+rij6);
		fforce=c6_[itype][jtype]*((6.0-expf*(6.0+b2*(6.0+b2*(3.0+b2))))/(dr6*rsq)-6.0*rsq*rsq*den*den);
		return c6_[itype][jtype]*((1.0-expf*(1.0+b2*(1.0+0.5*b2)))/dr6-den);
	#elif LDAMP_A == 12
		//a=12,b=1/2
		const double rij6=rvdw6_[itype][jtype];
		const double den=1.0/sqrt(dr6*dr6+rij6*rij6);
		fforce=c6_[itype][jtype]*((6.0-expf*(6.0+b2*(6.0+b2*(3.0+b2))))/(dr6*rsq)-6.0*rsq*rsq*dr6*den*den*den);
		return c6_[itype][jtype]*((1.0-expf*(1.0+b2*(1.0+0.5*b2)))/dr6-den);
	#else 
		#error Unsupported LDAMP exponent
	#endif
}

/* ---------------------------------------------------------------------- */

void *PairLDampLong::extract(const char *str, int &dim){
	if(PAIR_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"PairLDampLong::extract(const char*,int&):\n";
	if(strcmp(str,"cut_london")==0){
		dim=0;
		return (void *) &rc_;
	}
	if(strcmp(str,"cut_coul")==0){
		dim=0;
		return (void *) &rc_;
	}
	if(strcmp(str,"cut_LJ")==0){
		dim=0;
		return (void *) &rc_;
	}
	if(strcmp(str,"c6")==0){
		dim=2;
		return (void *)c6_;
	}
	if(strcmp(str,"B")==0){
		dim=2;
		return (void *)c6_;
	}
	if(strcmp(str,"rvdw")==0){
		dim=2;
		return (void *)rvdw_;
	}
	if(strcmp(str,"ewald_mix")==0){
		dim=0;
		return (void *)&mix_flag;
	}
	if(strcmp(str,"ewald_order")==0){
		dim=0;
		return (void *)&ewald_order;
	}
	return nullptr;
}
