/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

//c libraries
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//c++ libraries
#include <iostream>
// lammps libraries
#include "pair_nn.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include "output.h"
//ann - cutoff
#include "ann_cutoff.h"
// ann - chemical info
#include "ann_ptable.h"
// ann - serialization
#include "ann_serialize.h"
// ann - string
#include "ann_string.h"
#include "ann_token.h"

using namespace LAMMPS_NS;
using namespace MathConst;

//==========================================================================

PairNN::PairNN(LAMMPS *lmp):Pair(lmp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::PairNN(LAMMPS):\n";
	writedata=1;//write coefficients to data file
	//set defaults
	rc_=0;
	nspecies_=0;
	//flags
	single_enable=0;//no single() function
	manybody_flag=1;//many-body potential
	unit_convert_flag=0;//the units must match the training procedure
	reinitflag=0;//whether to be used with fix adapt and alike
	restartinfo=1;//writes restart info
}

//==========================================================================

PairNN::~PairNN(){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::~PairNN():\n";
	if(allocated){
		//global pair data
		memory->destroy(setflag);
		memory->destroy(cutsq);
		//==== neural network hamiltonians ====
		nspecies_=0;
		map4type2nnp_.clear();
		nnh_.clear();
		dOdZ_.clear();
		//==== symmetry functions ====
		symm_.clear();
	}
}

//==========================================================================

void PairNN::compute(int eflag, int vflag){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::compute(int,int):\n";
	//======== local variables ========
	//atom properties - global
	double** x = atom->x;//positions
	double** f = atom->f;//forces
	const int* type = atom->type;//types
	const int nlocal = atom->nlocal;//number of local atoms
	const bool newton_pair=force->newton_pair;//must be true (see init_style)
	double etotal_=0;//total energy (local to processor)
	//atom properties - local
	std::vector<double> eatoml_(nlocal);//atomic energy
	//neighbors - ith atom
	const int inum = list->inum;// # of ith atom's neighbors
	const int* index_list = list->ilist;// local indices of I atoms
	const int* numneigh = list->numneigh;// # of J neighbors for each I atom
	int** firstneigh = list->firstneigh;// ptr to 1st J int value of each I atom
	
	//======== set energy/force/virial calculations ========
	if (eflag || vflag) ev_setup(eflag,vflag);
	else evflag = vflag_fdotr = 0;
	
	//======== compute energy and forces ========
	for(int ii=0; ii<inum; ++ii){
		//==== get the index of type of i ====
		const int i=index_list[ii];//get the index
		const int II=map4type2nnp_[type[i]-1];//get the index in the NNP
		if(II<0) continue; //skip if current type is not included in the NNP
		
		//==== get the nearest neighbors of i (full list) ====
		const Eigen::Vector3d rI=(Eigen::Vector3d()<<x[i][0],x[i][1],x[i][2]).finished();
		const int* nn_list=firstneigh[i];//get the list of nearest neighbors
		const int num_nn=numneigh[i];//get the number of neighbors
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"atomi "<<type[i]<<" "<<i<<"\n";
		
		//==== compute the symmetry function ====
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"computing symmetry function\n";
		//reset the symmetry function
		symm_[II].setZero();
		//loop over all pairs
		for(int jj=0; jj<num_nn; ++jj){
			//get the index of type of j
			const int j=nn_list[jj]&NEIGHMASK;//get the index, clear two highest bits
			const int JJ=map4type2nnp_[type[j]-1];//get the index in the NNP
			if(JJ<0) continue; //skip if current type is not included in the NNP
			if(PAIR_NN_PRINT_DATA>2) std::cout<<"\tatomj "<<type[j]<<" "<<j<<"\n";
			//compute rIJ
			const Eigen::Vector3d rIJ=(Eigen::Vector3d()<<rI[0]-x[j][0],rI[1]-x[j][1],rI[2]-x[j][2]).finished();
			const double dIJ2=rIJ.squaredNorm();//compute norm
			if(dIJ2<rc2_){
				const double dIJ=sqrt(dIJ2);
				//compute the IJ contribution to all radial basis functions
				const int offsetR_=nnh_[II].offsetR(JJ);//input vector offset
				nnh_[II].basisR(JJ).symm(dIJ,symm_[II].data()+offsetR_);
				//loop over all unique triples
				for(int kk=jj+1; kk<num_nn; ++kk){
					const int k=nn_list[kk]&NEIGHMASK;//get the index, clear two highest bits
					const int KK=map4type2nnp_[type[k]-1];//get the index in the NNP
					if(KK<0) continue; //skip if current type is not included in the NNP
					if(PAIR_NN_PRINT_DATA>3) std::cout<<"\t\tatomk "<<type[k]<<" "<<k<<"\n";
					//compute dIK
					const Eigen::Vector3d rIK=(Eigen::Vector3d()<<rI[0]-x[k][0],rI[1]-x[k][1],rI[2]-x[k][2]).finished();
					const double dIK2=rIK.squaredNorm();//compute norm
					if(dIK2<rc2_){
						const double dIK=sqrt(dIK2);
						//compute the IJ,IK,JK contribution to all angular basis functions
						if(PAIR_NN_PRINT_DATA>4) std::cout<<"\t\t\ti j k "<<index_list[ii]<<" "<<index_list[jj]<<" "<<index_list[kk]<<"\n";
						const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);//input vector offset
						const double cosIJK=rIJ.dot(rIK)/(dIJ*dIK);//cosine of ijk interior angle - i at vertex
						const double d[2]={dIJ,dIK};//utility vector (reduces number of function arguments)
						nnh_[II].basisA(JJ,KK).symm(d,cosIJK,symm_[II].data()+offsetA_);
					}
				}
			}
		}
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"symm["<<i<<"] = "<<symm_[II].transpose()<<"\n";
		
		//==== compute the energy ====
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"computing energy\n";
		//execute the network
		nnh_[II].nn().execute(symm_[II]);
		//accumulate the energy
		//const double eatom_=nnh_[II].nn().out()[0]+nnh_[II].type().energy().val();//local energy + intrinsic energy
		const double eatom_=nnh_[II].nn().out()[0];
		etotal_+=eatom_;
		eatoml_[i]=eatom_;
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"energy-atom["<<i<<"] "<<eatom_<<"\n";
		
		//==== compute the network gradients ====
		//dodi() - do/di - derivative of output w.r.t. input
		//row(0) - derivative of zeroth output node (note: only one output node by definition)
		dOdZ_[II].grad(nnh_[II].nn());
		const Eigen::VectorXd& dEdG=dOdZ_[II].dodi().row(0);//dEdG - dE/dG - gradient of energy (E) w.r.t. nn inputs (G)
		
		//==== compute forces ====
		for(int jj=0; jj<num_nn; ++jj){
			//get the index of type of j
			const int j=nn_list[jj]&NEIGHMASK;//get the index, clear two highest bits
			const int JJ=map4type2nnp_[type[j]-1];//get the index in the NNP
			if(JJ<0) continue; //skip if current type is not included in the NNP
			if(PAIR_NN_PRINT_DATA>1) std::cout<<"\tatom "<<type[j]<<" "<<j<<"\n";
			//compute rIJ
			const Eigen::Vector3d rIJ=(Eigen::Vector3d()<<rI[0]-x[j][0],rI[1]-x[j][1],rI[2]-x[j][2]).finished();
			const double dIJ2=rIJ.squaredNorm();//compute norm
			if(dIJ2<rc2_){
				const double dIJ=sqrt(dIJ2);
				const double dIJi=1.0/dIJ;
				//compute the IJ contribution to the force
				const double fpair=nnh_[II].basisR(JJ).force(
					dIJ,dEdG.data()+nnh_[II].offsetR(JJ)
				)*dIJi;
				f[i][0]+=fpair*rIJ[0]; f[i][1]+=fpair*rIJ[1]; f[i][2]+=fpair*rIJ[2];
				f[j][0]-=fpair*rIJ[0]; f[j][1]-=fpair*rIJ[1]; f[j][2]-=fpair*rIJ[2];
				//loop over all unique triplets
				for(int kk=jj+1; kk<num_nn; ++kk){
					//get the index and type of k
					const int k=nn_list[kk]&NEIGHMASK;//get the index, clear two highest bits
					const int KK=map4type2nnp_[type[k]-1];//get the index in the NNP
					if(KK<0) continue; //skip if current type is not included in the NNP
					if(PAIR_NN_PRINT_DATA>2) std::cout<<"\t\tatom "<<type[k]<<" "<<k<<"\n";
					//compute dIK
					const Eigen::Vector3d rIK=(Eigen::Vector3d()<<rI[0]-x[k][0],rI[1]-x[k][1],rI[2]-x[k][2]).finished();
					const double dIK2=rIK.squaredNorm();//compute norm
					if(dIK2<rc2_){
						const double dIK=sqrt(dIK2);
						const double dIKi=1.0/dIK;
						//compute the IJK contribution to the force
						const double cosIJK=rIJ.dot(rIK)*dIJi*dIKi;//cosine of (i,j,k) angle (i at vertex)
						const double d[2]={dIJ,dIK};//utility array to reduce number of function arguments
						double phi=0; double eta[2]={0,0};//force constants
						nnh_[II].basisA(JJ,KK).force(
							d,cosIJK,phi,eta,dEdG.data()+nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK)
						);
						//compute force on i
						const Eigen::Vector3d ffi=
							 (phi*(dIKi-cosIJK*dIJi)+eta[0])*rIJ*dIJi //compute force on i due to j
							+(phi*(dIJi-cosIJK*dIKi)+eta[1])*rIK*dIKi;//compute force on i due to k
						//compute force on j
						//const Eigen::Vector3d ffj=-(-phi*cosIJK*dIJi+eta[0])*rIJ*dIJi-phi*dIJi*rIK*dIKi-eta[2]*rJK*dJKi;
						const Eigen::Vector3d ffj=-(-phi*cosIJK*dIJi+eta[0])*rIJ*dIJi-phi*dIJi*rIK*dIKi;
						//compute force on k
						//const Eigen::Vector3d ffk=-(-phi*cosIJK*dIKi+eta[1])*rIK*dIKi-phi*dIKi*rIJ*dIJi+eta[2]*rJK*dJKi;
						const Eigen::Vector3d ffk=-(-phi*cosIJK*dIKi+eta[1])*rIK*dIKi-phi*dIKi*rIJ*dIJi;
						//write forces
						f[i][0]+=ffi[0]; f[i][1]+=ffi[1]; f[i][2]+=ffi[2];//write force on i
						f[j][0]+=ffj[0]; f[j][1]+=ffj[1]; f[j][2]+=ffj[2];//write force on j
						f[k][0]+=ffk[0]; f[k][1]+=ffk[1]; f[k][2]+=ffk[2];//write force on k
					}
				}
			}
		}
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"force["<<i<<"] "<<f[i][0]<<" "<<f[i][1]<<" "<<f[i][2]<<"\n";
	}
	
	//======== tally energy ========
	if(eflag_global) ev_tally(0,0,nlocal,newton_pair,etotal_,0.0,0.0,0.0,0.0,0.0);
	if(eflag_atom) for(int i=0; i<nlocal; ++i) eatom[i]=eatoml_[i];
	
	//======== compute virial ========
	if(vflag_fdotr) virial_fdotr_compute();
	
}

//----------------------------------------------------------------------
// allocate all arrays
//----------------------------------------------------------------------

void PairNN::allocate(){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::allocate():\n";
	//==== set variables ====
	allocated=1;//flag as allocated
	const int ntypes=atom->ntypes;//the number of types in the simulation
	if(PAIR_NN_PRINT_DATA>0) std::cout<<"ntypes = "<<ntypes<<"\n";
	//==== global pair data ====
	memory->create(cutsq,ntypes+1,ntypes+1,"pair:cutsq");
	memory->create(setflag,ntypes+1,ntypes+1,"pair:setflag");
	//==== neural network hamiltonians ====
	map4type2nnp_.resize(ntypes);
	//==== set flags ====
	for(int i=1; i<=ntypes; ++i){
		for(int j=1; j<ntypes; ++j){
			setflag[i][j]=0;
		}
	}
}

//----------------------------------------------------------------------
// global settings
//----------------------------------------------------------------------

void PairNN::settings(int narg, char **arg){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::settings(int,char**):\n";
	//==== local variables ====
	const int me=comm->me;
	//==== check arguments ====
	if(narg!=1) error->all(FLERR,"Illegal pair_style command");//cutoff
	//==== set the global cutoff ====
	rc_=utils::numeric(FLERR,arg[0],false,lmp);
	rc2_=rc_*rc_;
}

//----------------------------------------------------------------------
// set coeffs for one or more type pairs
//----------------------------------------------------------------------

void PairNN::coeff(int narg, char **arg){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::coeff(int,char**):\n";
	//pair_coeff * * nn_pot X Y Z
	//==== local variables ====
	const int me = comm->me;
	const int ntypes = atom->ntypes;
	//==== read pair coeffs ====
	//check nargs
	if(narg!=3+atom->ntypes) error->all(FLERR,"Number of species in nn coefficient line does not match the number of types.");
	if(!allocated) allocate();
	//ensure I,J args are both *
	if(strcmp(arg[0],"*")!=0 || strcmp(arg[1],"*")!=0) error->all(FLERR,"Coefficient types must be wildcards.");
	if(me==0){
		//==== read the potential ====
		read_pot(arg[2]);
		//==== read atom names/ids ====
		std::vector<std::string> names(ntypes);//names provided in the input file
		std::vector<int> ids(ntypes);//unique hash id's of the atom names
		for(int i=0; i<ntypes; ++i){
			names[i]=std::string(arg[i+3]);
			ids[i]=string::hash(names[i]);
		}
		const int idNULL=string::hash("NULL");
		//==== check atom names and build the map ====
		map4type2nnp_.resize(ntypes);
		for(int i=0; i<ntypes; ++i){
			map4type2nnp_[i]=-1;
			for(int j=0; j<nspecies_; ++j){
				if(ids[i]==nnh_[j].type().id()){
					map4type2nnp_[i]=j; break;
				}
			}
			if(ids[i]!=idNULL && map4type2nnp_[i]<0) error->all(FLERR,"Could not find atom name in NNP");
		}
		if(PAIR_NN_PRINT_DATA>-1){
			std::cout<<"*******************************************\n";
			std::cout<<"*************** SPECIES MAP ***************\n";
			std::cout<<"name type-lammps type-nnp\n";
			for(int i=0; i<map4type2nnp_.size(); ++i){
				if(map4type2nnp_[i]>=0){
					std::cout<<nnh_[map4type2nnp_[i]].type().name()<<" "<<i+1<<" "<<map4type2nnp_[i]<<"\n";
				} else {
					std::cout<<"NULL "<<i+1<<" "<<map4type2nnp_[i]<<"\n";
				}
			}
			std::cout<<"*******************************************\n";
			for(int i=0; i<nnh_.size(); ++i){
				std::cout<<"NNH "<<nnh_[i].type().name()<<"\n";
				std::cout<<nnh_[i]<<"\n";
			}
		}
	}
	//all flags are set since "coeff" only set once
	for (int i=1; i<=ntypes; ++i){
		for (int j=1; j<=ntypes; ++j){
			setflag[i][j]=1;
		}
	}
}

//----------------------------------------------------------------------
// init specific to this pair style
//----------------------------------------------------------------------

void PairNN::init_style(){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::init_style():\n";
	//==== local variables ====
	const int ntypes=atom->ntypes;
	const int me=comm->me;
	//==== flags ====
	if(atom->tag_enable==0) error->all(FLERR,"Pair style NN requires atom IDs");
	if(force->newton_pair==0) error->all(FLERR,"Pair style NN requires newton pair on");
	/*
		Note: calculating forces is relatively expensive for ann's.  Thus, it is best to have newton_pair
		turned on.  As newton_pair on/off requires completely different algorithms and code, and as
		we have chosen to have newton_pair on, we enforce that newton_pair is on.
	*/
	//==== need a full neighbor list ====
	/*
	const int irequest=neighbor->request(this,instance_me);
		neighbor->requests[irequest]->half=0;//disable half-neighbor list
		neighbor->requests[irequest]->full=1;//enable full-neighbor list
	*/
	
	neighbor->add_request(this, NeighConst::REQ_FULL);
	
	//==== broadcast data ====
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"b_casting data\n";
	//==== broadcast species/types ====
	MPI_Bcast(&nspecies_,1,MPI_INT,0,world);
	MPI_Bcast(map4type2nnp_.data(),ntypes,MPI_INT,0,world);
	MPI_Barrier(world);
	//==== resize/broadcast nnh ====
	MPI_Barrier(world);
	nnh_.resize(nspecies_);
	for(int n=0; n<nspecies_; ++n){
		int nBytes=0;
		if(me==0) nBytes=serialize::nbytes(nnh_[n]);
		MPI_Bcast(&nBytes,1,MPI_INT,0,world);
		char* arr=new char[nBytes];
		if(me==0) serialize::pack(nnh_[n],arr);
		MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
		if(me!=0) serialize::unpack(nnh_[n],arr);
		delete[] arr;
		MPI_Barrier(world);
	}
	//==== resize utility arrays ====
	symm_.resize(nspecies_);
	dOdZ_.resize(nspecies_);
	for(int n=0; n<nspecies_; ++n){
		symm_[n]=Eigen::VectorXd::Zero(nnh_[n].nn().nInp());
		dOdZ_[n].resize(nnh_[n].nn());
	}
	//==== mpi barrier ====
	MPI_Barrier(world);
}

//----------------------------------------------------------------------
// init for one type pair i,j and corresponding j,i
//----------------------------------------------------------------------

double PairNN::init_one(int i, int j){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::init_one(int,int):\n";
	return rc_;
}

//----------------------------------------------------------------------
// proc 0 writes to restart file
//----------------------------------------------------------------------

void PairNN::write_restart(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::write_restart(FILE*):\n";
	write_restart_settings(fp);
	const int ntypes=atom->ntypes;
	//==== write flags ====
	for(int i=1; i<=ntypes;++i){
		for(int j=1; j<=ntypes; ++j){
			fwrite(&setflag[i][j],sizeof(int),1,fp);
		}
	}
	//==== write species ====
	fwrite(&nspecies_,sizeof(int),1,fp);
	fwrite(map4type2nnp_.data(),sizeof(int),ntypes,fp);
	//==== loop over all types ====
	for(int n=0; n<nspecies_; ++n){
		const int size=serialize::nbytes(nnh_[n]);
		char* arr=new char[size];
		serialize::pack(nnh_[n],arr);
		//write size (bytes)
		fwrite(&size,sizeof(int),1,fp);
		//write object
		fwrite(arr,sizeof(char),size,fp);
		//free memory
		delete[] arr;
	}
}

//----------------------------------------------------------------------
// proc 0 reads from restart file, bcasts
//----------------------------------------------------------------------

void PairNN::read_restart(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::read_restart(FILE*):\n";
	read_restart_settings(fp);
	allocate();
	int nbytes=0;
	const int ntypes=atom->ntypes;
	const int me = comm->me;
	map4type2nnp_.resize(ntypes);
	//======== proc 0 reads from restart file ========
	if(me==0){
		for(int i=1; i<=ntypes;++i){
			for(int j=1; j<=ntypes; ++j){
				nbytes=fread(&setflag[i][j],sizeof(int),1,fp);
			}
		}
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"ntypes = "<<ntypes<<"\n";
		//==== species ====
		nbytes=fread(&nspecies_,sizeof(int),1,fp);
		nbytes=fread(map4type2nnp_.data(),sizeof(int),ntypes,fp);
		//==== loop over all species ====
		nnh_.resize(nspecies_);
		for(int n=0; n<nspecies_; ++n){
			//read size
			int size=0;
			nbytes=fread(&size,sizeof(int),1,fp);
			//read data
			char* arr=new char[size];
			nbytes=fread(arr,sizeof(char),size,fp);
			//unpack object
			serialize::unpack(nnh_[n],arr);
			//free memory
			delete[] arr;
		}
	}
	for(int i=1; i<=ntypes; ++i){
		for(int j=1; j<=ntypes; ++j){
			MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
		}
	}
}

//----------------------------------------------------------------------
// proc 0 writes to restart file
//----------------------------------------------------------------------

void PairNN::write_restart_settings(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::write_restart_settings(FILE*):\n";
	const int me=comm->me;
	//==== write cutoff ====
	if(me==0) fwrite(&rc_,sizeof(double),1,fp);
}

//----------------------------------------------------------------------
// proc 0 reads from restart file, bcasts
//----------------------------------------------------------------------

void PairNN::read_restart_settings(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::read_restart_settings(FILE*):\n";
	const int me=comm->me;
	int nbytes=0;
	//==== read cutoff ====
	if(me==0) nbytes=fread(&rc_,sizeof(double),1,fp);
	//==== bcast cutoff ====
	MPI_Bcast(&rc_,1,MPI_DOUBLE,0,world);
}

//----------------------------------------------------------------------
// proc 0 writes to data file
//----------------------------------------------------------------------

void PairNN::write_data(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::write_data(FILE*):\n";
}

//----------------------------------------------------------------------
// proc 0 writes all pairs to data file
//----------------------------------------------------------------------

void PairNN::write_data_all(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::write_data_all(FILE*):\n";
}

//==========================================================================

//read neural network potential file
void PairNN::read_pot(const char* file){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::read_pot(const char*):\n";
	if(PAIR_NN_PRINT_DATA>0) std::cout<<"file_ann = "<<file<<"\n";
	//==== local function variables ====
	char* input=new char[string::M];
	FILE* reader=NULL;//file pointer for reading
	Token token;
	//==== open the potential file ====
	reader=fopen(file,"r");
	if(reader==NULL) error->all(FLERR,"PairNN::read_pot(const char*): Could not open neural network potential file.");
	//==== header ====
	fgets(input,string::M,reader);
	//==== number of species ====
	token.read(fgets(input,string::M,reader),string::WS); token.next();
	nspecies_=std::atoi(token.next().c_str());
	if(nspecies_<=0) error->all(FLERR,"PairNN::read_pot(const char*): invalid number of species.");
	//==== species ====
	std::vector<Type> species(nspecies_);
	Map<int,int> map_;
	nnh_.resize(nspecies_);
	for(int n=0; n<nspecies_; ++n){
		Type::read(fgets(input,string::M,reader),species[n]);
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"species["<<n<<"] = "<<species[n]<<"\n";
		nnh_[n].resize(nspecies_);
		nnh_[n].type()=species[n];
		map_.add(string::hash(nnh_[n].type().name()),n);
	}
	//==== global cutoff ====
	token.read(fgets(input,string::M,reader),string::WS); token.next();
	const double rc=std::atof(token.next().c_str());
	if(rc!=rc_) error->all(FLERR,"PairNN::read_pot(const char*): invalid cutoff.");
	//==== basis ====
	for(int i=0; i<nspecies_; ++i){
		//read central species
		token.read(fgets(input,string::M,reader),string::WS); token.next();
		const int II=map_[string::hash(token.next())];
		//read basis - radial
		for(int j=0; j<nspecies_; ++j){
			//read species
			token.read(fgets(input,string::M,reader),string::WS); token.next();
			const int JJ=map_[string::hash(token.next())];
			//read basis
			BasisR::read(reader,nnh_[II].basisR(JJ));
			if(PAIR_NN_PRINT_DATA>1) std::cout<<"BasisR("<<nnh_[II].type().name()<<","<<nnh_[JJ].type().name()<<") = "<<nnh_[II].basisR(JJ)<<"\n";
		}
		//read basis - angular
		for(int j=0; j<nspecies_; ++j){
			for(int k=j; k<nspecies_; ++k){
				//read species
				token.read(fgets(input,string::M,reader),string::WS); token.next();
				const int JJ=map_[string::hash(token.next())];
				const int KK=map_[string::hash(token.next())];
				//read basis
				BasisA::read(reader,nnh_[II].basisA(JJ,KK));
				if(PAIR_NN_PRINT_DATA>1) std::cout<<"BasisA("<<nnh_[II].type().name()<<","<<nnh_[JJ].type().name()<<","<<nnh_[KK].type().name()<<") = "<<nnh_[II].basisA(JJ,KK)<<"\n";
			}
		}
		//initialize the inputs
		nnh_[II].init_input();
	}
	//==== neural network ====
	for(int n=0; n<nspecies_; ++n){
		//read species
		token.read(fgets(input,string::M,reader),string::WS); token.next();
		const int II=map_[string::hash(token.next())];
		//read network
		NN::ANN::read(reader,nnh_[II].nn());
	}
	//==== free local variables ====
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"freeing local variables\n";
	delete[] input;
	if(reader!=NULL) fclose(reader);
}

//==========================================================================

void *PairNN::extract(const char *str, int &dim){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::extract(const char*,int&):\n";
	dim=0;//global cutoff is the only "simple" parameter
	return NULL;
}
