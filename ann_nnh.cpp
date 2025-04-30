// c libraries
#include <cstdio>
// c++ libraries
#include <iostream>
// ann - math
#include "ann_const.h"
// ann - print
#include "ann_print.h"
// ann - nnh
#include "ann_nnh.h"

//************************************************************
// NEURAL NETWORK HAMILTONIAN
//************************************************************

//==== operators ====

/**
* print neural network hamiltonian
* @param out - output stream
* @param nnh - neural network hamiltonian
*/
std::ostream& operator<<(std::ostream& out, const NNH& nnh){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NN - HAMILTONIAN",str)<<"\n";
	//hamiltonian
	out<<"TYPE     = "<<nnh.type_<<"\n";
	//types
	out<<"N_TYPES  = "<<nnh.ntypes_<<"\n";
	//potential parameters
	out<<"N_INPUT  = "; std::cout<<nnh.nInput_<<" "; std::cout<<"\n";
	out<<"N_INPUTR = "; std::cout<<nnh.nInputR_<<" "; std::cout<<"\n";
	out<<"N_INPUTA = "; std::cout<<nnh.nInputA_<<" "; std::cout<<"\n";
	out<<nnh.nn_<<"\n";
	out<<print::title("NN - HAMILTONIAN",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

/**
* set NNH defaults
*/
void NNH::defaults(){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNH::defaults()\n";
	//hamiltonian
		ntypes_=0;
		type_.clear();
		nn_.clear();
	//basis for pair/triple interactions
		basisR_.clear();
		basisA_.clear();
	//network configuration
		nInput_=0;
		nInputR_=0;
		nInputA_=0;
		offsetR_.clear();
		offsetA_.clear();
}

/**
* resize the number of types
* @param ntypes - the total number of types
*/
void NNH::resize(int ntypes){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNH::resize(int)\n";
	if(ntypes<0) throw std::invalid_argument("NNH::resize(int): invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		basisR_.resize(ntypes_);
		basisA_.resize(ntypes_);
		offsetR_.resize(ntypes_);
		offsetA_.resize(ntypes_);
	}
}

/**
* Initialize the number of inputs and offsets associated with the basis functions.
* Must be done after the basis has been defined, otherwise the values will make no sense.
* Different from resizing: resizing sets the number of species, this sets the number of inputs
* associated with the basis associated with each species.
*/
void NNH::init_input(){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNH::init_input()\n";
	//radial inputs
	nInputR_=0;
	for(int i=0; i<ntypes_; ++i){
		nInputR_+=basisR_[i].size();
	}
	//radial offsets
	for(int i=0; i<ntypes_; ++i){
		if(i==0) offsetR_[i]=0;
		else offsetR_[i]=offsetR_[i-1]+basisR_[i-1].size();
	}
	//angular inputs
	nInputA_=0;
	for(int i=0; i<ntypes_; ++i){
		for(int j=i; j<ntypes_; ++j){
			nInputA_+=basisA_(j,i).size();
		}
	}
	//angular offsets
	for(int i=0; i<basisA_.size(); ++i){
		if(i==0) offsetA_[i]=0;
		else offsetA_[i]=offsetA_[i-1]+basisA_[i-1].size();
	}
	//total number of inputs
	nInput_=nInputR_+nInputA_;
}

/**
* compute energy of atom with symmetry function "symm"
* @param symm - the symmetry function
*/
double NNH::energy(const Eigen::VectorXd& symm){
	if(NNP_PRINT_FUNC>0) std::cout<<"NNH::energy(const Eigen::VectorXd&)\n";
	return nn_.execute(symm)[0]+type_.energy().val();
}

//************************************************************
// serialization
//************************************************************

namespace serialize{
	
//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const NNH& obj){
	if(NNP_PRINT_FUNC>0) std::cout<<"nbytes(const NNH&):\n";
	int size=0;
	//hamiltonian
	size+=nbytes(obj.type());
	size+=nbytes(obj.nn());
	//species
	size+=nbytes(obj.ntypes());//ntypes_
	//basis for pair/triple interactions
	for(int j=0; j<obj.ntypes(); ++j){
		size+=nbytes(obj.basisR(j));
	}
	for(int j=0; j<obj.ntypes(); ++j){
		for(int k=j; k<obj.ntypes(); ++k){
			size+=nbytes(obj.basisA(j,k));
		}
	}
	//return the size
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const NNH& obj, char* arr){
	if(NNP_PRINT_FUNC>0) std::cout<<"pack(const NNH&,char*):\n";
	int pos=0;
	//hamiltonian
	pos+=pack(obj.type(),arr+pos);
	pos+=pack(obj.nn(),arr+pos);
	//species
	pos+=pack(obj.ntypes(),arr+pos);
	//basis for pair/triple interactions
	for(int j=0; j<obj.ntypes(); ++j){
		pos+=pack(obj.basisR(j),arr+pos);
	}
	for(int j=0; j<obj.ntypes(); ++j){
		for(int k=j; k<obj.ntypes(); ++k){
			pos+=pack(obj.basisA(j,k),arr+pos);
		}
	}
	//return bytes written
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(NNH& obj, const char* arr){
	if(NNP_PRINT_FUNC>0) std::cout<<"unpack(NNH&,const char*):\n";
	int pos=0;
	//hamiltonian
	pos+=unpack(obj.type(),arr+pos);
	pos+=unpack(obj.nn(),arr+pos);
	obj.dOutDVal().resize(obj.nn());
	//types
	int ntypes=0;
	pos+=unpack(ntypes,arr+pos);
	obj.resize(ntypes);
	//basis for pair/triple interactions
	for(int j=0; j<obj.ntypes(); ++j){
		pos+=unpack(obj.basisR(j),arr+pos);
	}
	for(int j=0; j<obj.ntypes(); ++j){
		for(int k=j; k<obj.ntypes(); ++k){
			pos+=unpack(obj.basisA(j,k),arr+pos);
		}
	}
	//intialize the inputs and offsets
	obj.init_input();
	//return bytes read
	return pos;
}

}