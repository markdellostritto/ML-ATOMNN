#pragma once
#ifndef BASIS_HPP
#define BASIS_HPP

// eigen
#include <Eigen/Dense>
//nnp
#include "ann_cutoff.h"
//mem
#include "ann_serialize.h"

#ifndef BASIS_PRINT_FUNC
#define BASIS_PRINT_FUNC 0
#endif

struct Basis{
protected:
	int size_;//number of functions
	Cutoff cutoff_;//cutoff
public:
	//==== constructors/destructors ====
	Basis(){clear();}
	Basis(double rc, Cutoff::Name cutname, int nf);
	~Basis(){clear();}
	
	//==== member access ====
	const int& size()const{return size_;}
	Cutoff& cutoff(){return cutoff_;}
	const Cutoff& cutoff()const{return cutoff_;}
	
	//==== member functions ====
	void clear();
	void resize(int nf);
};

#endif