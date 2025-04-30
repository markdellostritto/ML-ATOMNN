#pragma once
#ifndef BASIS_ANGULAR_HPP
#define BASIS_ANGULAR_HPP

// c++ libraries
#include <ostream>
//eigen
#include <Eigen/Dense>
// symmetry functions
#include "ann_cutoff.h"
#include "ann_basis.h"
// serialize
#include "ann_serialize.h"

#ifndef BASIS_ANGULAR_PRINT_FUNC
#define BASIS_ANGULAR_PRINT_FUNC 0
#endif

//*****************************************
// BasisA - angular basis
//*****************************************

class BasisA: public Basis{
public:
	class Name{
	public:
		enum Type{
			UNKNOWN,
			GAUSS,
			GAUSS2,
			SECH,
			STUDENT3,
			STUDENT4,
			STUDENT5
		};
		//constructor
		Name():t_(Type::UNKNOWN){}
		Name(Type t):t_(t){}
		//operators
		operator Type()const{return t_;}
		//member functions
		static Name read(const char* str);
		static const char* name(const Name& name);
	private:
		Type t_;
		//prevent automatic conversion for other built-in types
		//template<typename T> operator T() const;
	};
	
private:
	Name name_;//type of angular functions
	std::vector<double> eta_;//radial width
	std::vector<double> zeta_;//angular width
	std::vector<double> ieta2_;//1.0/eta^-2
	std::vector<double> lambdaf_;//lambda_ float
	std::vector<double> fdampr_;
	std::vector<double> gdampr_;
	std::vector<double> etar_;
	std::vector<double> ietar2_;
	std::vector<int> lambda_;//sign of cosine window
	std::vector<int> rflag_;
public:
	//==== constructors/destructors ====
	BasisA():Basis(),name_(Name::UNKNOWN){}
	BasisA(double rc, Cutoff::Name cutname, int nf, Name name);
	~BasisA();
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const BasisA& basisA);
	
	//==== reading/writing ====
	static void write(FILE* writer, const BasisA& basis);
	static void read(FILE* writer, BasisA& basis);
	
	//==== member access ====
	Name& name(){return name_;}
	const Name& name()const{return name_;}
	double& eta(int i){return eta_[i];}
	const double& eta(int i)const{return eta_[i];}
	double& zeta(int i){return zeta_[i];}
	const double& zeta(int i)const{return zeta_[i];}
	int& lambda(int i){return lambda_[i];}
	const int& lambda(int i)const{return lambda_[i];}
	std::vector<double>& eta(){return eta_;}
	const std::vector<double>& eta()const{return eta_;}
	std::vector<double>& zeta(){return zeta_;}
	const std::vector<double>& zeta()const{return zeta_;}
	std::vector<int>& lambda(){return lambda_;}
	const std::vector<int>& lambda()const{return lambda_;}
	
	//==== member functions ====
	void clear();
	void resize(int size);
	void init();
	void symm(const double d[2], double cos, double*);
	void force(const double d[2], double cos, double& phi, double* eta, const double* dEdG);
};
std::ostream& operator<<(std::ostream& out, const BasisA::Name& name);

bool operator==(const BasisA& basis1, const BasisA& basis2);
inline bool operator!=(const BasisA& basis1, const BasisA& basis2){return !(basis1==basis2);}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const BasisA& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const BasisA& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(BasisA& obj, const char* arr);
	
}

#endif