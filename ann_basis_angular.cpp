// c libraries
#include <cstring>
#include <cstdio>
// c++ libraries
#include <iostream>
#include <vector>
// math
#include "ann_special.h"
// string
#include "ann_string.h"
#include "ann_token.h"
// basis - angular
#include "ann_basis_angular.h"

//==== using statements ====

using math::constant::PI;
using math::special::fm_exp;

//*****************************************
// BasisA::Name - angular function names
//*****************************************

BasisA::Name BasisA::Name::read(const char* str){
	if(std::strcmp(str,"GAUSS")==0) return BasisA::Name::GAUSS;
	else if(std::strcmp(str,"GAUSS2")==0) return BasisA::Name::GAUSS2;
	else if(std::strcmp(str,"SECH")==0) return BasisA::Name::SECH;
	else if(std::strcmp(str,"STUDENT3")==0) return BasisA::Name::STUDENT3;
	else if(std::strcmp(str,"STUDENT4")==0) return BasisA::Name::STUDENT4;
	else if(std::strcmp(str,"STUDENT5")==0) return BasisA::Name::STUDENT5;
	else return BasisA::Name::UNKNOWN;
}

const char* BasisA::Name::name(const BasisA::Name& name){
	switch(name){
		case BasisA::Name::GAUSS: return "GAUSS";
		case BasisA::Name::GAUSS2: return "GAUSS2";
		case BasisA::Name::SECH: return "SECH";
		case BasisA::Name::STUDENT3: return "STUDENT3";
		case BasisA::Name::STUDENT4: return "STUDENT4";
		case BasisA::Name::STUDENT5: return "STUDENT5";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const BasisA::Name& name){
	switch(name){
		case BasisA::Name::GAUSS: out<<"GAUSS"; break;
		case BasisA::Name::GAUSS2: out<<"GAUSS2"; break;
		case BasisA::Name::SECH: out<<"SECH"; break;
		case BasisA::Name::STUDENT3: out<<"STUDENT3"; break;
		case BasisA::Name::STUDENT4: out<<"STUDENT4"; break;
		case BasisA::Name::STUDENT5: out<<"STUDENT5"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//==== constructors/destructors ====

/**
* constructor
*/
BasisA::BasisA(double rc, Cutoff::Name cutname, int size, BasisA::Name name):Basis(rc,cutname,size){
	if(name==BasisA::Name::UNKNOWN) throw std::invalid_argument("BasisA(rc,Cutoff::Name,int,BasisA::Name): invalid angular function type");
	else name_=name;
	resize(size);
}

/**
* destructor
*/
BasisA::~BasisA(){
	clear();
}

//==== operators ====

/**
* print basis
* @param out - the output stream
* @param basis - the basis to print
* @return the output stream
*/
std::ostream& operator<<(std::ostream& out, const BasisA& basis){
	out<<"BasisA "<<basis.cutoff().name()<<" "<<basis.cutoff().rc()<<" "<<basis.name_<<" "<<basis.size_;
	for(int i=0; i<basis.size(); ++i){
		out<<"\n\t"<<basis.eta_[i]<<" "<<basis.zeta_[i]<<" "<<basis.lambda_[i];
	}
	return out;
}

//==== reading/writing ====

/**
* write basis to file
* @param writer - file pointer
* @param basis - the basis to be written
*/
void BasisA::write(FILE* writer, const BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::write(FILE*):\n";
	const char* str_tcut=Cutoff::Name::name(basis.cutoff().name());
	const char* str_name=BasisA::Name::name(basis.name());
	fprintf(writer,"BasisA %s %f %s %i\n",str_tcut,basis.cutoff().rc(),str_name,basis.size());
	for(int i=0; i<basis.size(); ++i){
		fprintf(writer,"\t%f %f %i\n",basis.eta(i),basis.zeta(i),basis.lambda(i));
	}
}

/**
* read basis from file
* @param writer - file pointer
* @param basis - the basis to be read
*/
void BasisA::read(FILE* reader, BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::read(FILE*, BasisA&):\n";
	//local variables
	char* input=new char[string::M];
	//read header
	Token token(fgets(input,string::M,reader),string::WS); token.next();
	const Cutoff::Name cutname=Cutoff::Name::read(token.next().c_str());
	const double rc=std::atof(token.next().c_str());
	const BasisA::Name name=BasisA::Name::read(token.next().c_str());
	const int size=std::atoi(token.next().c_str());
	//initialize
	basis=BasisA(rc,cutname,size,name);
	//read parameters
	for(int i=0; i<basis.size(); ++i){
		token.read(fgets(input,string::M,reader),string::WS);
		basis.eta(i)=std::atof(token.next().c_str());
		basis.zeta(i)=std::atof(token.next().c_str());
		basis.lambda(i)=std::atoi(token.next().c_str());
	}
	basis.init();
	//free local variables
	delete[] input;
}

//==== member functions ====

/**
* clear basis
*/
void BasisA::clear(){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::clear():\n";
	Basis::clear();
	name_=BasisA::Name::UNKNOWN;
	eta_.clear();
	zeta_.clear();
	ieta2_.clear();
	lambdaf_.clear();
	fdampr_.clear();
	gdampr_.clear();
	etar_.clear();
	ietar2_.clear();
	rflag_.clear();
	lambda_.clear();
}	

/**
* resize interval vectors
*/
void BasisA::resize(int size){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::resize(int):\n";
	Basis::resize(size);
	if(size>0){
		eta_.resize(size);
		zeta_.resize(size);
		ieta2_.resize(size);
		lambdaf_.resize(size_);
		fdampr_.resize(size_);
		gdampr_.resize(size_);
		etar_.resize(size_);
		ietar2_.resize(size_);
		rflag_.resize(size_);
		lambda_.resize(size_);
	}
}

/**
* initialize pre-computed coefficients
*/
void BasisA::init(){
	for(int i=0; i<size_; ++i){
		ieta2_[i]=1.0/(eta_[i]*eta_[i]);
		lambdaf_[i]=0.5*lambda_[i];
	}
	if(size_>0){
		int flag=0;
		rflag_[0]=flag;
		etar_.clear();
		etar_.push_back(eta_[0]);
		for(int n=1; n<size_; ++n){
			if(std::fabs(eta_[n]-eta_[n-1])<1.0e-6){
				rflag_[n]=flag;
			} else {
				rflag_[n]=++flag;
				etar_.push_back(eta_[n]);
			}
		}
		flag++;
		ietar2_.resize(flag);
		for(int n=0; n<flag; ++n){
			ietar2_[n]=1.0/(etar_[n]*etar_[n]);
		}
		fdampr_.resize(flag);
		gdampr_.resize(flag);
	}
}

/**
* compute symmetry functions, adding to the supplied vector
* @param cos - the cosine of the triple
* @param dr - the triple distances: dr={rij,rik,rjk} with i at the vertex
* Note: we take the absolute value of the cosine argument in order to account for very small
* negative values due to numerical noise when the triple angle is close to Pi
*/
void BasisA::symm(const double dr[2], double cos, double* symm){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::symm(double,const double*):\n";
	const double cprod=cutoff_.cutf(dr[0])*cutoff_.cutf(dr[1]);
	const double r2s=dr[0]*dr[0]+dr[1]*dr[1];
	switch(name_){
		case BasisA::Name::GAUSS:{
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=cprod*fm_exp(-ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//symm[i]+=cprod*pow(fabs(0.5+lambdaf_[i]*cos),zeta_[i])*fm_exp(-ieta2_[i]*r2s);
				symm[i]+=pow(fabs(0.5+lambdaf_[i]*cos),zeta_[i])*fdampr_[rflag_[i]];
			}
		} break;
		case BasisA::Name::GAUSS2:{
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=fm_exp(-0.5*PI*ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//symm_[i]=cprod*pow(fabs(0.5+lambdaf_[i]*cos),zeta_[i])*fm_exp(-ieta2_[i]*r2s);
				symm[i]+=cprod*pow(fabs(0.5+lambdaf_[i]*cos),zeta_[i])*2.0*fdampr_[rflag_[i]]/(1.0+fdampr_[rflag_[i]]);
			}
		} break;
		case BasisA::Name::SECH:{
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=fm_exp(-0.5*PI*ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//const double fexp=fm_exp(-ieta2_[i]*r2s);
				const double fexp=fdampr_[rflag_[i]];
				symm[i]+=cprod*pow(fabs(0.5+lambdaf_[i]*cos),zeta_[i])*2.0*fexp/(1.0+fexp*fexp);
			}
		} break;
		case BasisA::Name::STUDENT3:{
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=cprod*math::special::powint(1.0/sqrt(1.0+ietar2_[j]*r2s),4);
			for(int i=0; i<size_; ++i){
				//symm[i]+=cprod*pow(fabs(0.5+lambdaf_[i]*cos),zeta_[i])*math::special::powint(1.0/sqrt(1.0+ieta2_[i]*r2s),4);
				symm[i]+=pow(fabs(0.5+lambdaf_[i]*cos),zeta_[i])*fdampr_[rflag_[i]];
			}
		} break;
		case BasisA::Name::STUDENT4:{
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=cprod*math::special::powint(1.0/sqrt(1.0+ietar2_[j]*r2s),5);
			for(int i=0; i<size_; ++i){
				//symm[i]+=cprod*pow(fabs(0.5+lambdaf_[i]*cos),zeta_[i])*math::special::powint(1.0/sqrt(1.0+ieta2_[i]*r2s),5);
				symm[i]+=pow(fabs(0.5+lambdaf_[i]*cos),zeta_[i])*fdampr_[rflag_[i]];
			}
		} break;
		case BasisA::Name::STUDENT5:{
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=cprod*math::special::powint(1.0/sqrt(1.0+ietar2_[j]*r2s),6);
			for(int i=0; i<size_; ++i){
				//symm[i]+=cprod*pow(fabs(0.5+lambdaf_[i]*cos),zeta_[i])*math::special::powint(1.0/sqrt(1.0+ieta2_[i]*r2s),6);
				symm[i]+=pow(fabs(0.5+lambdaf_[i]*cos),zeta_[i])*fdampr_[rflag_[i]];
			}
		} break;
		default:
			throw std::invalid_argument("BasisA::symm(double): Invalid symmetry function.");
		break;
	}
}

/**
* compute angular force coefficients "phi" and "eta"
* note: both commented, "simple" math and optimized math statements are included
* @param phi - stores angular gradients
* @param eta - stores radial gradients
* @param cos - the cosine of the triple
* @param dr - the triple distances: r={rij,rik,rjk} with i at the vertex
* @param dEdG - gradient of energy w.r.t. the inputs
*/
void BasisA::force(const double dr[2], double cos, double& phi, double* eta, const double* dEdG){
	//compute cutoffs
	const double c[2]={
		cutoff_.cutf(dr[0]),//cut(rij)
		cutoff_.cutf(dr[1])//cut(rik)
	};
	const double g[2]={
		cutoff_.cutg(dr[0]),//cut'(rij)
		cutoff_.cutg(dr[1])//cut'(rik)
	};
	//compute phi, eta
	phi=0;
	eta[0]=0;
	eta[1]=0;
	const double r2s=dr[0]*dr[0]+dr[1]*dr[1];
	switch(name_){
		case BasisA::Name::GAUSS:{
			const double dij=2.0*dr[0]*c[0];
			const double dik=2.0*dr[1]*c[1];
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=fm_exp(-ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//compute angular values
				const double cw=fabs(0.5+lambdaf_[i]*cos);
				//const double gangle=dEdG[i]*pow(cw,zeta_[i]-1.0)*fm_exp(-ieta2_[i]*r2s);
				const double gangle=dEdG[i]*pow(cw,zeta_[i]-1.0)*fdampr_[rflag_[i]];
				const double fangle=cw*gangle;
				//compute phi
				phi-=zeta_[i]*lambdaf_[i]*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ieta2_[i]+g[0]);
				eta[1]-=fangle*(-dik*ieta2_[i]+g[1]);
			}
		} break;
		case BasisA::Name::GAUSS2:{
			const double dij=PI*dr[0]*c[0];
			const double dik=PI*dr[1]*c[1];
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=fm_exp(-0.5*PI*ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				const double fexp=fdampr_[rflag_[i]];
				const double den=1.0/(1.0+fexp);
				//compute angular values
				const double cw=fabs(0.5+lambdaf_[i]*cos);
				const double gangle=dEdG[i]*pow(cw,zeta_[i]-1.0)*2.0*fexp*den;
				const double fangle=cw*gangle;
				//compute phi
				phi-=zeta_[i]*lambdaf_[i]*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ieta2_[i]*den+g[0]);
				eta[1]-=fangle*(-dik*ieta2_[i]*den+g[1]);
			}
		} break;
		case BasisA::Name::SECH:{
			const double dij=PI*dr[0]*c[0];
			const double dik=PI*dr[1]*c[1];
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=fm_exp(-0.5*PI*ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//compute distance values
				//const double fexp=fm_exp(-ieta2_[i]*r2s);
				const double fexp=fdampr_[rflag_[i]];
				const double fexp2=fexp*fexp;
				const double den=1.0/(1.0+fexp2);
				const double ftanh=(1.0-fexp2)*den;
				const double fsech=2.0*fexp*den;
				//compute angular values
				const double cw=fabs(0.5+lambdaf_[i]*cos);
				const double gangle=dEdG[i]*pow(cw,zeta_[i]-1.0)*fsech;
				const double fangle=cw*gangle;
				//compute phi
				phi-=zeta_[i]*lambdaf_[i]*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ieta2_[i]*ftanh+g[0]);
				eta[1]-=fangle*(-dik*ieta2_[i]*ftanh+g[1]);
			}
		} break;
		case BasisA::Name::STUDENT3:{
			const double dij=4.0*dr[0]*c[0];
			const double dik=4.0*dr[1]*c[1];
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=1.0/sqrt(1.0+ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//compute angular values
				//const double den=1.0/sqrt(1.0+ieta2_[i]*r2s);
				const double den=fdampr_[rflag_[i]];
				const double cw=fabs(0.5+lambdaf_[i]*cos);
				const double gangle=pow(cw,zeta_[i]-1.0)*dEdG[i]*math::special::powint(den,4);
				const double fangle=cw*gangle;
				//compute phi
				phi-=zeta_[i]*lambdaf_[i]*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ieta2_[i]*den*den+g[0]);
				eta[1]-=fangle*(-dik*ieta2_[i]*den*den+g[1]);
			}
		} break;
		case BasisA::Name::STUDENT4:{
			const double dij=5.0*dr[0]*c[0];
			const double dik=5.0*dr[1]*c[1];
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=1.0/sqrt(1.0+ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//compute angular values
				//const double den=1.0/sqrt(1.0+ieta2_[i]*r2s);
				const double den=fdampr_[rflag_[i]];
				const double cw=fabs(0.5+lambdaf_[i]*cos);
				const double gangle=pow(cw,zeta_[i]-1.0)*dEdG[i]*math::special::powint(den,5);
				const double fangle=cw*gangle;
				//compute phi
				phi-=zeta_[i]*lambdaf_[i]*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ieta2_[i]*den*den+g[0]);
				eta[1]-=fangle*(-dik*ieta2_[i]*den*den+g[1]);
			}
		} break;
		case BasisA::Name::STUDENT5:{
			const double dij=6.0*dr[0]*c[0];
			const double dik=6.0*dr[1]*c[1];
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=1.0/sqrt(1.0+ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//compute angular values
				//const double den=1.0/sqrt(1.0+ieta2_[i]*r2s);
				const double den=fdampr_[rflag_[i]];
				const double cw=fabs(0.5+lambdaf_[i]*cos);
				const double gangle=pow(cw,zeta_[i]-1.0)*dEdG[i]*math::special::powint(den,6);
				const double fangle=cw*gangle;
				//compute phi
				phi-=zeta_[i]*lambdaf_[i]*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ieta2_[i]*den*den+g[0]);
				eta[1]-=fangle*(-dik*ieta2_[i]*den*den+g[1]);
			}
		} break;
		default:
			throw std::invalid_argument("BasisA::force(double&,double*,double,const double[3],const double*)const: Invalid symmetry function.");
		break;
	}
	//normalize
	phi*=c[0]*c[1];
	eta[0]*=c[1];
	eta[1]*=c[0];
}

//==== serialization ====

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const BasisA& obj){
		if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"nbytes(const BasisA&):\n";
		int size=0;
		size+=sizeof(obj.size());//number of symmetry functions
		size+=sizeof(obj.name());//name of symmetry functions
		size+=nbytes(obj.cutoff());
		const int s=obj.size();
		size+=sizeof(double)*s;//eta
		size+=sizeof(double)*s;//zeta
		size+=sizeof(int)*s;//lambda
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const BasisA& obj, char* arr){
		if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"pack(const BasisA&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.size(),sizeof(obj.size())); pos+=sizeof(obj.size());
		std::memcpy(arr+pos,&obj.name(),sizeof(obj.name())); pos+=sizeof(obj.name());
		pos+=pack(obj.cutoff(),arr+pos);
		const int size=obj.size();
		if(size>0){
			std::memcpy(arr+pos,obj.eta().data(),size*sizeof(double)); pos+=size*sizeof(double);
			std::memcpy(arr+pos,obj.zeta().data(),size*sizeof(double)); pos+=size*sizeof(double);
			std::memcpy(arr+pos,obj.lambda().data(),size*sizeof(int)); pos+=size*sizeof(int);
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(BasisA& obj, const char* arr){
		if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"unpack(BasisA&,const char*):\n";
		int pos=0; int size=0;
		BasisA::Name name=BasisA::Name::UNKNOWN;
		std::memcpy(&size,arr+pos,sizeof(size)); pos+=sizeof(size);
		std::memcpy(&name,arr+pos,sizeof(BasisA::Name)); pos+=sizeof(BasisA::Name);
		pos+=unpack(obj.cutoff(),arr+pos);
		obj=BasisA(obj.cutoff().rc(),obj.cutoff().name(),size,name);
		if(size>0){
			std::memcpy(obj.eta().data(),arr+pos,size*sizeof(double)); pos+=size*sizeof(double);
			std::memcpy(obj.zeta().data(),arr+pos,size*sizeof(double)); pos+=size*sizeof(double);
			std::memcpy(obj.lambda().data(),arr+pos,size*sizeof(int)); pos+=size*sizeof(int);
		}
		obj.init();
		return pos;
	}
	
}
