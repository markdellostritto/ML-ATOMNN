// c libraries
#include <cstring>
#include <cstdio>
// c++ libraries
#include <iostream>
#include <vector>
// math
#include "ann_special.h"
#include "ann_const.h"
// str
#include "ann_string.h"
#include "ann_token.h"
// basis - radial
#include "ann_basis_radial.h"

//==== using statements ====

using math::special::fm_exp;
using math::constant::Rad2;
using math::constant::RadPI;

//==== constants ====

//*****************************************
// BasisR::Name - radial function names
//*****************************************

BasisR::Name BasisR::Name::read(const char* str){
	if(std::strcmp(str,"GAUSSIAN")==0) return BasisR::Name::GAUSSIAN;
	else if(std::strcmp(str,"SECH")==0) return BasisR::Name::SECH;
	else if(std::strcmp(str,"LOGISTIC")==0) return BasisR::Name::LOGISTIC;
	else if(std::strcmp(str,"TANH")==0) return BasisR::Name::TANH;
	else if(std::strcmp(str,"LOGCOSH")==0) return BasisR::Name::LOGCOSH;
	else if(std::strcmp(str,"LOGCOSH2")==0) return BasisR::Name::LOGCOSH2;
	else return BasisR::Name::UNKNOWN;
}

const char* BasisR::Name::name(const BasisR::Name& name){
	switch(name){
		case BasisR::Name::GAUSSIAN: return "GAUSSIAN";
		case BasisR::Name::SECH: return "SECH";
		case BasisR::Name::LOGISTIC: return "LOGISTIC";
		case BasisR::Name::TANH: return "TANH";
		case BasisR::Name::LOGCOSH: return "LOGCOSH";
		case BasisR::Name::LOGCOSH2: return "LOGCOSH2";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const BasisR::Name& name){
	switch(name){
		case BasisR::Name::GAUSSIAN: out<<"GAUSSIAN"; break;
		case BasisR::Name::SECH: out<<"SECH"; break;
		case BasisR::Name::LOGISTIC: out<<"LOGISTIC"; break;
		case BasisR::Name::TANH: out<<"TANH"; break;
		case BasisR::Name::LOGCOSH: out<<"LOGCOSH"; break;
		case BasisR::Name::LOGCOSH2: out<<"LOGCOSH2"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//*****************************************
// BasisR - radial basis
//*****************************************

//==== constructors/destructors ====

/**
* constructor
*/
BasisR::BasisR(double rc, Cutoff::Name cutname, int size, BasisR::Name name):Basis(rc,cutname,size){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR(rc,Cutoff::Name,int,BasisR::Name):\n";
	if(name==BasisR::Name::UNKNOWN) throw std::invalid_argument("BasisR(rc,Cutoff::Name,int,BasisR::Name): invalid radial function type");
	else name_=name;
	resize(size);
}

/**
* destructor
*/
BasisR::~BasisR(){
	clear();
}


//==== operators ====

/**
* print basis
* @param out - the output stream
* @param basis - the basis to print
* @return the output stream
*/
std::ostream& operator<<(std::ostream& out, const BasisR& basis){
	out<<"BasisR "<<basis.cutoff().name()<<" "<<basis.cutoff().rc()<<" "<<basis.name_<<" "<<basis.size_;
	for(int i=0; i<basis.size(); ++i){
		out<<"\n\t"<<basis.rs_[i]<<" "<<basis.eta_[i]<<" ";
	}
	return out;
}

//==== reading/writing ====

/**
* write basis to file
* @param writer - file pointer
* @param basis - the basis to be written
*/
void BasisR::write(FILE* writer,const BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::write(FILE*,const BasisR&):\n";
	const char* str_tcut=Cutoff::Name::name(basis.cutoff().name());
	const char* str_name=BasisR::Name::name(basis.name());
	fprintf(writer,"BasisR %s %f %s %i\n",str_tcut,basis.cutoff().rc(),str_name,basis.size());
	for(int i=0; i<basis.size(); ++i){
		fprintf(writer,"\t%f %f\n",basis.rs(i),basis.eta(i));
	}
}

/**
* read basis from file
* @param writer - file pointer
* @param basis - the basis to be read
*/
void BasisR::read(FILE* reader, BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::read(FILE*, BasisR&):\n";
	//local variables
	char* input=new char[string::M];
	//read header
	Token token(fgets(input,string::M,reader),string::WS); token.next();
	const Cutoff::Name cutname=Cutoff::Name::read(token.next().c_str());
	const double rc=std::atof(token.next().c_str());
	const BasisR::Name name=BasisR::Name::read(token.next().c_str());
	const int size=std::atoi(token.next().c_str());
	//initialize
	basis=BasisR(rc,cutname,size,name);
	//read parameters
	for(int i=0; i<basis.size(); ++i){
		token.read(fgets(input,string::M,reader),string::WS);
		basis.rs(i)=std::atof(token.next().c_str());
		basis.eta(i)=std::atof(token.next().c_str());
	}
	//free local variables
	delete[] input;
}

//==== member functions ====

/**
* clear basis
*/
void BasisR::clear(){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::clear():\n";
	Basis::clear();
	name_=BasisR::Name::UNKNOWN;
	rs_.clear();
	eta_.clear();
}

/**
* resize symmetry function and parameter arrays
* @param size - the total number of symmetry functions/parameters
*/
void BasisR::resize(int size){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::resize(int):\n";
	Basis::resize(size);
	if(size_>0){
		rs_.resize(size);
		eta_.resize(size);
	}
}

/**
* compute symmetry functions, adding the the supplied array
* @param dr - the distance between the central atom and a neighboring atom
*/
void BasisR::symm(double dr, double* symm)const{
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::symm(double):\n";
	const double cutf=cutoff_.cutf(dr);
	switch(name_){
		case BasisR::Name::GAUSSIAN:{
			for(int i=0; i<size_; ++i){
				const double arg=eta_[i]*(dr-rs_[i]);
				symm[i]+=fm_exp(-arg*arg)*cutf;
			}
		} break;
		case BasisR::Name::SECH:{
			for(int i=0; i<size_; ++i){
				//symm_[i]=math::special::sech(eta_[i]*(dr[0]-rs_[i]))*cutf;
				const double expf=fm_exp(-eta_[i]*(dr-rs_[i]));
				symm[i]+=2.0*expf/(1.0+expf*expf)*cutf;
			}
		} break;
		case BasisR::Name::LOGISTIC:{
			for(int i=0; i<size_; ++i){
				//const double sechf=math::special::sech(eta_[i]*(dr[0]-rs_[i]));
				//symm_[i]=sechf*sechf*cutf;
				const double expf=fm_exp(-2.0*eta_[i]*(dr-rs_[i]));
				symm[i]+=4.0*expf/((1.0+expf)*(1.0+expf))*cutf;
			}
		} break;
		case BasisR::Name::TANH:{
			for(int i=0; i<size_; ++i){
				symm[i]+=0.5*(tanh(-eta_[i]*(dr-rs_[i]))+1.0)*cutf;
			}
		} break;
		case BasisR::Name::LOGCOSH:{
			for(int i=0; i<size_; ++i){
				symm[i]+=0.5*log1p(fm_exp(-2.0*eta_[i]*(dr-rs_[i])))*cutf;
			}
		} break;
		case BasisR::Name::LOGCOSH2:{
			for(int i=0; i<size_; ++i){
				const double arg=-eta_[i]*(dr-rs_[i]);
				const double fexp=fm_exp(2.0*Rad2*arg);
				symm[i]+=(arg==0.0)?0.5*cutf/Rad2:arg*fexp/(fexp-1.0)*cutf;
			}
		} break;
		default:
			throw std::invalid_argument("BasisR::symm(double,double*): Invalid symmetry function.");
		break;
	}
}

/**
* compute force
* @param dr - the distance between the central atom and a neighboring atom
* @param dEdG - gradient of energy w.r.t. the inputs
* @return - the amplitude of the two-body force at distance dr
*/
double BasisR::force(double dr, const double* dEdG)const{
	double amp=0;//force amplitude
	const double cutf=cutoff_.cutf(dr);
	const double cutg=cutoff_.cutg(dr);
	switch(name_){
		case BasisR::Name::GAUSSIAN:{
			for(int i=0; i<size_; ++i){
				const double arg=eta_[i]*(dr-rs_[i]);
				amp-=dEdG[i]*fm_exp(-arg*arg)*(-2.0*eta_[i]*arg*cutf+cutg);
			}
		} break;
		case BasisR::Name::SECH:{
			for(int i=0; i<size_; ++i){
				/*const double sechf=math::special::sech(eta_[i]*(dr[0]-rs_[i]));
				const double tanhf=math::special::tanh(eta_[i]*(dr[0]-rs_[i]));
				amp-=dEdG[i]*sechf*(-1.0*eta_[i]*tanhf*cutf+cutg);*/
				const double expf=fm_exp(-eta_[i]*(dr-rs_[i]));
				const double expf2=expf*expf;
				const double den=1.0/(1.0+expf2);
				const double tanhf=(1.0-expf2)*den;
				const double sechf=2.0*expf*den;
				amp-=dEdG[i]*sechf*(-eta_[i]*tanhf*cutf+cutg);
			}
		} break;
		case BasisR::Name::LOGISTIC:{
			for(int i=0; i<size_; ++i){
				/*const double sechf=math::special::sech(eta_[i]*(dr[0]-rs_[i]));
				const double tanhf=math::special::tanh(eta_[i]*(dr[0]-rs_[i]));
				amp-=dEdG[i]*sechf*sechf*(-2.0*eta_[i]*tanhf*cutf+cutg);*/
				const double fexp2=fm_exp(-2.0*eta_[i]*(dr-rs_[i]));
				const double den=1.0/(1.0+fexp2);
				const double tanhf=(1.0-fexp2)*den;
				const double sech2f=4.0*fexp2*den*den;
				amp-=dEdG[i]*sech2f*(-2.0*eta_[i]*tanhf*cutf+cutg);
			}
		} break;
		case BasisR::Name::TANH:{
			for(int i=0; i<size_; ++i){
				const double tanhf=tanh(-eta_[i]*(dr-rs_[i]));
				amp-=dEdG[i]*0.5*(-eta_[i]*(1.0-tanhf*tanhf)*cutf+(1.0+tanhf)*cutg);
			}
		} break;
		case BasisR::Name::LOGCOSH:{
			for(int i=0; i<size_; ++i){
				const double fexp=fm_exp(-2.0*eta_[i]*(dr-rs_[i]));
				amp-=dEdG[i]*(-eta_[i]*fexp/(1.0+fexp)*cutf+0.5*log1p(fexp)*cutg);
			}
		} break;
		case BasisR::Name::LOGCOSH2:{
			for(int i=0; i<size_; ++i){
				const double arg=-eta_[i]*(dr-rs_[i]);
				const double fexp=fm_exp(2.0*Rad2*arg);
				const double den=1.0/(fexp-1.0);
				const double val=(arg==0.0)?0.5*(-eta_[i]*cutg+cutf/Rad2):fexp*(eta_[i]*(1.0+2.0*Rad2*arg-fexp)*den*cutf+arg*cutg)*den;
				amp-=dEdG[i]*val;
			}
		} break;
		default:
			throw std::invalid_argument("BasisR::force(double,double*): Invalid symmetry function.");
		break;
	}
	return amp;
}

//==== serialization ====

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const BasisR& obj){
		if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"nbytes(const BasisR&):\n";
		int size=0;
		size+=sizeof(obj.size());//number of symmetry functions
		size+=sizeof(obj.name());//name of symmetry functions
		size+=nbytes(obj.cutoff());
		const int s=obj.size();
		size+=sizeof(double)*s;//rs
		size+=sizeof(double)*s;//zeta
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const BasisR& obj, char* arr){
		if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"pack(const BasisR&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.size(),sizeof(obj.size())); pos+=sizeof(obj.size());//number of symmetry functions
		std::memcpy(arr+pos,&obj.name(),sizeof(obj.name())); pos+=sizeof(obj.name());//name of symmetry functions
		pos+=pack(obj.cutoff(),arr+pos);
		const int size=obj.size();
		if(size>0){
			std::memcpy(arr+pos,obj.rs().data(),size*sizeof(double)); pos+=size*sizeof(double);
			std::memcpy(arr+pos,obj.eta().data(),size*sizeof(double)); pos+=size*sizeof(double);
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(BasisR& obj, const char* arr){
		if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"unpack(BasisR&,const char*):\n";
		int pos=0; int size=0;
		BasisR::Name name=BasisR::Name::UNKNOWN;
		std::memcpy(&size,arr+pos,sizeof(size)); pos+=sizeof(size);
		std::memcpy(&name,arr+pos,sizeof(BasisR::Name)); pos+=sizeof(BasisR::Name);
		pos+=unpack(obj.cutoff(),arr+pos);
		obj=BasisR(obj.cutoff().rc(),obj.cutoff().name(),size,name);
		if(obj.size()>0){
			std::memcpy(obj.rs().data(),arr+pos,obj.size()*sizeof(double)); pos+=obj.size()*sizeof(double);
			std::memcpy(obj.eta().data(),arr+pos,obj.size()*sizeof(double)); pos+=obj.size()*sizeof(double);
		}
		return pos;
	}
	
}
