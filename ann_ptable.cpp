#include "ann_ptable.h"

#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#endif

namespace PTable{
	
//*********************************************
//Function
//*********************************************

//************** NAME ***************
const char* name(int an){return ELEMENT_NAME[an-1];}
//********** ATOMIC_NUMBER **********
int an(const char* name){
	for(int i=0; i<NUM_ELEMENTS; i++){
		if(std::strcmp(name,ELEMENT_NAME[i])==0) return i+1;
	}
	return 0;
}
int an(double mass){
	double min=100;
	int an=0;
	for(int i=0; i<NUM_ELEMENTS; ++i){
		if(fabs(mass-ELEMENT_MASS[i])<min){
			min=fabs(mass-ELEMENT_MASS[i]);
			an=i+1;
		}
	}
	return an;
}
//************** MASS ***************
double mass(int an){return ELEMENT_MASS[an-1];}
//************* RADIUS **************
double atomicRadius(int an){return ATOMIC_RADII[an-1];}
double covalentRadius(int an){return COVALENT_RADII[an-1];}

}

