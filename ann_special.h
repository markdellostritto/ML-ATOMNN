#ifndef MATH_SPECIAL_ANN_HPP
#define MATH_SPECIAL_ANN_HPP

// c libaries
#include <cstdio>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#endif
// c++ libraries
#include <iosfwd>
#include <vector>
// ann - math
#include "ann_const.h"

#ifndef DEBUG_MATH_SPECIAL
#define DEBUG_MATH_SPECIAL 0
#endif 

namespace math{
	
namespace special{
	
	static const double prec=1E-8;
	
	//**************************************************************
	// sign function
	//**************************************************************

	template <typename T> int sgn(T val){return (T(0)<val)-(val<T(0));}
	
	//**************************************************************
	//trig (fdlibm)
	//**************************************************************
	
	static const double cos_const[6]={
		4.16666666666666019037e-02,
		-1.38888888888741095749e-03,
		2.48015872894767294178e-05,
		-2.75573143513906633035e-07,
		2.08757232129817482790e-09,
		-1.13596475577881948265e-11
	};
	//cosine function
	double cos(double x);
	double coscut(double x);
	
	static const double sin_const[6]={
		-1.66666666666666324348e-01,
		8.33333333332248946124e-03,
		-1.98412698298579493134e-04,
		2.75573137070700676789e-06,
		-2.50507602534068634195e-08,
		1.58969099521155010221e-10
	};
	//sine function
	double sin(double x);
	double sincut(double x);
	
	//**************************************************************
	//Hypberbolic Functions
	//**************************************************************
	
	double sinh(double x);
	double cosh(double x);
	double tanh(double x);
	double csch(double x);
	double sech(double x);
	double coth(double x);
	void tanhsech(double x, double& ftanh, double& fsech);
	
	//**************************************************************
	//Power
	//**************************************************************
	
	double powint(double x, int p);
	
	//**************************************************************
	//Sigmoid function
	//**************************************************************
	
	inline double sigmoid(double x){return 1.0/(1.0+exp(-x));}
	
	//**************************************************************
	//Softplus
	//**************************************************************
	
	double softplus(double x);
	
	//**************************************************************
	//Exponential
	//**************************************************************
	
	double exp2_x86(double x);
	double fm_exp(double x);
	
	//**************************************************************
	//Gamma Function
	//**************************************************************
	
	static const double gammac[15]={
		0.99999999999999709182,
		57.156235665862923517,
		-59.597960355475491248,
		14.136097974741747174,
		-0.49191381609762019978,
		0.33994649984811888699e-4,
		0.46523628927048575665e-4,
		-0.98374475304879564677e-4,
		0.15808870322491248884e-3,
		-0.21026444172410488319e-3,
		0.21743961811521264320e-3,
		-0.16431810653676389022e-3,
		0.84418223983852743293e-4,
		-0.26190838401581408670e-4,
		0.36899182659531622704e-5
	};
	double lgamma(double x);
	double tgamma(double x);
	
	//**************************************************************
	//Beta Function
	//**************************************************************
	
	double beta(double z, double w);
	
}

namespace poly{
	
	//**************************************************************
	//Legendre Poylnomials
	//**************************************************************
	double legendre(int n, double x);
	std::vector<double>& legendre_c(int n, std::vector<double>& c);
	
	//**************************************************************
	//Chebyshev Polynomials
	//**************************************************************
	double chebyshev1l(int n, double x);//chebyshev polynomial of the first kind
	double chebyshev2l(int n, double x);//chebyshev polynomial of the second kind
	std::vector<double>& chebyshev1_c(int n, double x, std::vector<double>& r);//polynomial coefficients
	std::vector<double>& chebyshev2_c(int n, double x, std::vector<double>& r);//polynomial coefficients
	std::vector<double>& chebyshev1_r(int n, std::vector<double>& r);//polynomial roots
	std::vector<double>& chebyshev2_r(int n, std::vector<double>& r);//polynomial roots
	
	//**************************************************************
	//Jacobi Polynomials
	//**************************************************************
	double jacobi(int n, double a, double b, double x);
	std::vector<double>& jacobi(int n, double a, double b, std::vector<double>& c);
	
	//**************************************************************
	//Laguerre Polynomials
	//**************************************************************
	double laguerre(int n, double x);
	std::vector<double>& laguerre_c(int n, std::vector<double>& c);
	
}

}

#endif
