// c libaries
#include <cstdint>
// c++ libraries
#include <ostream>
#include <stdexcept>
// ann - math - special
#include "ann_special.h"

namespace math{

namespace special{

	
	//**************************************************************
	//trig (fdlibm)
	//**************************************************************
	
	//cosine function
	double cos(double x){
		x*=x;
		return 1.0+x*(-0.5+x*(cos_const[0]+x*(cos_const[1]+x*(cos_const[2]+x*(cos_const[3]+x*(cos_const[4]+x*cos_const[5]))))));
	}
	
	double coscut(double x){
		x-=0.5*math::constant::PI;
		const double x2=x*x;
		//return x*(-1.0+x2*(1.0/6.0+x2*(-1.0/120.0+x2*(1.0/5040.0+x2*(-1.0/362880.0+x2*1.0/39916800.0)))));//n11
		//return x*(-1.0+x2*(1.0/6.0+x2*(-1.0/120.0+x2*(1.0/5040.0+x2*(-1.0/362880.0+x2*(1.0/39916800.0-x2*1.0/6227020800.0))))));//n13
		return x*(-1.0+x2*(1.0/6.0+x2*(-1.0/120.0+x2*(1.0/5040.0+x2*(-1.0/362880.0+x2*(1.0/39916800.0+x2*(-1.0/6227020800.0+x2*1.0/1307674368000.0)))))));//n15
	}
	
	//sine function
	double sin(double x){
		const double r=x*x;
		return x*(1.0+r*(sin_const[0]+r*(sin_const[1]+r*(sin_const[2]+r*(sin_const[3]+r*(sin_const[4]+r*sin_const[5]))))));
	}
	
	double sincut(double x){
		x-=0.5*math::constant::PI;
		const double x2=x*x;
		//return 1.0+x2*(-1.0/2.0+x2*(1.0/24.0+x2*(-1.0/720.0+x2*(1.0/40320.0+x2*(-1.0/3628800.0+x2*1.0/479001600.0)))));//n12
		//return 1.0+x2*(-1.0/2.0+x2*(1.0/24.0+x2*(-1.0/720.0+x2*(1.0/40320.0+x2*(-1.0/3628800.0+x2*(1.0/479001600.0-x2*1.0/87178291200.0))))));//n14
		return 1.0+x2*(-1.0/2.0+x2*(1.0/24.0+x2*(-1.0/720.0+x2*(1.0/40320.0+x2*(-1.0/3628800.0+x2*(1.0/479001600.0+x2*(-1.0/87178291200.0+x2*1.0/20922789888000.0)))))));//n16
	}
	
	//**************************************************************
	//Hypberbolic Function
	//**************************************************************
	
	double sinh(double x){
		if(x>=0){
			const double expf=exp(-x);
			return (1.0-expf*expf)/(2.0*expf);
		} else {
			const double expf=exp(x);
			return (expf*expf-1.0)/(2.0*expf);
		}
	}
	
	double cosh(double x){
		if(x>=0){
			const double expf=exp(-x);
			return (1.0+expf*expf)/(2.0*expf);
		} else {
			const double expf=exp(x);
			return (expf*expf+1.0)/(2.0*expf);
		}
	}
	
	double tanh(double x){
		if(x>=0){
			const double expf=exp(-2.0*x);
			return (1.0-expf)/(1.0+expf);
		} else {
			const double expf=exp(2.0*x);
			return (expf-1.0)/(expf+1.0);
		}
	}
	
	double csch(double x){
		if(x>=0){
			const double expf=exp(-x);
			return 2.0*expf/(1.0-expf*expf);
		} else {
			const double expf=exp(x);
			return 2.0*expf/(expf*expf-1.0);
		}
	}
	
	double sech(double x){
		if(x>=0){
			const double expf=exp(-x);
			return 2.0*expf/(1.0+expf*expf);
		} else {
			const double expf=exp(x);
			return 2.0*expf/(expf*expf+1.0);
		}
	}
	
	double coth(double x){
		if(x>=0){
			const double expf=exp(-2.0*x);
			return (1.0+expf)/(1.0-expf);
		} else {
			const double expf=exp(2.0*x);
			return (expf+1.0)/(expf-1.0);
		}
	}
	
	void tanhsech(double x, double& ftanh, double& fsech){
		if(x>=0){
			const double fexp=exp(-x);
			const double fexp2=fexp*fexp;
			const double den=1.0/(1.0+fexp2);
			ftanh=(1.0-fexp2)*den;
			fsech=2.0*fexp*den;
		} else {
			const double fexp=exp(x);
			const double fexp2=fexp*fexp;
			const double den=1.0/(1.0+fexp2);
			ftanh=(fexp2-1.0)*den;
			fsech=2.0*fexp*den;
		}
	}
	
	//**************************************************************
	//Power
	//**************************************************************
	
	double powint(double x, const int n){
		double yy, ww;
		if (n == 0) return 1.0;
		if (x == 0.0) return 0.0;
		int nn = (n > 0) ? n : -n;
		ww = x;
		for (yy = 1.0; nn != 0; nn >>= 1, ww *= ww)
		if (nn & 1) yy *= ww;
		return (n > 0) ? yy : 1.0 / yy;
	}
	
	//**************************************************************
	//Softplus
	//**************************************************************
	
	double softplus(double x){
		if(x>=1.0) return x+log1p(exp(-x));
		else return log1p(exp(x));
	}
	
	//**************************************************************
	//Exponential
	//**************************************************************
	
	/* optimizer friendly implementation of exp2(x).
	*
	* strategy:
	*
	* split argument into an integer part and a fraction:
	* ipart = floor(x+0.5);
	* fpart = x - ipart;
	*
	* compute exp2(ipart) from setting the ieee754 exponent
	* compute exp2(fpart) using a pade' approximation for x in [-0.5;0.5[
	*
	* the result becomes: exp2(x) = exp2(ipart) * exp2(fpart)
	*/

	/* IEEE 754 double precision floating point data manipulation */
	typedef union {
		double   f;
		uint64_t u;
		struct {int32_t  i0,i1;} s;
	}  udi_t;

	static const double fm_exp2_q[] = {
	/*  1.00000000000000000000e0, */
		2.33184211722314911771e2,
		4.36821166879210612817e3
	};
	static const double fm_exp2_p[] = {
		2.30933477057345225087e-2,
		2.02020656693165307700e1,
		1.51390680115615096133e3
	};

	/* double precision constants */
	#define FM_DOUBLE_LOG2OFE  1.4426950408889634074
	
	double exp2_x86(double x){
	#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
		double   ipart, fpart, px, qx;
		udi_t    epart;

		ipart = floor(x+0.5);
		fpart = x - ipart;
		epart.s.i0 = 0;
		epart.s.i1 = (((int) ipart) + 1023) << 20;

		x = fpart*fpart;

		px =        fm_exp2_p[0];
		px = px*x + fm_exp2_p[1];
		qx =    x + fm_exp2_q[0];
		px = px*x + fm_exp2_p[2];
		qx = qx*x + fm_exp2_q[1];

		px = px * fpart;

		x = 1.0 + 2.0*(px/(qx-px));
		return epart.f*x;
	#else
		return pow(2.0, x);
	#endif
	}
	
	double fm_exp(double x)
	{
	#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
	    if (x < -1022.0/FM_DOUBLE_LOG2OFE) return 0;
	    if (x > 1023.0/FM_DOUBLE_LOG2OFE) return INFINITY;
	    return exp2_x86(FM_DOUBLE_LOG2OFE * x);
	#else
	    return ::exp(x);
	#endif
	}
	
}

namespace poly{
	
	//**************************************************************
	//Legendre Polynomials
	//**************************************************************
	
	double legendre(int n, double x){
		if(n<0) throw std::runtime_error("legendre(int,double): invalid order");
		else if(n==0) return 1;
		else {
			double rm2=1.0,rm1=x,r=x;
			for(int i=2; i<=n; ++i){
				r=((2.0*n-1.0)*x*rm1-(n-1.0)*rm2)/i;
				rm2=rm1; rm1=r;
			}
			return r;
		}
	}
	
	std::vector<double>& legendre_c(int n, std::vector<double>& c){
		if(n==0) c.resize(n+1,1.0);
		else if(n==1){c.resize(n+1,0.0); c[1]=1.0;}
		else {
			c.resize(n+1,0.0);
			std::vector<double> ct1(n+1,0.0),ct2(n+1,0.0);
			ct1[0]=1.0; ct2[1]=1.0;
			for(int m=2; m<=n; ++m){
				for(int l=m; l>0; --l) c[l]=(2.0*m-1.0)/m*ct2[l-1];
				c[0]=0.0;
				for(int l=m; l>=0; --l) c[l]-=(m-1.0)/m*ct1[l];
				ct1=ct2; ct2=c;
			}
		}
		return c;
	}
	
	//**************************************************************
	//Chebyshev Polynomials
	//**************************************************************
	
	double chebyshev1(int n, double x){
		if(n<0) throw std::runtime_error("chebyshev1(int,double): invalid order");
		else if(n==0) return 1;
		else {
			double rm2=1.0,rm1=x,r=x;
			for(int i=2; i<=n; ++i){
				r=2*x*rm1-rm2;
				rm2=rm1; rm1=r;
			}
			return r;
		}
	}
	
	double chebyshev2(int n, double x){
		if(n<0) throw std::runtime_error("chebyshev2(int,double): invalid order");
		else if(n==0) return 1;
		else {
			double rm2=1,rm1=2.0*x,r=2.0*x;
			for(int i=2; i<=n; ++i){
				r=2.0*x*rm1-rm2;
				rm2=rm1; rm1=r;
			}
			return r;
		}
	}
	
	std::vector<double>& chebyshev1_c(int n, double x, std::vector<double>& r){
		if(r.size()!=n+1) throw std::invalid_argument("Invalid vector size.");
		r[0]=1;
		if(n>=1){
			r[1]=x;
			for(int i=2; i<=n; ++i){
				r[i]=2*x*r[i-1]-r[i-2];
			}
		}
		return r;
	}
	
	std::vector<double>& chebyshev2_c(int n, double x, std::vector<double>& r){
		if(r.size()!=n+1) throw std::invalid_argument("Invalid vector size.");
		r[0]=1;
		if(n>=1){
			r[1]=2*x;
			for(int i=2; i<=n; ++i){
				r[i]=2*x*r[i-1]-r[i-2];
			}
		}
		return r;
	}
	
	std::vector<double>& chebyshev1_r(int n, std::vector<double>& r){
		r.resize(n);
		for(int i=0; i<n; i++) r[i]=cos((2.0*i+1.0)/(2.0*n)*constant::PI);
		return r;
	}
	
	std::vector<double>& chebyshev2_r(int n, std::vector<double>& r){
		r.resize(n);
		for(int i=0; i<n; i++) r[i]=cos((i+1.0)/(n+1.0)*constant::PI);
		return r;
	}
	
	//**************************************************************
	//Jacobi Polynomials
	//**************************************************************
	
	double jacobi(int n, double a, double b, double x){
		if(n==0) return 1;
		else if(n==1) return 0.5*(2*(a+1)+(a+b+2)*(x-1));
		else return 
			(2*n+a+b-1)*((2*n+a+b)*(2*n+a+b-2)*x+a*a-b*b)/(2*n*(n+a+b)*(2*n+a+b-2))*jacobi(n-1,a,b,x)
			-(n+a-1)*(n+b-1)*(2*n+a+b)/(n*(n+a+b)*(2*n+a+b-2))*jacobi(n-2,a,b,x);
	}
	
	std::vector<double>& jacobi(int n, double a, double b, std::vector<double>& c){
		if(n==0) c.resize(1,1);
		else if(n==1){
			c.resize(2,0);
			c[0]=0.5*(a-b);
			c[1]=0.5*(a+b+2);
		} else {
			c.resize(n+1,0.0);
			std::vector<double> ct1(n+1,0.0),ct2(n+1,0.0);
			ct1[0]=1.0;
			ct2[0]=0.5*(a-b);
			ct2[1]=0.5*(a+b+2);
			for(int m=2; m<=n; ++m){
				c[0]=0.0;
				for(int l=m; l>0; --l) c[l]=(2*m+a+b-1)*(2*m+a+b)/(2*m*(m+a+b))*ct2[l-1];
				for(int l=m; l>=0; --l) c[l]+=(2*m+a+b-1)*(a*a-b*b)/(2*m*(m+a+b)*(2*m+a+b-2))*ct2[l];
				for(int l=m; l>=0; --l) c[l]-=(m+a-1)*(m+b-1)*(2*m+a+b)/(m*(m+a+b)*(2*m+a+b-2))*ct1[l];
				ct1=ct2; ct2=c;
			}
		}
		return c;
	}
	
	//**************************************************************
	//Laguerre Polynomials
	//**************************************************************
	
	double laguerre(int n, double x){
		if(n<0) throw std::runtime_error("chebyshev2(int,double): invalid order");
		else if(n==0) return 1;
		else {
			double rm2=1.0,rm1=1.0-x,r=1.0-x;
			for(int i=2; i<=n; ++i){
				r=((2.0*i-1.0-x)*rm1-(i-1.0)*rm2)/i;
				rm2=rm1; rm1=r;
			}
			return r;
		}
	}
	
	std::vector<double>& laguerre(int n, std::vector<double>& c){
		if(n==0) c.resize(1,1);
		else if(n==1){
			c.resize(2,1);
			c[1]=-1;
		} else {
			c.resize(n+1,0);
			std::vector<double> ct1(n+1,0.0),ct2(n+1,0.0);
			ct1[0]=1;
			ct2[0]=1; ct2[1]=-1;
			for(int m=2; m<=n; ++m){
				c[0]=0;
				for(int l=m; l>0; --l) c[l]=-1.0/m*ct2[l-1];
				for(int l=m; l>=0; --l) c[l]+=(2.0*m-1.0)/m*ct2[l];
				for(int l=m; l>=0; --l) c[l]-=(m-1.0)/m*ct1[l];
				ct1=ct2; ct2=c;
			}
		}
		return c;
	}
	
}

}