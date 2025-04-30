#pragma once
#ifndef NN_HPP
#define NN_HPP

#define EIGEN_NO_DEBUG

// c++ libraries
#include <iosfwd>
// eigen
#include <Eigen/Dense>
// ann - math
#include "ann_random.h"
// ann - mem
#include "ann_serialize.h"

namespace NN{

typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecXd;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatXd;
	
//***********************************************************************
// COMPILER DIRECTIVES
//***********************************************************************

#ifndef NN_PRINT_FUNC
#define NN_PRINT_FUNC 0
#endif

#ifndef NN_PRINT_STATUS
#define NN_PRINT_STATUS 0
#endif

#ifndef NN_PRINT_DATA
#define NN_PRINT_DATA 0
#endif

//***********************************************************************
// FORWARD DECLARATIONS
//***********************************************************************

class ANN;
class ANNP;
class Cost;
class DODZ;
class DODP;

//***********************************************************************
// INITIALIZATION METHOD
//***********************************************************************

class Init{
public:
	//enum
	enum Type{
		UNKNOWN=0,
		RAND=1,
		LECUN=2,
		HE=3,
		XAVIER=4
	};
	//constructor
	Init():t_(Type::UNKNOWN){}
	Init(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Init read(const char* str);
	static const char* name(const Init& init);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Init& init);

//***********************************************************************
// TRANSFER FUNCTIONS 
//***********************************************************************

class Neuron{
public:
	//type
	enum Type{
		UNKNOWN,
		//linear
		LINEAR,
		//sigmoidal
		SIGMOID,
		TANH,
		ISRU,
		ARCTAN,
		RELU,
		ELU,
		TANHRE,
		SQRE,
		//gated-switch
		SWISH,
		GELU,
		MISH,
		PFLU,
		LOGISH,
		//switch
		SOFTPLUS,
		SQPLUS,
		ATISH,
		//test
		TEST
	};
	//constructor
	Neuron():t_(Type::UNKNOWN){}
	Neuron(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Neuron read(const char* str);
	static const char* name(const Neuron& tf);
	//function
	//linear
	static void tf_lin(double c, const VecXd& z, VecXd& a, VecXd& d);
	//sigmoidal
	static void tf_sigmoid(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void tf_tanh(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void tf_isru(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void tf_arctan(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void tf_relu(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void tf_elu(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void tf_tanhre(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void tf_sqre(double c, const VecXd& z, VecXd& a, VecXd& d);
	//gated-switch
	static void tf_swish(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void tf_gelu(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void tf_mish(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void tf_pflu(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void tf_logish(double c, const VecXd& z, VecXd& a, VecXd& d);
	//switch
	static void tf_softplus(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void tf_sqplus(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void tf_atish(double c, const VecXd& z, VecXd& a, VecXd& d);
	//test
	static void tf_test(double c, const VecXd& z, VecXd& a, VecXd& d);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Neuron& tf);

//***********************************************************************
// ANN
//***********************************************************************

/*
DEFINITIONS:
	ensemble - total set of all data (e.g. training "ensemble")
	element - single datum from ensemble
	c - "c" donotes the cost function, e.g. the gradient of the cost function w.r.t. the value of a node is dc/da
	z - "z" is the input to each node, e.g. the gradient of a node w.r.t. to its input is da/dz
	a - "a" is the value of a node, e.g. the gradient of a node w.r.t. to its input is da/dz
	o - "o" is the output of the network (i.e. out_), e.g. the gradient of the output w.r.t. the input is do/di
	i - "i" is the input of the network (i.e. inp_), e.g. the gradient of the output w.r.t. the input is do/di
PRIVATE:
	VecXd inp_ - raw input data for a single element of the ensemble (e.g. training set)
	VecXd inw_ - weight used to scale the input data
	VecXd inb_ - bias used to shift the input data
	VecXd out_ - raw output data given a single input element
	VecXd outw_ - weight used to scale the output data
	VecXd outb_ - bias used to shift the output data
	int nlayer_ - 
		total number of hidden layers
		best thought of as the number of "connections" between layers
		nlayer_ must be greater than zero for an initialized network
		this is true even for a network with zero "hidden" layers
		an uninitialized network has nlayer_ = 0
		if we have just the input and output: nlayer_ = 1
			one set of weights,biases connecting input/output
		if we have one hidden layer: nlayer_ = 2
			two sets of weights,biases connecting input/layer0/output
		if we have two hidden layers: nlayer_ = 3
			three sets of weights,biases connecting input/layer0/layer1/output
		et cetera
	std::vector<VecXd> node_ - 
		all nodes, including the input, output, and hidden layers
		the raw input and output (inp_,out_) are separate from "node_"
		this is because the raw input/output may be shifted/scaled before being used
		thus, while inp_/out_ are the "raw" input/output,
		the front/back of "node_" can be thought of the "scaled" input/output
		note that scaling is not necessary, but made optional with the use of inp_/out_
		has a size of "nlayer_+1", as there are "nlayer_" connections between "nlayer_+1" nodes
	std::vector<VecXd> bias_ - 
		the bias of each layer, best thought of as the bias "between" layers n,n+1
		bias_[n] must have the size node_[n+1] - we add this bias when going from node_[n] to node_[n+1]
		has a size of "nlayer_", as there are "nlayer_" connections between "nlayer_+1" nodes
	std::vector<MatXd> edge_ -
		the weights of each layer, best though of as transforming from layers n to n+1
		edge_[n] must have the size (node_[n+1],node_[n]) - matrix multiplying (node_[n]) to get (node_[n+1])
		has a size of "nlayer_", as there are "nlayer_" connections between "nlayer_+1" nodes
	std::vector<VecXd> dadz_ - 
		the gradient of the value of a node (a) w.r.t. the input of the node (z) - da/dz
		practically, the gradient of the transfer function of each layer
		best thought of as the gradient associated with function transferring "between" layers n,n+1
		thus, dadz_[n] must have the size node_[n+1]
		has a size of "nlayer_", as there are "nlayer_" connections between "nlayer_+1" nodes
	neuron_ -
		the type of the transfer function
		note the transfer function for the last layer is always linear
	neuronp_ - 
		(Transfer Function, Function Derivative, Vector)
		the transfer function for each layer, operates on entire vector at once
		computes both function and derivative simultaneously
*/
class ANN{
private:
	//typedefs
		typedef void (*NeuronP)(double,const VecXd&,VecXd&,VecXd&);
	//layers
		int nlayer_;//number of layers (weights,biases)
		double c_;//sharpness parameter
	//transfer functions
		Neuron neuron_;//transfer function type
		std::vector<NeuronP> neuronp_;//transfer function - input for indexed layer (nlayer_)
	//input/output
		VecXd inp_;//input - raw
		VecXd ins_;//input - scaled and shifted
		VecXd out_;//output
		VecXd inpw_,inpb_;//input weight, bias
		VecXd outw_,outb_;//output weight, bias
	//gradients - nodes
		std::vector<VecXd> dadz_;//node derivative - not including input layer (nlayer_)
	//node weights and biases
		std::vector<VecXd> a_;//node values
		std::vector<VecXd> z_;//node inputs
		std::vector<VecXd> b_;//biases
		std::vector<MatXd> w_;//weights
public:
	//==== constructors/destructors ====
	ANN(){defaults();}
	~ANN(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const ANN& n);
	friend FILE* operator<<(FILE* out, const ANN& n);
	friend VecXd& operator>>(const ANN& nn, VecXd& v);
	friend ANN& operator<<(ANN& nn, const VecXd& v);
	
	//==== access ====
	//network dimensions
		int nlayer()const{return nlayer_;}
		double& c(){return c_;}
		const double& c()const{return c_;}
	//nodes
		VecXd& inp(){return inp_;}
		const VecXd& inp()const{return inp_;}
		VecXd& ins(){return ins_;}
		const VecXd& ins()const{return ins_;}
		VecXd& out(){return out_;}
		const VecXd& out()const{return out_;}
	//scaling
		VecXd& inpw(){return inpw_;}
		const VecXd& inpw()const{return inpw_;}
		VecXd& inpb(){return inpb_;}
		const VecXd& inpb()const{return inpb_;}
		VecXd& outw(){return outw_;}
		const VecXd& outw()const{return outw_;}
		VecXd& outb(){return outb_;}
		const VecXd& outb()const{return outb_;}
	//nodes
		VecXd& a(int l){return a_[l];}
		const VecXd& a(int l)const{return a_[l];}
		const std::vector<VecXd>& a()const{return a_;}
		VecXd& z(int l){return z_[l];}
		const VecXd& z(int l)const{return z_[l];}
		const std::vector<VecXd>& z()const{return z_;}
	//bias
		VecXd& b(int l){return b_[l];}
		const VecXd& b(int l)const{return b_[l];}
		const std::vector<VecXd>& b()const{return b_;}
	//edge
		MatXd& w(int l){return w_[l];}
		const MatXd& w(int l)const{return w_[l];}
		const std::vector<MatXd>& w()const{return w_;}
	//size
		int nInp()const{return inp_.size();}
		int nOut()const{return out_.size();}
		int nNodes(int n)const{return a_[n].size();}
	//gradients - nodes
		VecXd& dadz(int n){return dadz_[n];}
		const VecXd& dadz(int n)const{return dadz_[n];}
		const std::vector<VecXd>& dadz()const{return dadz_;}
	//transfer functions
		Neuron& neuron(){return neuron_;}
		const Neuron& neuron()const{return neuron_;}
		NeuronP neuronp(int l){return neuronp_[l];}
		const NeuronP neuronp(int l)const{return neuronp_[l];}
		
	//==== member functions ====
	//clearing/initialization
		void defaults();
		void clear();
	//info
		int size()const;
		int nBias()const;
		int nWeight()const;
	//resizing
		void resize(const ANNP& init, int nInput, int nOutput);
		void resize(const ANNP& init, int nInput, const std::vector<int>& nNodes, int nOutput);
		void resize(const ANNP& init, const std::vector<int>& nNodes);
		void resize(const ANNP& init, int nInput, const std::vector<int>& nNodes);
	//error
		double error_lambda()const;
		VecXd& grad_lambda(VecXd& grad)const;
	//execution
		const VecXd& execute();
		const VecXd& execute(const VecXd& in){inp_=in;return execute();}
		
	//==== static functions ====
	static void write(FILE* writer, const ANN& nn);
	static void write(const char*, const ANN& nn);
	static void read(FILE* writer, ANN& nn);
	static void read(const char*, ANN& nn);
};

//***********************************************************************
// ANNP
//***********************************************************************

class ANNP{
private:
	int seed_;//random seed	
	rng::dist::Name dist_;//distribution type
	Init init_;//initialization scheme
	Neuron neuron_;//nueron type
	double bInit_;//initial value - bias
	double wInit_;//initial value - weight
	double sigma_;//distribution size parameter
	double c_;//sharpness
public:
	//==== constructors/destructors ====
	ANNP(){defaults();}
	~ANNP(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const ANNP& init);
	
	//==== access ====
	int& seed(){return seed_;}
	const int& seed()const{return seed_;}
	rng::dist::Name& dist(){return dist_;}
	const rng::dist::Name& dist()const{return dist_;}
	Init& init(){return init_;}
	const Init& init()const{return init_;}
	Neuron& neuron(){return neuron_;}
	const Neuron& neuron()const{return neuron_;}
	double& bInit(){return bInit_;}
	const double& bInit()const{return bInit_;}
	double& wInit(){return wInit_;}
	const double& wInit()const{return wInit_;}
	double& sigma(){return sigma_;}
	const double& sigma()const{return sigma_;}
	double& c(){return c_;}
	const double& c()const{return c_;}
	
	//==== member functions ====
	void defaults();
	void clear(){defaults();}
	
	//==== static functions ====
	static void read(const char*, ANNP& annp);
	static void read(FILE* writer, ANNP& annp);
};

//***********************************************************************
// Cost
//***********************************************************************

/*
dcdz_ - 
	the gradient of the cost function (c) w.r.t. the node inputs (z) - dc/dz
*/
class Cost{
private:
	std::vector<VecXd> dcdz_;//derivative of cost function w.r.t. node inputs (nlayer_)
	VecXd grad_;//gradient of the cost function with respect to each parameter (bias + weight)
public:
	//==== constructors/destructors ====
	Cost(){}
	Cost(const ANN& nn){resize(nn);}
	~Cost(){}
	
	//==== access ====
	std::vector<VecXd>& dcdz(){return dcdz_;}
	const std::vector<VecXd>& dcdz()const{return dcdz_;}
	VecXd& grad(){return grad_;}
	const VecXd& grad()const{return grad_;}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	const VecXd& grad(const ANN& nn, const VecXd& dcdo);
};

//***********************************************************************
// DODZ
//***********************************************************************

/*
dodz_ -
	the derivative of the output (o) w.r.t. the input (z)
dodi_ -
	the derivative of the output w.r.t. the raw input
*/
class DODZ{
private:
	MatXd dodi_;//derivative of out_ w.r.t. to inp_ (out_.size() x inp_.size())
	std::vector<MatXd> dodz_;//derivative of out_ w.r.t. to the value "a" of all nodes (nlayer_+1)
public:
	//==== constructors/destructors ====
	DODZ(){}
	DODZ(const ANN& nn){resize(nn);}
	~DODZ(){}
	
	//==== access ====
	MatXd& dodi(){return dodi_;}
	const MatXd& dodi()const{return dodi_;}
	MatXd& dodz(int n){return dodz_[n];}
	const MatXd& dodz(int n)const{return dodz_[n];}
	std::vector<MatXd>& dodz(){return dodz_;}
	const std::vector<MatXd>& dodz()const{return dodz_;}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	void grad(const ANN& nn);
};

//***********************************************************************
// DODP
//***********************************************************************

class DODP{
private:
	std::vector<MatXd> dodz_;//derivative of output w.r.t. node inputs (nlayer_)
	std::vector<std::vector<VecXd> > dodb_;//derivative of output w.r.t. biases
	std::vector<std::vector<MatXd> > dodw_;//derivative of output w.r.t. weights
public:
	//==== constructors/destructors ====
	DODP(){}
	DODP(const ANN& nn){resize(nn);}
	~DODP(){}
	
	//==== access ====
	MatXd& dodz(int n){return dodz_[n];}
	const MatXd& dodz(int n)const{return dodz_[n];}
	std::vector<MatXd>& dodz(){return dodz_;}
	const std::vector<MatXd>& dodz()const{return dodz_;}
	MatXd& dodb(int n){return dodz_[n];}
	const MatXd& dodb(int n)const{return dodz_[n];}
	std::vector<std::vector<VecXd> >& dodb(){return dodb_;}
	const std::vector<std::vector<VecXd> >& dodb()const{return dodb_;}
	std::vector<std::vector<MatXd> >& dodw(){return dodw_;}
	const std::vector<std::vector<MatXd> >& dodw()const{return dodw_;}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	void grad(const ANN& nn);
};

//***********************************************************************
// DZDI
//***********************************************************************

class DZDI{
private:
	std::vector<MatXd> dzdi_;
public:
	//==== constructors/destructors ====
	DZDI(){}
	DZDI(const ANN& nn){resize(nn);}
	~DZDI(){}
	
	//==== access ====
	const std::vector<MatXd>& dzdi(){return dzdi_;}
	MatXd& dzdi(int n){return dzdi_[n];}
	const MatXd& dzdi(int n)const{return dzdi_[n];}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	void grad(const ANN& nn);
};

}

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NN::ANNP& obj);
	template <> int nbytes(const NN::ANN& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NN::ANNP& obj, char* arr);
	template <> int pack(const NN::ANN& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NN::ANNP& obj, const char* arr);
	template <> int unpack(NN::ANN& obj, const char* arr);
	
}

#endif