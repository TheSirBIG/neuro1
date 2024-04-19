#pragma once
#include <cmath>
#include "global.h"

class tst
{
public:
	double aa();
private:
	double bb();
};

class classNeuronNetwork
{
	friend void* createNeuronNetwork(funcType, double);
public:
	//нужно для функций LeakyReLU и ELU!!!
	double alpha;
	double (classNeuronNetwork::*activationFunc) (double);
	double (classNeuronNetwork::*dactivationFunc) (double);
private:
	double funcSigmoid(double x) { return(1 / (1 + exp(-x))); }
	double dfuncSigmoid(double x) { return(funcSigmoid(x) * (1 - funcSigmoid(x))); }

	double funcTanh(double x) { return((exp(2 * x) - 1) / (exp(2 * x) + 1)); }
	double dfuncTanh(double x) { return(1 - funcTanh(x) * funcTanh(x)); }

	double funcReLU(double x) { return(x > 0 ? x : 0); }
	double dfuncReLU(double x) { return(x > 0 ? 1 : 0); }

	double funcLeakyReLU(double x) { return(x > 0 ? x : alpha * x); }
	double dfuncLeakyReLU(double x) { return(x > 0 ? 1 : alpha); }

	double funcELU(double x) { return(x > 0 ? x : alpha * (exp(x) - 1)); }
	double dfuncELU(double x) { return(x > 0 ? 1 : alpha * exp(x)); }

	double funcSoftplus(double x) { return(log(1 + exp(x))); }
	double dfuncSoftplus(double x) { return(1 / (1 + exp(-x))); }

	double funcSoftsign(double x) { return(x / (1 + abs(x))); }
	double dfuncSoftsign(double x) { return(1 / ((1 + abs(x)) * (1 + abs(x)))); }
protected:
};


