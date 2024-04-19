// neuroLib.cpp : Определяет функции для статической библиотеки.
//

#include "pch.h"
#include "framework.h"

#include "global.h"
#include "neuronclasses.h"

// TODO: Это пример библиотечной функции.
double fnneuroLib()
{
	byte a;
	word b;
	dword c;
	double (*myFunc) (double x);

	double d = 0;
//	d = 5+funcSoftsign(d);
//	a = d;
//	myFunc = funcSigmoid;
//	myFunc(8);
//	myFunc = funcSigmoid;
//	return(myFunc(8));
	return(100);
}

void* createNeuronNetwork(funcType _functionType = funcType::Sigmoid, double _alpha = 0)
{
	classNeuronNetwork* cnw = new classNeuronNetwork();

	cnw->alpha = _alpha;
	switch (_functionType)
	{
	case funcType::ELU:
		cnw->activationFunc = &classNeuronNetwork::funcELU;
		cnw->dactivationFunc = &classNeuronNetwork::dfuncELU;
		break;
	case funcType::LeakyReLU:
		cnw->activationFunc = &classNeuronNetwork::funcLeakyReLU;
		cnw->dactivationFunc = &classNeuronNetwork::dfuncLeakyReLU;
		break;
	case funcType::ReLU:
		cnw->activationFunc = &classNeuronNetwork::funcReLU;
		cnw->dactivationFunc = &classNeuronNetwork::dfuncReLU;
		break;
	case funcType::Softplus:
		cnw->activationFunc = &classNeuronNetwork::funcSoftplus;
		cnw->dactivationFunc = &classNeuronNetwork::dfuncSoftplus;
		break;
	case funcType::Softsign:
		cnw->activationFunc = &classNeuronNetwork::funcSoftsign;
		cnw->dactivationFunc = &classNeuronNetwork::dfuncSoftsign;
		break;
	case funcType::Tanh:
		cnw->activationFunc = &classNeuronNetwork::funcTanh;
		cnw->dactivationFunc = &classNeuronNetwork::dfuncTanh;
		break;
	case funcType::Sigmoid:
	default:
		cnw->activationFunc = &classNeuronNetwork::funcSigmoid;
		cnw->dactivationFunc = &classNeuronNetwork::dfuncSigmoid;
	}
	return cnw;
}

void deleteNeuronNetwork(void* neuronInstance)
{
	classNeuronNetwork *cnw = (classNeuronNetwork*)neuronInstance;
	delete cnw;
}

double calc(void* neuronInstance, double x)
{
	classNeuronNetwork* cnw = (classNeuronNetwork*)neuronInstance;
	return (cnw->*(cnw->activationFunc))(x);
}



