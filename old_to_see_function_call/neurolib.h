#pragma once
//this is an implementation header

enum class funcType { Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Softplus, Softsign };

class tst
{
public:
	double aa();
//private:
//	double bb();
};
double fnneuroLib();

void* createNeuronNetwork(funcType _functionType, double _alpha = 0);
void deleteNeuronNetwork(void* neuronInstance);
double calc(void* neuronInstance, double x);
