#include "simpleneuronetworkclass.h"

simpleNeuroNetworkClass::simpleNeuroNetworkClass(funcType _functionType, double _alpha)
{
    alpha = _alpha;
    switch (_functionType)
    {
    case funcType::ELU:
        activationFunc = &simpleNeuroNetworkClass::funcELU;
        dactivationFunc = &simpleNeuroNetworkClass::dfuncELU;
        break;
    case funcType::LeakyReLU:
        activationFunc = &simpleNeuroNetworkClass::funcLeakyReLU;
        dactivationFunc = &simpleNeuroNetworkClass::dfuncLeakyReLU;
        break;
    case funcType::ReLU:
        activationFunc = &simpleNeuroNetworkClass::funcReLU;
        dactivationFunc = &simpleNeuroNetworkClass::dfuncReLU;
        break;
    case funcType::Softplus:
        activationFunc = &simpleNeuroNetworkClass::funcSoftplus;
        dactivationFunc = &simpleNeuroNetworkClass::dfuncSoftplus;
        break;
    case funcType::Softsign:
        activationFunc = &simpleNeuroNetworkClass::funcSoftsign;
        dactivationFunc = &simpleNeuroNetworkClass::dfuncSoftsign;
        break;
    case funcType::Tanh:
        activationFunc = &simpleNeuroNetworkClass::funcTanh;
        dactivationFunc = &simpleNeuroNetworkClass::dfuncTanh;
        break;
    case funcType::Sigmoid:
    default:
        activationFunc = &simpleNeuroNetworkClass::funcSigmoid;
        dactivationFunc = &simpleNeuroNetworkClass::dfuncSigmoid;
    }
}

simpleNeuroNetworkClass::~simpleNeuroNetworkClass()
{
    delete[] network[numOfInternalLayers+1].values;
    for(int i=numOfInternalLayers; i>=0; i--)
    {
        delete[] network[i].weights;
        delete [] network[i].values;
    };
    delete[] network;
}

void simpleNeuroNetworkClass::createInputLayer(int numOfNeurons)
{
//    numOfInputNeurons = numOfNeurons;
//    inputLayer = new neuronStruct[numOfInputNeurons];
}

void simpleNeuroNetworkClass::createNetwork(int _numOfInternalLayers, int _numOfNeurons[])
{
    numOfInternalLayers = _numOfInternalLayers;
    network = new layerStruct[numOfInternalLayers];
    for(int i=0; i<numOfInternalLayers+1; i++)
    {
        network[i].numOfNeurons = _numOfNeurons[i];
        network[i].values = new double[_numOfNeurons[i]];
        if(i!=numOfInternalLayers+1)                //в выходном слое не надо создавать веса
        {
            network[i].numOfNextLevelNeurons = _numOfNeurons[i+1];
            network[i].weights = new double[_numOfNeurons[i+1]*_numOfNeurons[i]];
        }
    }
}
