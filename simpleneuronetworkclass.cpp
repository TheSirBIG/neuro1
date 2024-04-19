#include "simpleneuronetworkclass.h"

/*
void simpleNeuroNetworkClass::setFunctionType(funcType _functionType, double _alpha)
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
*/

void simpleNeuroNetworkClass::setActivationFunc(funcType *types, funcType outtype)
{
    //для внутренних и выходного
    for(int i=0;i<numOfInternalLayers;i++)
    {
        switch (types[i])
        {
        case funcType::ELU:
            internalLayer[i]->activationFunc = &simpleNeuroNetworkClass::funcELU;
            internalLayer[i]->dactivationFunc = &simpleNeuroNetworkClass::dfuncELU;
            break;
        case funcType::LeakyReLU:
            internalLayer[i]->activationFunc = &simpleNeuroNetworkClass::funcLeakyReLU;
            internalLayer[i]->dactivationFunc = &simpleNeuroNetworkClass::dfuncLeakyReLU;
            break;
        case funcType::ReLU:
            internalLayer[i]->activationFunc = &simpleNeuroNetworkClass::funcReLU;
            internalLayer[i]->dactivationFunc = &simpleNeuroNetworkClass::dfuncReLU;
            break;
        case funcType::Softplus:
            internalLayer[i]->activationFunc = &simpleNeuroNetworkClass::funcSoftplus;
            internalLayer[i]->dactivationFunc = &simpleNeuroNetworkClass::dfuncSoftplus;
            break;
        case funcType::Softsign:
            internalLayer[i]->activationFunc = &simpleNeuroNetworkClass::funcSoftsign;
            internalLayer[i]->dactivationFunc = &simpleNeuroNetworkClass::dfuncSoftsign;
            break;
        case funcType::Tanh:
            internalLayer[i]->activationFunc = &simpleNeuroNetworkClass::funcTanh;
            internalLayer[i]->dactivationFunc = &simpleNeuroNetworkClass::dfuncTanh;
            break;
        case funcType::Sigmoid:
        default:
            internalLayer[i]->activationFunc = &simpleNeuroNetworkClass::funcSigmoid;
            internalLayer[i]->dactivationFunc = &simpleNeuroNetworkClass::dfuncSigmoid;
        }
    }

    switch (outtype)
    {
    case funcType::ELU:
        outputLayer->activationFunc = &simpleNeuroNetworkClass::funcELU;
        outputLayer->dactivationFunc = &simpleNeuroNetworkClass::dfuncELU;
        break;
    case funcType::LeakyReLU:
        outputLayer->activationFunc = &simpleNeuroNetworkClass::funcLeakyReLU;
        outputLayer->dactivationFunc = &simpleNeuroNetworkClass::dfuncLeakyReLU;
        break;
    case funcType::ReLU:
        outputLayer->activationFunc = &simpleNeuroNetworkClass::funcReLU;
        outputLayer->dactivationFunc = &simpleNeuroNetworkClass::dfuncReLU;
        break;
    case funcType::Softplus:
        outputLayer->activationFunc = &simpleNeuroNetworkClass::funcSoftplus;
        outputLayer->dactivationFunc = &simpleNeuroNetworkClass::dfuncSoftplus;
        break;
    case funcType::Softsign:
        outputLayer->activationFunc = &simpleNeuroNetworkClass::funcSoftsign;
        outputLayer->dactivationFunc = &simpleNeuroNetworkClass::dfuncSoftsign;
        break;
    case funcType::Tanh:
        outputLayer->activationFunc = &simpleNeuroNetworkClass::funcTanh;
        outputLayer->dactivationFunc = &simpleNeuroNetworkClass::dfuncTanh;
        break;
    case funcType::Sigmoid:
    default:
        outputLayer->activationFunc = &simpleNeuroNetworkClass::funcSigmoid;
        outputLayer->dactivationFunc = &simpleNeuroNetworkClass::dfuncSigmoid;
    }

}


void simpleNeuroNetworkClass::setWeights(double *input, double *internal)
{
    //входной слой
    //сначала все веса от первого нейрона, и т.д.
    for(int k=0; k<inputLayer->numOfNeurons; k++)
        for(int i=0; i<inputLayer->numOfNextLevelNeurons; i++)
        {
            inputLayer->weights[k][i] = *input;
            input++;
        }
    //внутренние слои
    //сначала для 1-го слоя как у входного, и т.д.
    for(int k=0; k<numOfInternalLayers; k++)
        for(int j=0; j<internalLayer[k]->numOfNeurons; j++)
            for(int i=0; i<internalLayer[k]->numOfNextLevelNeurons; i++)
            {
                internalLayer[k]->weights[j][i] = *internal;
                internal++;
            }
}

void simpleNeuroNetworkClass::setB(double *input, double *internal)
{
    //входной слой
    for(int i=0; i<inputLayer->numOfNextLevelNeurons; i++)
    {
        inputLayer->b[i] = *input;
        input++;
    }
    //внутренние слои
    //сначала смещения для 1-го слоя, и т.д.
    for(int k=0; k<numOfInternalLayers; k++)
        for(int i=0; i<internalLayer[k]->numOfNextLevelNeurons; i++)
        {
            internalLayer[k]->b[i] = *internal;
            internal++;
        }
}

void simpleNeuroNetworkClass::setAlpha(double _alpha)
{
    alpha = _alpha;
}

void simpleNeuroNetworkClass::setInitialValues(double input[])
{
    for(int i=0; i<inputLayer->numOfNeurons; i++)
        inputLayer->values[i] = input[i];
}

void simpleNeuroNetworkClass::getOutputValues(double output[])
{
    for(int i=0; i<outputLayer->numOfNeurons; i++)
        output[i] = outputLayer->values[i];
}

void simpleNeuroNetworkClass::Calculate()
{
    layerStruct *prevLayer;
    double summ;

    for(int k=0; k<numOfInternalLayers; k++)
    {
        if(k==0) prevLayer = inputLayer; else prevLayer = internalLayer[k-1];
        for(int j=0; j<internalLayer[k]->numOfNeurons; j++)
        {
            summ = 0;
            if(useB) summ = prevLayer->b[j];
            for(int i=0; i<prevLayer->numOfNeurons; i++)
                summ += prevLayer->values[i]*prevLayer->weights[i][j];
//            internalLayer[k]->values[j] = (this->*activationFunc)(summ);
            internalLayer[k]->values[j] = (this->*(internalLayer[k]->activationFunc))(summ);
        }
    }
    prevLayer = internalLayer[numOfInternalLayers-1];
    for(int j=0; j<outputLayer->numOfNeurons; j++)
    {
        summ = 0;
        if(useB) summ = prevLayer->b[j];
        for(int i=0; i<prevLayer->numOfNeurons; i++)
            summ += prevLayer->values[i]*prevLayer->weights[i][j];
//        outputLayer->values[j] = (this->*activationFunc)(summ);
        outputLayer->values[j] = (this->*(outputLayer->activationFunc))(summ);
    }
}

void simpleNeuroNetworkClass::correctWeights(double wanted_output[])
{
    double *delta;

    int num = outputLayer->numOfNeurons;
    delta = new double[num];
    for(int i=0; i<num; i++)
        delta[i] = outputLayer->values[i] - wanted_output[i];

}

void simpleNeuroNetworkClass::setUsingB(bool mustUseB)
{
    useB = mustUseB;
}

simpleNeuroNetworkClass::simpleNeuroNetworkClass()
{
//    setFunctionType(_functionType, _alpha);
}

simpleNeuroNetworkClass::~simpleNeuroNetworkClass()
{
    if(outputLayer!=NULL)
    {
        delete[] outputLayer->values;
        delete outputLayer;
    }

    if(internalLayer != NULL)
    {
        for(int k=0; k<numOfInternalLayers; k++)
        {
            for(int i=0; i<internalLayer[k]->numOfNeurons; i++)
            {
                delete[] internalLayer[k]->weights[i];
            }
            delete[] internalLayer[k]->weights;
            delete[] internalLayer[k]->values;
            delete[] internalLayer[k]->b;
            delete internalLayer[k];
        }
        delete[] internalLayer;
    }

    if(inputLayer != NULL)
    {
        for(int i=0; i<inputLayer->numOfNeurons; i++)
        {
            delete[] inputLayer->weights[i];
        }
        delete[] inputLayer->weights;
        delete[] inputLayer->values;
        delete[] inputLayer->b;
        delete inputLayer;
    }
}

void simpleNeuroNetworkClass::createNetwork(int _numInput, int _numInternal, int _numOutput, int _numInEachInternal[])
{
//    numOfInputNeurons = _numInput;
//    numOfOutputNeurons = _numOutput;
    numOfInternalLayers = _numInternal;

    inputLayer = new layerStruct;
    inputLayer->numOfNeurons = _numInput;
    inputLayer->numOfNextLevelNeurons = _numInEachInternal[0];
    inputLayer->values = new double[inputLayer->numOfNeurons];
    inputLayer->b = new double[inputLayer->numOfNextLevelNeurons];
    for(int i=0; i<inputLayer->numOfNextLevelNeurons; i++)
        inputLayer->b[i] = 0;
    inputLayer->weights = new double*[inputLayer->numOfNeurons];
    for(int i=0; i<inputLayer->numOfNeurons; i++)
    {
        inputLayer->weights[i] = new double[inputLayer->numOfNextLevelNeurons];
    }

    internalLayer = new layerStruct*[numOfInternalLayers];
    for(int k=0; k<numOfInternalLayers; k++)
    {
        internalLayer[k] = new layerStruct;
        internalLayer[k]->numOfNeurons = _numInEachInternal[k];
        if(k==numOfInternalLayers-1) internalLayer[k]->numOfNextLevelNeurons = _numOutput;
            else internalLayer[k]->numOfNextLevelNeurons = _numInEachInternal[k+1];
        internalLayer[k]->values = new double[internalLayer[k]->numOfNeurons];
        internalLayer[k]->b = new double[internalLayer[k]->numOfNextLevelNeurons];
        for(int i=0; i<internalLayer[k]->numOfNextLevelNeurons; i++)
            internalLayer[k]->b[i] = 0;
        internalLayer[k]->weights = new double*[internalLayer[k]->numOfNeurons];
        for(int i=0; i<internalLayer[k]->numOfNeurons; i++)
        {
            internalLayer[k]->weights[i] = new double[internalLayer[k]->numOfNextLevelNeurons];
        }
    }

    outputLayer = new layerStruct;
    outputLayer->numOfNeurons = _numOutput;
    outputLayer->values = new double[outputLayer->numOfNeurons];
}
