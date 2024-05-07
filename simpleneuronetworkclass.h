#ifndef SIMPLENEURONETWORKCLASS_H
#define SIMPLENEURONETWORKCLASS_H

#include <cmath>
#include <iostream>
#include <fstream>

#define byte __Int8
#define word __Int16
#define dword __Int32

enum class funcType { Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Softplus, Softsign };

class simpleNeuroNetworkClass;
typedef struct
{
    int numOfNeurons = 0;                   //число нейронов в слое
    int numOfNextLevelNeurons = 0;          //по сути - количество "синапсов"
    double *values = NULL;                  //динамический массив значений нейрона
    double **weights = NULL;                 //динамический массив значений весов "синапсов"
    double *b = NULL;                       //смещения
    double (simpleNeuroNetworkClass::*activationFunc) (double);
    double (simpleNeuroNetworkClass::*dactivationFunc) (double);
} layerStruct;

class simpleNeuroNetworkClass
{
public:
//    simpleNeuroNetworkClass(funcType _functionType = funcType::Sigmoid, double _alpha = 1);
    simpleNeuroNetworkClass();
    ~simpleNeuroNetworkClass();

    double alpha = 1;
    bool useB = false;
    double learningRate = 0.3;
//    double (simpleNeuroNetworkClass::*activationFunc) (double);
//    double (simpleNeuroNetworkClass::*dactivationFunc) (double);

    void createNetwork(int _numInput, int _numInternal, int _numOutput, int _numInEachInternal[]);
    layerStruct *inputLayer = NULL;
    layerStruct **internalLayer = NULL;
    layerStruct *outputLayer = NULL;

//    void setFunctionType(funcType _functionType = funcType::Sigmoid, double _alpha = 0);
    void setWeights(double *input, double *internal);
    void setB(double *input, double *internal);
    void setAlpha(double _alpha);
    void setActivationFunc(funcType *inttypes, funcType outtype);
    void setInitialValues(double input[]);
    void getOutputValues(double output[]);
    void Calculate(void);
    void setUsingB(bool mustUseB);
    void setLearningRate(double _learningRate);

    void correctWeights(double wanted_output[]);

    bool saveToFile(std::string fileName);
    bool readFromFile(std::string fileName);
private:
    int numOfInternalLayers = 1;

    /* функции активации и их производные */
    double funcSigmoid(double x) { return(1 / (1 + exp(-x))); }
//    double dfuncSigmoid(double x) { return(funcSigmoid(x) * (1 - funcSigmoid(x))); }
    double dfuncSigmoid(double x) { return(x * (1 - x)); }

    double funcTanh(double x) { return((exp(2 * x) - 1) / (exp(2 * x) + 1)); }
//    double dfuncTanh(double x) { return(1 - funcTanh(x) * funcTanh(x)); }
    double dfuncTanh(double x) { return(1 - x * x); }

    double funcReLU(double x) { return(x > 0 ? x : 0); }
//    double dfuncReLU(double x) { return(x > 0 ? 1 : 0); }
    double dfuncReLU(double x) { return(x > 0 ? 1 : 0); }

    double funcLeakyReLU(double x) { return(x > 0 ? x : alpha * x); }
//    double dfuncLeakyReLU(double x) { return(x > 0 ? 1 : alpha); }
    double dfuncLeakyReLU(double x) { return(x > 0 ? 1 : alpha); }

    double funcELU(double x) { return(x > 0 ? x : alpha * (exp(x) - 1)); }
//    double dfuncELU(double x) { return(x > 0 ? 1 : alpha * exp(x)); }
    double dfuncELU(double x) { return(x > 0 ? 1 : alpha * exp(x)); }

    double funcSoftplus(double x) { return(log(1 + exp(x))); }
//    double dfuncSoftplus(double x) { return(1 / (1 + exp(-x))); }
    double dfuncSoftplus(double x) { return(1 / (1 + exp(-x))); }

    double funcSoftsign(double x) { return(x / (1 + abs(x))); }
//    double dfuncSoftsign(double x) { return(1 / ((1 + abs(x)) * (1 + abs(x)))); }
    double dfuncSoftsign(double x) { return(1 / ((1 + abs(x)) * (1 + abs(x)))); }
};

#endif // SIMPLENEURONETWORKCLASS_H
