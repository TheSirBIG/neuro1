#ifndef SIMPLENEURONETWORKCLASS_H
#define SIMPLENEURONETWORKCLASS_H

#include <cmath>

#define byte __Int8
#define word __Int16
#define dword __Int32

enum class funcType { Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Softplus, Softsign };

typedef struct
{
    int numOfNeurons = 0;                   //число нейронов в слое
    int numOfNextLevelNeurons = 0;          //по сути - количество "синапсов"
    double *values = NULL;                  //динамический массив значений нейрона
    double **weights = NULL;                 //динамический массив значений весов "синапсов"
} layerStruct;

class simpleNeuroNetworkClass
{
public:
    simpleNeuroNetworkClass(funcType _functionType = funcType::Sigmoid, double _alpha = 0);
    ~simpleNeuroNetworkClass();

    double alpha;
    double (simpleNeuroNetworkClass::*activationFunc) (double);
    double (simpleNeuroNetworkClass::*dactivationFunc) (double);

    void createNetwork(int _numInput, int _numInternal, int _numOutput, int _numInEachInternal[]);
    layerStruct *inputLayer = NULL;
    layerStruct **internalLayer = NULL;
    layerStruct *outputLayer = NULL;

    void setFunctionType(funcType _functionType = funcType::Sigmoid, double _alpha = 0);
    void setWeights(double *input, double *internal);
    void setInitialValues(double input[]);
    void getOutputValues(double output[]);
    void Calculate(void);
private:
    int numOfInternalLayers = 1;

    /* функции активации и их производные */
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
};

#endif // SIMPLENEURONETWORKCLASS_H
