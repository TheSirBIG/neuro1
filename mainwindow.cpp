#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "simpleneuronetworkclass.h"
#include <iostream>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_released()
{
    simpleNeuroNetworkClass network;

//    network.setFunctionType(funcType::Sigmoid);
/*
    int qwerty1[2];
    int qwerty2;
    int *qwerty3;
    network.createNetwork(2,1,1,qwerty1);
    network.createNetwork(2,1,1,&qwerty2);
    network.createNetwork(2,1,1,qwerty3);
*/
    std::cout << "test" << std::endl;

    network.setUsingB(false);
//    network.setUsingB(true);

    int qq=2;
    network.createNetwork(3,1,1,&qq);
    std::cout << "created" << std::endl;

    funcType fti[1] = {funcType::Sigmoid};
    funcType fto = funcType::Sigmoid;
    network.setActivationFunc(fti,fto);
    std::cout << "activation functions was setted" << std::endl;
    double init[3] = {1,0,1};
    network.setInitialValues(init);
    std::cout << "init vals" << std::endl;

    double w1[6] = {0.43,0.11,0.18,0.27,-0.21,0.31};
    double w2[2] = {0.22,0.47};
    network.setWeights(w1,w2);
    std::cout << "weight set" << std::endl;

    network.Calculate();
    std::cout << "calculated" << std::endl;

    double www;
    www = network.internalLayer[0]->values[0];
    std::cout << www << std::endl;
    www = network.internalLayer[0]->values[1];
    std::cout << www << std::endl;
    double out;
    network.getOutputValues(&out);
    std::cout << out << std::endl;

    www = 0;
    network.correctWeights(&www);
    std::cout << "new weight" << std::endl;
    www = network.internalLayer[0]->weights[0][0];
    std::cout << www << std::endl;
    www = network.internalLayer[0]->weights[1][0];
    std::cout << www << std::endl;

    network.setLearningRate(0.5);
    for(int i=0; i<1000; i++)
    {
        std::cout << "new calc:" << std::endl;
        network.Calculate();
        network.getOutputValues(&out);
        std::cout << out << std::endl;
        www = 0;
        network.correctWeights(&www);
    }


    std::cout << "just test 2d&3d arrays" << std::endl;
    double ar1[3][3][3] = {{{0,1,2},{3,4,5},{6,7,8}},
    {{10,11,12},{13,14,15},{16,17,18}},
    {{20,21,22},{23,24,25},{26,27,28}}};
    double *arp;
    arp=&ar1[0][0][0];
    for(int i=0;i<27;i++)
    {
        std::cout << *arp << ",";
        arp++;
    }
    std::cout << std::endl;

    simpleNeuroNetworkClass network2;

//    network2.setFunctionType(funcType::ReLU);
/*
    int qwerty1[2];
    int qwerty2;
    int *qwerty3;
    network.createNetwork(2,1,1,qwerty1);
    network.createNetwork(2,1,1,&qwerty2);
    network.createNetwork(2,1,1,qwerty3);
*/
    std::cout << "test2" << std::endl;

    network2.setUsingB(false);

    int qq2[2] = {3,2};
    network2.createNetwork(3,2,3,qq2);
    std::cout << "created2" << std::endl;

    funcType fti2[2] = {funcType::ReLU, funcType::ReLU};
    funcType fto2 = funcType::ReLU;
    network2.setActivationFunc(fti2,fto2);
    std::cout << "activation functions2 was setted" << std::endl;

    double init2[3] = {1,2,3};
    network2.setInitialValues(init2);
    std::cout << "init vals2" << std::endl;

    double ww1[9] = {
        1,2,3,
        10,20,30,
        100,200,300
    };
    double ww2[3*2+2*3] = {
        1,2,
        10,20,
        100,200,

        1,2,3,
        10,20,30
    };
    network2.setWeights(ww1,ww2);
    std::cout << "weight set2" << std::endl;

    network2.Calculate();
    std::cout << "calculated" << std::endl;

    www = network2.internalLayer[0]->values[0];
    std::cout << www << std::endl;
    www = network2.internalLayer[0]->values[1];
    std::cout << www << std::endl;
    www = network2.internalLayer[0]->values[2];
    std::cout << www << std::endl;
    www = network2.internalLayer[1]->values[0];
    std::cout << www << std::endl;
    www = network2.internalLayer[1]->values[1];
    std::cout << www << std::endl;
    double out2[3];
    network2.getOutputValues(out2);
    std::cout << out2[0] << "," << out2[1] << "," << out2[2] << std::endl;
}
