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

    network.setFunctionType(funcType::Sigmoid);
/*
    int qwerty1[2];
    int qwerty2;
    int *qwerty3;
    network.createNetwork(2,1,1,qwerty1);
    network.createNetwork(2,1,1,&qwerty2);
    network.createNetwork(2,1,1,qwerty3);
*/
    std::cout << "test" << std::endl;

    int qq=2;
    network.createNetwork(3,1,1,&qq);
    std::cout << "created" << std::endl;

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
}
