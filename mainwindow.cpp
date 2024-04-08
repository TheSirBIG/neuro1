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
}
