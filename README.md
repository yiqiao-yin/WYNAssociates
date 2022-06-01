# W.Y.N. Associates, LLC

[![WYN Associates](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://yinscapital.com/research/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

This is the official quantitative and statistical software package by W.Y.N. Associates, LLC or WYN Associates. Artificial Intelligence (AI) is the logical extension of human will. At WYN, we investigate marketable securities through not only the lens of human experience but also by machine power to discover the undiscovered intrinsic value of securities.

<p align="center">
  <img width="800" src="https://github.com/yiqiao-yin/WYNAssociates/blob/main/figs/maintitle.gif">
</p>
<p align="center">
	<img src="https://img.shields.io/badge/stars-30+-blue.svg"/>
	<img src="https://img.shields.io/badge/license-CC0-blue.svg"/>
</p>

- Copyright © Official quantitative and statistical software published by WYN Associates.
- Copyright © 2010 – Present Yiqiao Yin
- Contact: Yiqiao Yin
- Official Site: https://www.WYN-Associates.com
- Email: Yiqiao.Yin@wyn-associates.com

## Installation and Development
	
Welcome to install our package. The entire package is written in *python*. One can use the following code to install this Github package.

```
# from command line
pip install git+https://github.com/yiqiao-yin/WYNAssociates.git # in command line
```

```
# from jupyter
!pip install git+https://github.com/yiqiao-yin/WYNAssociates.git # in command line
```

For developers, one can clone package by simple **git clone** command (assuming in a desired directory).

```
git clone https://github.com/yiqiao-yin/WYNAssociates.git
```

## Documentation

- A sample notebook for RNN education can be found [here](https://github.com/yiqiao-yin/WYNAssociates/blob/main/docs/python_MM_LSTM_StockPriceForecast.ipynb). We provide some basic education of Recurrent Neural Network (aka Long Short-Term Memory). The term Recurrent Neural Network is short for RNN and Long Short-Term Memory is short for LSTM.

## Data-Centric Solutions

### AI-Driven Pipelines

There are three global sections of AI-driven solutions in our enterprise. They are listed below.

- [AI_solution](https://github.com/yiqiao-yin/WYNAssociates/tree/main/AI_solution) - The module contains deployable functions of a variety of different solutions in representation learning.
- [FIN_solution](https://github.com/yiqiao-yin/WYNAssociates/tree/main/FIN_solution) - The module contains deployable functions in financial technology.
- [ML_solution](https://github.com/yiqiao-yin/WYNAssociates/tree/main/ML_solution) - The module contains machine learning or ML-driven tools.

| [AI Solution](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/AI_solution/modules.py#L17) | Definition |
| ---         | ---        |
| [NN3](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/AI_solution/modules.py#L29) | LeNet3-style neural network |
| [NeuralNet Regressor](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/AI_solution/modules.py#L1033) | Generalized code for neural network based regressor | 
| [NeuralNet Classifier](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/AI_solution/modules.py#L1299) | Generalized code for neural network based classifier | 
| [LSTM Regressor](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/AI_solution/modules.py#L1463) | A generalized code for developing sequential models |
| [U-Net Model](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/AI_solution/modules.py#L1565) | Inception-style image segmentation model |

| [Financial Solution](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/FIN_solution/modules.py#L25) | Definition |
| ---         | ---          |
| [Yin Timer](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/FIN_solution/modules.py#L33) | A timing algorithm proprietary at W.Y.N. Associates |
| [RSI Timer](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/FIN_solution/modules.py#L302) | A timing algorithm based on technical indicator RSI | 
| [RNN Regressor](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/FIN_solution/modules.py#L416) | A recurrent neural network for stock specifically |
| [Neural Sequence Translation](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/FIN_solution/modules.py#L702) | A sequence-to-sequence model for stock data | 
| [Automated Neural Sequence Translation](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/FIN_solution/modules.py#L966) | A sequence-to-sequence model for tabular data |

| [Machine Learning Solution](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/ML_solution/modules.py#L25) | Definition |
| ---         | ---          |
| [Logistic Classifier](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/ML_solution/modules.py#L33) | Logistic-based classifier |
| [K-Nearest Neighbors Classifier](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/ML_solution/modules.py#L90) | K Nearest Neighborhood based classifier | 
| [Decision Tree Classifier](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/ML_solution/modules.py#L147) | Decision Tree classifier |
| [Decision Tree Regressor](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/ML_solution/modules.py#L213) | Decision Tree regressor | 
| [Random Forest Classifier](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/ML_solution/modules.py#L266) | Random Forest classifier |
| [Random Forest Regressor](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/ML_solution/modules.py#L323) | Random Forest regressor | 
| [Gradient Boosting Classifier](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/ML_solution/modules.py#L403) | Gradient Boosting classifier |
| [Gradient Boosting Regressor](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/ML_solution/modules.py#L465) | Gradient Boosting regressor | 
| [Support Vector Machine Regressor](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/ML_solution/modules.py#L517) | Support Vector Machine regressor | 
| [Adam Regressor](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/ML_solution/modules.py#L606) | A general code for fitting tabgular data using Adam optimization |
| [Area-Under-Curve and Receiver-Operating-Characteristics](https://github.com/yiqiao-yin/WYNAssociates/blob/d8b1244df5b206c32577e5fabb41afbaaa5b4876/ML_solution/modules.py#L735) | An efficient function of calculating AUC and plotting ROC |
