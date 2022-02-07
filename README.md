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
pip install git+https://github.com/yiqiao-yin/WYNAssociates.git # in command line
!pip install git+https://github.com/yiqiao-yin/WYNAssociates.git # in jupyter notebook
```

For developers, one can clone package by simple **git clone** command (assuming in a desired directory).

```
git clone https://github.com/yiqiao-yin/WYNAssociates.git
```

## Documentation

- A sample notebook for RNN education can be found [here](https://github.com/yiqiao-yin/WYNAssociates/blob/main/docs/python_MM_LSTM_StockPriceForecast.ipynb). We provide some basic education of Recurrent Neural Network (aka Long Short-Term Memory). The term Recurrent Neural Network is short for RNN and Long Short-Term Memory is short for LSTM.

## List of Functions

| Name  | Definition |
| ------------- | ------------- |
| [Yin_Timer](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L27)  | A stock market timing strategy  |
| [RSI_Timer](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L296)  | A stock market timing strategy  |
| [RNN_Regressor](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L410) | A stock price forecast algorithm |
| [Neural_Sequence_Translation](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L696) | A candlestick sequence forecast algorithm |
| [Autonomous_Neural_Sequence_Translation](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L960) | A sequence-to-sequence prediction model |
| [Embedding_Neural_Sequence_Translation](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L1223) | A sequence-to-sequence prediction model with embedding layer | 
| [YinsML](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L1509) | A class of functions in machine learning | 
| - [LogisticRegression_Classifier](https://github.com/yiqiao-yin/WYNAssociates/blob/42022078dc8f00fdf250f19d80f34a1cd685cda6/AI_solution/modules.py#L1517) | Logistic regression classifier |
| - [DecisionTree_Classifier](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L1517) | Decision tree classifier |
| - [DecisionTree_Regressor](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L1574) | Decision tree regressor |
| - [RandomForest_Classifier](https://github.com/yiqiao-yin/WYNAssociates/blob/d3f5ef496403a0aef6b940c1740734e174043e71/AI_solution/modules.py#L1616) | Random Forest Classifier algorithm | 
| - [GradientBoosting_Classifier](https://github.com/yiqiao-yin/WYNAssociates/blob/d3f5ef496403a0aef6b940c1740734e174043e71/AI_solution/modules.py#L1673) | Gradient Boosting Classifier algorithm |
| - [Adam_Regressor](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L1616) | Regressor trained using Adam | 
| - [ResultAUCROC](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L1745) | Compute AUCROC of a predictor to its ground truth |
| [YinsDL](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L1769) | A class of functions in deep learning | 
| - [NN3_Classifier](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L1781) | A neural network with 3 layers |
| - [NN10 Classifier](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L1917) | A neural network with 10 layers | 
| - [plotOneImage](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L2066) | A helper function to plot images | 
| - [ConvOperationC1](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L2101) | Hand crafted convolutional operation with one pre-defined filter | 
| - [C1NN3_Classifier](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L2193) | A CNN with one convolutional layer and 3 neural network layers | 
| - [C2NN3_Classifier](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L2366) | A CNN with two convolutional layer and 3 neural network layers | 
| - [RNN4_Regressor](https://github.com/yiqiao-yin/WYNAssociates/blob/2b5994f77a74038dd10e55182a0cc16e71168a32/AI_solution/modules.py#L2606) | A RNN regressor model for stock price prediction |
