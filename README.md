# CryptoOfTheFuture

![Pipeline](fig.png)

This project will implement a deep learning reinforcement algorithm — proximal policy optimization — to devise an automatically generating strategy for Ethereum transactions. Long short-term memory is used to make predictions for next day closing prices, which will in turn be used to construct the automatic policy. The completed stages of the project are as follows:

1. Extracted hourly Ethereum price data from Binance API beginning on August 21, 2017 through February 17, 2021
2. Preprocessed data using MinMaxScaler
3. Defined the LSTM model using TensorFlow. Generate train, validation, and test sets for the model.
4. Trained the model.
5. Visualized the actual and predicted prices in plt graph
6. Visualized the MSE loss of the training and testing sets in plt graph
7. Created a new custom trading environment using gym
8. Used Proximal Policy Optimization to train an agent to effectively trade Ethereum. 
9. Visualized a summary of all executed transactions (buy, sell, hold)


## Dependencies

* **json + requests**: To fetch data from Binance API
* **sklearn**: For data preprocessing and train-test split
* **TensorFlow**: Machine Learning Library. For the LSTM tensorflow 2 was used however for the trading environment tensorflow 1 was required due to stable baselines only having functionality in tensorflow 1.
* **Matplotlib**: For plotting loss and prediction graphs
* **empyrical**: For calculating the omega ratio
* **gym**: For creating the custom trading environment
* **gym-anytrading**: For the baseline agents
* **stable_baselines** For the PPO policies and all other policies (including A2C)


## Description of Files
1. BinanceDataRequest.py: Extracts data from Binance API
2. LSTM_Deprecated.py: Previous version of the LSTM which uses PyTorch instead of Tensorflow
3. LSTM.py: Current version of the LSTM network using Tensorflow
4. ETH_predictions.csv: The predictions that we got from our runthrough of the LSTM network.
5. PPObaseline.py: The baseline models that we compare our custom model with.
6. TradingETH.py:  The custom environment that our agent trades in.

## How to Run (Two approaches)
Common Steps
1. Run this command: '$ git clone https://github.com/arielnlee/Crypto-of-the-Future-Reinforcement-Learning.git
2. Run this command: cd Crypto-of-the-Future-Reinforcement-Learning
3. [Install Python](https://www.python.org/downloads/)
4. Install all dependencies

Approach 1: Using our CSV

4. Run this command: python TradingETH.py

Approach 2: Creating CSV Using custom made LSTM

4. Uncomment Lines 111 and 112 in LSTM.py
5. Run python LSTM.py
6. Run python TradingETH.py

## References
[1] Patrick Jaquart, David Dann, Christof Weinhardt. 2021. “Short-term bitcoin market prediction via machine learning”, Journal of Finance and Data Science. https://doi.org/10.1016/j.jfds.2021.03.001.

[2] Mingxi Liu, Guowen Li, Jianping Li, Xiaoqian Zhu, Yinhong Yao. 2021. “Forecasting the price of Bitcoin using deep learning”, Finance Research Letters. https://doi.org/10.1016/j.frl.2020.101755.

[3] Thanasis Zoumpekas, Elias Houstis, Manolis Vavalis. 2020. “ETH analysis and predictions utilizing deep learning”, Expert Systems with Applications. https://doi.org/10.1016/j.eswa.2020.113866.

[4] McNally, S., Roche, J., Caton, S., 2018. “Predicting the price of bitcoin using machine learning”. In: The 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing. https://doi.org/10.1109/PDP2018.2018.00060.

[5] F Liu, Y Li, B Li, J Li, H Xie. 2021. “Bitcoin transaction strategy construction based on deep reinforcement learning”. Applied Soft Computing. https://doi.org/10.1016/j.asoc.2021.107952.

[6] Zhuoran Xiong, Xiao-Yang Liu, Shan Zhong, Hongyang Yang, Anwar Walid. 2018. “Practical Deep Reinforcement Learning Approach for Stock Trading”. ArXiv. https://doi.org/10.48550/arXiv.1811.07522.

[7] Ouyang, Z.; Ravier, P.; Jabloun, M. STL Decomposition of Time Series Can Benefit Forecasting Done by Statistical Methods but Not by Machine Learning Ones. Eng. Proc. 2021, 5, 42. https://doi.org/10.3390/ engproc2021005042

[8] Passalis N., Seficha S., Tsantekidis A., Tefas A. 2021. Learning Sentiment-Aware Trading Strategies for Bitcoin Leveraging Deep Learning-Based Financial News Analysis. In: Maglogiannis I., Macintyre J., Iliadis L. (eds) Artificial Intelligence Applications and Innovations. AIAI 2021. IFIP Advances in Information and Communication Technology, vol 627. Springer, Cham. https://doi.org/10.1007/978-3-030-79150-6_59

[9] Dev V. 2021. “Creating a Market Trading Bot Using Open AI Gym Anytrading”. Analytics India Magazine. https://analyticsindiamag.com/creating-a-market-trading-bot-using-open-ai-gym-anytrading/ 

[10] Palanisamy, P. 2021. TensorFlow 2 Reinforcement Learning Cookbook: Over 50 recipes to help you build, train, and deploy learning agents for real-world applications. Packt Publishing.

[11] Eric Benhamou, Beatrice Guez, and Nicolas Paris1. 2019. Omega and Sharpe ratio. arXiv:1911.10254 [q-fin], October. arXiv: 1911.10254.

[12] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal Policy Optimization Algorithms. arXiv:1707.06347 [cs], August. arXiv: 1707.06347.

[13] CryptoDataDownload. "How to Pull Binance Data for Any Specific Time Period Using Python." Cryptodatadownload.com. 16 Mar. 2022. Web. 6 May 2022.

[14] Biondo, A.E., Pluchino, A., Rapisarda, A. and Helbing, D., 2013. Are random trading strategies more successful than technical ones?. PloS one, 8(7), p.e68344.

[15] Brock, W., Lakonishok, J. and LeBaron, B., 1992. Simple technical trading rules and the stochastic properties of stock returns. The Journal of finance, 47(5), pp.1731-1764.



References used as models for codes are:

[9]: Used as a model for the baselines.

[10]: Helped significantly with Tensorflow

[13]: Helped with extracting the data.

## Contibutors

Ariel Lee, Gaurav Koley, Rahul Razdan
