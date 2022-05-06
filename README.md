# CryptoOfTheFuture

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

### Requires Git, Python 3.7–3.10, and pip >= 19.0

1. Clone this repo: `$ https://github.com/arielnlee/Crypto-of-the-Future-Reinforcement-Learning.git`
2. [Install Jupyter](https://jupyter.org/install)
3. Install Dependencies (P1): For LSTM structure: `$ pip install -U matplotlib scikit-learn tensorflow`
4. Install Dependencies (P2): For trading environment: pip install tensorflow==1.15.0 stable-baselines gym-anytrading gym empyrical
5. Start the Project Jupyter notebook: `$ jupyter notebook CryptoOfTheFuture/Project.ipynb`

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

References used as models for codes are:

[9]: Used as a model for the baselines.

[10]: Helped significantly with Tensorflow

## Contibutors

Ariel Lee, Gaurav Koley, Rahul Razdan
