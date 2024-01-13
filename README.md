# AI Trading Chatbot

This project implements an AI chatbot to provide personalized trading recommendations using Bayesian networks. The chatbot leads users through the process of constructing a Bayesian network tailored to their specific trading scenario, and utilizes the network to offer data-driven insights.

## Overview
- Implements conversational AI using GPT-4 via LangChain and Streamlit
- Allows users to describe their unique trading scenario
- Dynamically generates relevant nodes and edges for a Bayesian network
- Incorporates real-time financial data via FRED and YFinance APIs
- Estimates conditional probability distributions using maximum likelihood
- Performs inferences on the Bayesian network
- Provides trading recommendations based on probabilistic analysis

# Features
- Interactive Node Creation: Users can customize the nodes in their Bayesian network, adding, removing or modifying them through conversation.
- Intelligent Edge Recommendations: The chatbot suggests potential edges based on the defined nodes, maintaining network coherency.
- Real-Time Data Integration: Financial time series data is integrated from FRED and YFinance to inform the probability distributions.
- Conditional Probability Estimation: CPDs are estimated using maximum likelihood estimation given the edges and real-time data.
- Trading Recommendations: The chatbot analyzes the Bayesian network to recommend optimal trading decisions based on probabilistic inferences.
- Natural Conversation: The entire interaction from node creation to final recommendations happens through natural dialogue powered by GPT-4.


## Installation Requirements:

Python 3.7+
Streamlit
LangChain
pgmpy
pandas
yfinance
fredapi
bash

Copy code

pip install streamlit langchain pgmpy pandas yfinance fredapi

## Usage
Copy code

streamlit run risk_bot/🤖_Bot.py

The app will be served at http://localhost:8501. Follow the conversational prompts to construct your Bayesian network and receive trading recommendations tailored to your scenario.

## Demo
A video demo of the app can be found here: https://github.com/pareshraut/Automated-Bayesian-Networks/issues/2#issue-2079936495)https://github.com/pareshraut/Automated-Bayesian-Networks/issues/2#issue-2079936495

References
The core methodology was adapted from:

LangChain library
pgmpy library
