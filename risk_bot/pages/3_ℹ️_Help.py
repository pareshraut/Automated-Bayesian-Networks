import streamlit as st


with open("styles.css") as f:                                                
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html= True) 

# About Us Page Content
st.title("AI Chatbot for Investment Decisions")

# Introduction
st.header("About Bespoke Trade AI Chatbot")
st.markdown("Welcome to Bespoke Trade AI Chatbot, your intelligent companion for making informed investment decisions. "
            "Our chatbot leverages advanced AI technology to provide tailored strategies based on your unique investment scenarios.")

# Understanding Bayesian Networks
st.header("Understanding Bayesian Networks")
st.markdown("A Bayesian Network is a probabilistic graphical model that represents a set of variables and their "
            "conditional dependencies via a directed acyclic graph. In the context of investment, these networks "
            "can analyze potential outcomes and offer predictive insights, making them invaluable for decision-making.")

# Link to Bayesian Network Resources
st.markdown("[Learn More About Bayesian Networks](https://en.wikipedia.org/wiki/Bayesian_network)")

# Our Methodology
st.header("Our Methodology")
st.markdown("Our chatbot follows a structured methodology to assist you in your investment decisions.")

# Investment Scenario Analysis
st.subheader("1. Investment Scenario Analysis")
st.markdown("Start by presenting your investment scenario. Our chatbot utilizes GPT-4 to generate an initial set of "
            "decision nodes, tailored to your specific needs.")

# Interactive Node Customization
st.subheader("2. Interactive Node Customization")
st.markdown("You have the control to add, remove, or modify these nodes, ensuring a highly personalized investment strategy.")

# Intelligent Edge Recommendations
st.subheader("3. Intelligent Edge Recommendations")
st.markdown("Based on a set of predefined rules and GPT-4's advanced capabilities, our system recommends potential edges. "
            "Customize them as you see fit to refine your strategy.")

# Data-Driven Insights
st.subheader("4. Data-Driven Insights")
st.markdown("Utilizing FRED API and YFinance API, we integrate real-time financial data to inform your decisions.")
st.markdown("[FRED API](https://fred.stlouisfed.org/) | [YFinance API](https://pypi.org/project/yfinance/)")

# CPD Generation with Maximum Likelihood Estimator
st.subheader("5. CPD Generation with Maximum Likelihood Estimator")
st.markdown("Employing the Maximum Likelihood Estimator, we generate accurate Conditional Probability Distributions for comprehensive analysis.")
st.markdown("[Maximum Likelihood Estimation - Overview](insert link)")

# User-Centric CPD Contribution
st.subheader("6. User-Centric CPD Contribution")
st.markdown("Your insights matter. Contribute to the remaining CPDs, adding a personal touch to the investment analysis.")

# Finalizing Your Investment Decision
st.subheader("7. Finalizing Your Investment Decision")
st.markdown("All these elements come together in a Bayesian Network, guiding you to make informed investment decisions.")

# Why Choose Bespoke Trade AI Chatbot?
st.header("Why Choose Bespoke Trade AI Chatbot?")
st.markdown("Experience a unique blend of AI intelligence and personal customization. Read our success stories and "
            "get in touch to start your bespoke investment journey today.")

# Success Stories
st.markdown("[Read Our Success Stories](insert link)")

# Contact Information
st.header("Contact Information")
st.markdown("Contact us at [contact@email.com](mailto:contact@email.com)")

# Explore Chatbot CTA
st.button("Explore our chatbot and start your bespoke investment journey today!")

# Legal Information
st.sidebar.header("Legal Information")
st.sidebar.markdown("[Terms of Use](insert link) | [Privacy Policy](insert link)")
