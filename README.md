# What is this?
Project for scraping Danish tweets related to economics and compute a 'Twitter Sentiment Indicator' (TSI), which quantifies the degree to which Danish economics tweets express positive or negative sentiment.
It also contains methods to assess the impact of various topics on the TSI, as well as tools to extract the Consumer Trust Indicator (CTI) from Statistics Denmark.

# Setup TODO FINSH GUIDE
- clone from gitlab and navigate to project folder.
- Setup virtual environment: run setup.bat


# How to run
- Navigate to project folder and start environment with venv\Scripts\activate
- To generate the plots from the corona tracker, run 'python generate_tracker_plots.py'
- The utils module contains helper methods to download twitter data to store locally and compute sentiments. Running update_all() will update local tweets and compute sentiments. If no data is stored locally, this will take a lot of time (10+ hours) and may crash if and when connection issues arise. Simply running the method again should make it pick up where it left off.