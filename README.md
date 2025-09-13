# CORD19_Metadata_Analysis
Analysis of the CORD-19 COVID-19 metadata dataset: data cleaning, exploratory analysis, visualizations, and an optional Streamlit dashboard.
"""
CORD-19 Metadata Analysis
Author: Millicent Nabututu Makokha

📌 Project Description
This project analyzes the COVID-19 Open Research Dataset (CORD-19) metadata.  
It demonstrates data cleaning, exploratory analysis, visualization, report generation, and building an interactive dashboard.  
The focus is on publication trends, top journals, frequent keywords, and multidisciplinary study areas.

🎯 Objectives
- Load and clean CORD-19 metadata
- Analyze publication trends over time
- Identify top journals and frequent keywords
- Visualize findings with charts
- Generate an auto-report
- Provide an optional interactive dashboard

🛠 Tools & Libraries
- Python 3
- pandas, matplotlib, seaborn
- requests (for dataset download)
- wordcloud (optional)
- streamlit (optional, for dashboard)

▶ How to Run
1. Install dependencies:
   pip install pandas matplotlib seaborn requests wordcloud streamlit

2. Run analysis (saves charts + report in outputs/ folder):
   python CORD19_assignment.py

3. (Optional) Run interactive dashboard:
   streamlit run CORD19_assignment.py

📊 Key Insights
- Surge in publications after 2020 due to COVID-19.
- Top journals like bioRxiv and medRxiv published many papers.
- Frequent keywords include "COVID-19", "SARS-CoV-2", "pandemic".
- Research covers medicine, biology, public health, and social sciences.

💡 Reflections
- Gained hands-on experience in data wrangling and visualization.
- Learned to optimize code for low-resource devices.
- Understood the importance of interactive dashboards.
- Saw the power of data-driven insights in global health crisess
