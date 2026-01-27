# Intelligent systems for bioinformatics 25/26 

Work done by: 
- [Alexandre Ferreira](https://github.com/Alexsf35) (pg55691)
- [João Faria](https://github.com/JohnnyFarians24) (pg55700)
- [Sami Kawashita](https://github.com/samildo) (pg55704)
- [Vítor Silva](https://github.com/VitorSilva-3) (pg55538) 

This repository contains the source code and analysis for the practical project of the **Intelligent Systems for Bioinformatics** course. The project focuses on modeling the antigenic evolution of the **Influenza A (H3N2)** virus using genomic surveillance data from Japan and machine learning techniques.

##  Project overview

The main objective is to predict antigenic escape and understand the evolutionary dynamics of the H3N2 Hemagglutinin (HA) protein. The project integrates **epidemiological data** (case counts) with **genomic data** (HA sequences) to build predictive models.

### Key features
* **Data engineering:** Processing of raw FASTA sequences, Multiple Sequence Alignment (MSA), and epitope mapping.
* **Feature extraction:** Calculation of **pEpitope** (antigenic distance), N-glycosylation sites, and physicochemical property changes.
* **Machine learning modeling:**
    * **Linear regression:** Baseline model for trend analysis.
    * **Random forest regressor:** Non-linear model to capture complex evolutionary patterns (e.g., glycosylation gain/loss).
* **Validation:** Time-series split validation to simulate real-world forecasting scenarios.

## Repository structure

```text
├── data_prep/              # Data preprocessing scripts
│   ├── epidmiological_data/ # CSV files with Japan surveillance data
│   ├── omic_data/           # FASTA sequences and R Markdown scripts
│   └── Projeto_sib_r.Rmd    # R script for sequence alignment and cleaning
│
├── epi_pipeline/           # Python modules for the analysis pipeline
│   ├── agents.py            # Agent definitions (Virus/Host abstractions)
│   ├── epidemic_pipeline.py # Main execution logic
│   ├── features.py          # Feature engineering (mutations, glycosylation)
│   └── pepitope.py          # pEpitope calculation logic
│
├── ML/                     # Machine Learning analysis
│   └── notebook.ipynb       # Main Jupyter Notebook (Training & Evaluation)
│
├── images/                 # Plots and figures used in reports
└── README.md               # Project documentation
