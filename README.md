Detecting Phishing Websites Using URL-Based, HTML-Based, and Domain-Based Features
 Project Overview

This project focuses on building a machine learning–based system to detect phishing websites using a structured dataset containing URL features, HTML/JavaScript properties, and domain information. The goal is to accurately classify websites as legitimate or phishing based on patterns extracted from the data.

So far, the project includes:

Dataset loading and preprocessing

Exploratory Data Analysis (EDA)

Identification and handling of missing values

Feature encoding and scaling

Correlation analysis and heatmap visualization

Feature selection

Preparation for supervised modelling

 Dataset Description

The dataset contains features grouped into three categories:

1. URL-Based Features

Examples:

url_length

num_dots

num_hyphens

has_ip_address

uses_https

2. HTML and JavaScript Features

Examples:

num_iframes

num_scripts

external_resources

popup_count

3. Domain-Based Features

Examples:

domain_age

dns_record

domain_registration_length

Target Variable

status:

legitimate (0)

phishing (1)

 Data Cleaning & Preprocessing
Completed steps:

Dropped irrelevant columns such as row identifiers.

Fixed data types (converted string numerics to integers).

Checked for missing values and confirmed dataset integrity.

Encoded target variable:
