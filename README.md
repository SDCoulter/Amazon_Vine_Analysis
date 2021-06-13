# Amazon Vine Review Analysis

## Challenge - Overview

In this module we look at Big Data and Natural Language Processing, with a slight dabble in Machine Learning. The context around the data analysis is that we are looking to understand how to understand product (or any type) reviews can be understood by a machine, that primarily deals in numerical values. In the module we also interact with Amazon Web Services, S3 Buckets, Spark and the Python version PySpark, SQL and pgAdmin, Google Colab, and have a brief look at Hadoop and MapReduce and their function in data science.

For NLP we took data, Tokenized it, Stop Word Removed it, Hashed it, and fed it into an IDF method (Term Frequency-Inverse Document Frequency Weight). All before fitting/training it to a Machine Learning model and running test data through it. We understood these concepts individually, and then worked to add each step to a **Data Pipeline** in PySpark.

For the challenge we needed to take one of fifty different review datasets to run analysis on. We were given a list of Amazon review datasets, containing both paid and unpaid reviews, split into different categories - dataset list found [here](https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt). We needed to pick a dataset, I chose the [Apparel](https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Apparel_v1_00.tsv.gz) one, and we would begin our analysis.

Once we had our chosen dataset, we would first use PySpark to perform the Extract-Transform-Load process on it. Then we would connect to an AWS RDS instance, and load the transformed data into pgAdmin. This would be done in Google Colab by installing PySpark and the driver that works with PostgreSQL. Once we had created our needed tables for analysis, we exported the `vine_table` table to a CSV file, to breakdown. In Jupyter Notebook we used `Pandas` to split the data up further, and finally, determine if there was any bias toward the rating of the review, based on whether it was a paid (`vine == 'Y'`) review or not.

## Challenge - Results

<!--
Results: Using bulleted lists and images of DataFrames as support, address the following questions:

How many Vine reviews and non-Vine reviews were there?
How many Vine reviews were 5 stars? How many non-Vine reviews were 5 stars?
What percentage of Vine reviews were 5 stars? What percentage of non-Vine reviews were 5 stars?
Results:

There is a bulleted list that addresses the three questions for unpaid and paid program reviews (7 pt)

-->

## Challenge - Summary

<!--
Summary: In your summary, state if there is any positivity bias for reviews in the Vine program. Use the results of your analysis to support your statement. Then, provide one additional analysis that you could do with the dataset to support your statement.
Summary:

The summary states whether or not there is bias, and the results support this statement (2 pt)
An additional analysis is recommended to support the statement (2 pt)
-->

## Context

This is the Challenge Repo for Module 16 of the University of Toronto School of Continuing Studies Data Analysis Bootcamp Course - **Big Data and NLP** - PySpark, Machine Learning, NLP, and Data Interactions. Following the guidance of the module we end up pushing this selection of files to GitHub.
