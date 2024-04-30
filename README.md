# COS-470-Project-Part-II-
SimpleText Task 2 baseline by prompting Meta's Llama 2. 

## How to Run and Reproduce Results
To reproduce the baseline evaluation results, there are two main parts: 

First run the Google Colab file, making sure to properly authenticate the usage of Llama 2 by running the top code block and inputting a Hugging Face API key, say no to everything that follows. Insert the "task2_test.tsv" file into the Colab file tab so it can read the queries off of it. This code may take a while to run, so in this repo there is a provided "results.tsv" file that contains the results that will be gotten from running that Colab file. If you wish to run the Colab file, expect around a 7-minute wait to run the 6 queries. After Colab finishes running, it will produce a "results.tsv" file, that will be used for the next part of getting the evaluation results. 

Finally, after the "results.tsv" file is gotten by either using the one already provided in the repo or producing it through Colab, select your favorite editor and insert this file and "ground_truths.tsv" along with "evaluate_res.py". This is a standard Python script that parses the results from running Llama 2 and makes a TSV of the evaluation results. This TSV file is the basis for the evaluation results PDF, which is essentially a more readable version of it. 

After those two steps, you will see the results from the baseline. 

