# COS-470-Project-Part-III-
SimpleText Task 2 Proposed System by prompting Meta's Llama 2. 

## How to Run and Reproduce Results
To reproduce the Proposed system evaluation results: 

-- Note that you may also just run evaluate_res.py for I have provided the necessary files. To produce Llama2_results.tsv and Llama3_results.tsv run the collab files P2_task2 and P3_task2 respectively -- 

- First run the Google Colab file, making sure to properly authenticate the usage of Llama 2 by running the top code block and inputting a Hugging Face API key, say no to everything that follows. 
- Insert the "task2_test.tsv" file into the Colab file tab so it can read the queries off of it. This code may take a while to run, so in this repo there is a provided "results.tsv" file that contains the results that will be gotten from running that Colab file. If you wish to run the Colab file, expect around a 7-minute wait to run the 6 queries. 
- After Colab finishes running, it will produce a "Llama3_results.tsv" file, that will be used for the next part of getting the evaluation results. 
- In the directory, keep the Llama2_results.tsv file, which is from the last part, this is done to do significance testing between the results of the two. 

- Finally, after the "Llama3_results.tsv" file is gotten by either using the one already provided in the repo or producing it through Colab, select your favorite editor and insert this file and "ground_truths.tsv", "evaluate_res.py", and "Llama2_results.tsv". 
- This is a standard Python script that parses the results from running Llama 2 in both part 2 and part 3 and makes a TSV of the evaluation results for the proposed system and the baseline system and then compares the two for significance testing. 

- After those two steps, you will see the results from the baseline and the proposed system and the significance testing results will be displayed in the console.

- Another thing to note, Llama 2 results from running collab a reproducible, and Llama 3 results from running collab aren't. I believe this is a quirk of the Llama 3 model, so I reccomend you use the Llama3_results.tsv I provided in the submission.

