import pandas as pd
import nltk
import re
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from scipy.stats import ttest_ind

# For cosine sim
model = SentenceTransformer("all-mpnet-base-v2")


def import_tsv(file_path):
    return pd.read_csv(file_path, sep='\t')


# In results get each identified difficult term and clean them up
def extract_terms(output):
    terms = re.findall(r"(?<!\d)\b.*?(?=\n|$)", output)
    terms = [term.strip() for term in terms if term.strip() and not term.strip().isdigit()]

    terms = [re.sub(r'\d+\.\s*', '', term).strip() for term in terms]


    return terms


def calculate_cosine_similarity(generated_definition, ground_truth_definition):
    
    embeddings1 = model.encode(generated_definition, convert_to_tensor=True)
    embeddings2 = model.encode(ground_truth_definition, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    return cosine_scores[0][0]


def calculate_bleu_score(candidate, reference):
    candidate_tokens = nltk.word_tokenize(candidate)
    reference_tokens = nltk.word_tokenize(reference)

    smoothing = SmoothingFunction().method1  # Additive (Laplace) smoothing
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)


def calculate_rouge_score(candidate, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(candidate, reference)
    
    return {
        'ROUGE-1': scores['rouge1'].fmeasure,
        'ROUGE-2': scores['rouge2'].fmeasure,
        'ROUGE-L': scores['rougeL'].fmeasure
    }


# In results get each definition for the term
def parse_output(output):
    pattern = r'\d+\.\s+(.*?)(?=\d+\.\s+|\Z)'

    definitions = re.findall(pattern, output, re.DOTALL)
    
    return definitions


def main():
    tsv_res_files = ["Llama2_results.tsv", "Llama3_results.tsv"]
    
    evaluation_scores = []
    baseline_evaluation_scores = []
    proposed_system_evaluation_scores = []

    # Evaluate results for baseline and proposed system
    for tsv in tsv_res_files:
        baseline_results_data = import_tsv(tsv)
        passage_data = import_tsv("task2_test.tsv")
        ground_truths_data = import_tsv("ground_truths.tsv")
        
        # Information about each passage for testing
        terms = ground_truths_data["term"].tolist()
        snt_ids = passage_data["snt_id"].tolist()
        passages = passage_data["source_snt"].tolist()

        ground_truth_definitions = ground_truths_data["definition"].tolist()
        terms_to_ground_truth = {}
        
        snt_id_to_passage = {}
        for i, snt_id in enumerate(snt_ids):
            snt_id_to_passage[snt_id] = passages[i]

        ground_truth_paper_to_terms = {}
        for _, row in ground_truths_data.iterrows():
            snt_id = row['snt_id']
            term = row['term']
            if snt_id in ground_truth_paper_to_terms:
                ground_truth_paper_to_terms[snt_id].append(term)
            else:
                ground_truth_paper_to_terms[snt_id] = [term]
        
        
        for i, term in enumerate(terms):
            terms_to_ground_truth[term.lower()] = ground_truth_definitions[i].lower()

        # Get difficult term predictions and definitions from each paper
        paper_to_definitions = {}
        paper_to_terms = {}
        for index, row in baseline_results_data.iterrows():
            predicted_definitions = parse_output(row["Explanations"])
            predicted_terms = extract_terms(row["Identified Difficult Terms"])
            paper_to_definitions[row["snt_id"]] = predicted_definitions
            paper_to_terms[row["snt_id"]] = predicted_terms

        snt_id_list = []  
        passage_list = [] 
        predicted_terms_list = []  
        actual_terms_list = []   
        predicted_definitions_list = []   
        actual_definitions_list = []   
        cosine_similarity_list = [] 
        bleu_score_list = []
        rogue_score_list = []
        
        for paper, definitions in paper_to_definitions.items():
            # Load all data for a pd data frame for tsv
            ground_truth_definitions = [] # Gather all ground truths 4 batch of terms
            cos_sims = [] # To gather in batches of 5 for displaying
            bleu_scores = []
            rogue_scores = []
            terms = paper_to_terms[paper]
            predicted_terms_list.append(terms)
            actual_terms_list.append(ground_truth_paper_to_terms.get(paper, []))
            snt_id_list.append(paper)
            passage_list.append(snt_id_to_passage[paper])
            
            # Compare definition to ground truth
            for i, definition in enumerate(definitions):
                if i < len(terms): # If term correctly identified for a paper
                    if terms[i].lower() in terms_to_ground_truth and terms[i].lower() in ground_truth_paper_to_terms.get(paper, []):
                        ground_truth_definitions.append(terms_to_ground_truth[terms[i].lower()])
                        cos_similarity = calculate_cosine_similarity(definition, terms_to_ground_truth[terms[i].lower()])
                        bleu_score = calculate_bleu_score(definition, terms_to_ground_truth[terms[i].lower()])
                        rogue_score = calculate_rouge_score(definition, terms_to_ground_truth[terms[i].lower()])
                        cos_sim = calculate_cosine_similarity(definition, terms_to_ground_truth[terms[i].lower()])
                        cos_sims.append(cos_similarity)
                        bleu_scores.append(bleu_score)
                        rogue_scores.append(rogue_score)
            
            # Evaluation results for each passage
            cosine_similarity_list.append(cos_sims)
            rogue_score_list.append(rogue_scores)
            bleu_score_list.append(bleu_scores)
            predicted_definitions_list.append(definitions)
            actual_definitions_list.append(ground_truth_definitions)
        
        evaluation_results_list = []
        cos_sims_numerical = []
        for i in range(len(cosine_similarity_list)):
            evaluation_results_string = f'BLEU SCORES: {bleu_score_list[i]} \nROGUE SCORES: {rogue_score_list[i]} \nCOS_SIM SCORES: {cosine_similarity_list[i]}'
            evaluation_results_list.append(evaluation_results_string)
            # For significance testing
            cos_sims_numerical.append(cosine_similarity_list[i])
            
        # Panda data frame for evaluated results
        data = {
            'passage': passage_list,
            'predicted_complex_terms': predicted_terms_list,
            'actual_complex_terms': actual_terms_list,
            'predicted_definitions': predicted_definitions_list,
            'actual_definitions': actual_definitions_list,
            'evaluation_results': evaluation_results_list
        }   
        df = pd.DataFrame(data)
        tsv_name = f'{tsv}_evaluated.tsv'
        # Store scores across both baseline and proposed system
        evaluation_scores.append(cos_sims_numerical)
        df.to_csv(tsv_name, sep='\t', index=False)
        
        # Convert DataFrame to LaTeX table
        latex_table = df.to_latex(index=False, escape=False)

        with open('evaluated_results.tex', 'w') as f:
            f.write(latex_table)
    
    # Cos sims
    baseline_evaluation_scores = evaluation_scores[0]
    proposed_system_evaluation_scores = evaluation_scores[1]

    # Flatten array of arrays
    flattened_scores_baseline = [score for passage_scores in baseline_evaluation_scores for score in passage_scores]
    flattened_scores_proposed_system = [score for passage_scores in proposed_system_evaluation_scores for score in passage_scores]


    # Significance Testing
    t_statistic, p_value = ttest_ind(flattened_scores_baseline, flattened_scores_proposed_system)
    test_results = {
        "t_statistic": t_statistic,
        "p_value": p_value,
        "significant": p_value < 0.05
        }
    
    print("Cosine Similarity:")
    print(f"T-statistic: {test_results['t_statistic']}")
    print(f"P-value: {test_results['p_value']}")
    print(f"Significant: {test_results['significant']}")

main()
