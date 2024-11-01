# Legal Documents analyzer

* A Large Language Model(LLM) for Summarizing case law, contracts, and regulations, and identifying similar cases or clauses.
* This case analyzer is built using gemini pro 1.5 for summarizing.
* SBERT is used for finding similarity score between existing cases and user's input.
* This demo is shown for California Case Laws dataset: https://www.kaggle.com/datasets/itshappy/california-case-laws
* Important features like case title, time_stamp and document is given as input for generating summary.
* The output is shown in table format with title, summary of cases and similarity_score (for user input and existing cases).
* The overall optput shows how much the user's case is matching the existing cases.
* The model is integrated into a website using Streamlit.


# Important features:
* Automated Summaries: Generates concise summaries of cases, contracts, or statutes to capture key details.
* Clause or Case Similarity Analysis: Identifies similar cases, legal precedents, or contract clauses to aid in comparative analysis.
*  Named Entity Recognition (NER): Detect entities like parties involved, jurisdictions, dates, and legal terms.

# Future enhancements:
* The input Excel file is limited to 5 MB (up to 13 rows). In the future, Excel file inputs will be eliminated by adopting a RAG-based approach.
