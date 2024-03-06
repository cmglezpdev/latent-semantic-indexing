from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import ir_datasets
from collections import defaultdict
from query_builder import boolean_model_retrieveral, documents_retrieveral_LSI

def get_data():
    """Structure corpus in three variables

    Returns:
        _type_: queries, qrels and docs
    """
    
    dataset = ir_datasets.load("cranfield")
    queries = { query.query_id: query.text for query in dataset.queries_iter() }
    
    qrels = defaultdict(list)
    [qrels[query.query_id].append(int(query.doc_id)) for query in dataset.qrels_iter()]
    qrels = dict(qrels)

    docs = { doc.doc_id: doc.text for doc in dataset.docs_iter() }
    
    return queries, qrels, docs
    

def gen_comp_vecs(true_docs, ret_docs, docs):
    doc_ids = np.array(docs)
    y_true = np.array([1 if doc_id in true_docs else 0 for doc_id in doc_ids])
    y_pred = np.array([1 if doc_id in ret_docs else 0 for doc_id in doc_ids])
    return y_true, y_pred


def evaluate_model(fn_process_query):
    precisions = []
    recalls = []
    f1_scores = []

    queries, qrels, docs = get_data()
    for query_id in queries.keys():
        if query_id not in qrels.keys():
            continue
        
        # retrived documents using own model implementation
        retrived_docs = [id for id, _ in fn_process_query(queries[query_id])]
        relevant_docs = qrels[query_id]
        docs_ids = list(map(lambda x: int(x), list(docs.keys())))
        y_true, y_pred = gen_comp_vecs(relevant_docs, retrived_docs, docs_ids)
        
        p_s = precision_score(y_true, y_pred)
        r_s = recall_score(y_true, y_pred)
        f1_s = f1_score(y_true, y_pred)
        
        precisions.append(p_s)
        recalls.append(r_s)
        f1_scores.append(f1_s)

    return {
        'precision': np.average(precisions),
        'recall': np.average(recalls),
        'f1_score': np.average(f1_scores) 
    }
    
    
    
    

if __name__ == "__main__":
    lsi_metrics = evaluate_model(documents_retrieveral_LSI)  
    # boolean_metrics = evaluate_model(boolean_model_retrieveral)
    
    print(f"Metrics for LSI\n{lsi_metrics}")
    # print(f"Metrics for Boolean Model\n{boolean_metrics}")
      