import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from pyserini.search.lucene import LuceneSearcher
import sys

from kilt_bm25.rerank_passage import rerank_passage

sys.path.append('./')
from kilt.knowledge_source import KnowledgeSource

# from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
# get the knowledge souce
ks = KnowledgeSource()

bm25_searcher = LuceneSearcher('../indexes/lucene-index.wikipedia-kilt-doc.20210421.f29307')


# encoder = TctColBertQueryEncoder('../../../../huggingface/dpr-question_encoder-multiset-base')
# dpr_searcher = FaissSearcher('../indexes/faiss.wikipedia-dpr-100w.dpr_multi.20200127.f403c3.fe307ef2e60ab6e6f3ad66e24a4144ae',encoder)

# Create your views here.

@csrf_exempt
def bm25_search(request):
    response = {}
    try:
        queries = json.loads(request.body)["queries"]
        batch_hits = bm25_searcher.batch_search(queries, qids=queries, k=10)
        response["search_results"] = []
        for qids, hits in batch_hits.items():
            doc_ids = [hit.docid for hit in hits]
            scores = [hit.score for hit in hits]
            docs = [ks.get_page_by_id(doc_id)["text"] for doc_id in doc_ids]
            response["search_results"].append({"doc_ids": doc_ids, "scores": scores, "docs": docs})
    except Exception as e:
        response["error"] = str(e)
    return JsonResponse(response)


@csrf_exempt
def bm25_search_e5_rerank(request):
    response = {}
    try:
        queries = json.loads(request.body)["queries"]
        question = json.loads(request.body)["question"]
        batch_hits = bm25_searcher.batch_search(queries, qids=queries, k=10)
        response["search_results"] = []
        for qids, hits in batch_hits.items():
            doc_ids = [hit.docid for hit in hits]
            scores = [hit.score for hit in hits]
            docs = [ks.get_page_by_id(doc_id)["text"] for doc_id in doc_ids]

            response["search_results"].append({"doc_ids": doc_ids, "scores": scores, "docs": docs})
        #  do rerank
        rerank_results = rerank_passage(question, response["search_results"])
        response["search_results"] = rerank_results
    except Exception as e:
        response["error"] = str(e)
    return JsonResponse(response)

# @csrf_exempt
# def dpr_search(request):
#     response = {}
#     # try:
#     queries = json.loads(request.body)["queries"]
#     batch_hits = dpr_searcher.batch_search(queries, q_ids=queries, k=10)
#     response["search_results"] = []
#     for qids, hits in batch_hits.items():
#         doc_ids = [hit.docid for hit in hits]
#         scores = [hit.score for hit in hits]
#         docs = [ks.get_page_by_id(doc_id)["text"] for doc_id in doc_ids]
#         response["search_results"].append({"doc_ids": doc_ids, "scores": scores, "docs": docs})
#     # except Exception as e:
#     #     response["error"] = str(e)
#     return JsonResponse(response)
