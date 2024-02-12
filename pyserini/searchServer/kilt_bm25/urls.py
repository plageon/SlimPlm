from django.urls import path

from . import views

urlpatterns = [
    path("bm25/", views.bm25_search, name="bm25_search"),
    path("bm25_e5_rerank/", views.bm25_search_e5_rerank, name="bm25_search_e5_rerank"),
    # path("dpr/", views.dpr_search, name="dpr_search"),
]