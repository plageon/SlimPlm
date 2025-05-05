# -*- coding: utf-8 -*-
import sys
import json
import logging
import requests
from urllib.parse import urlencode

class BingSearch(object):
    
    BASE_PATH = "https://api.bing.microsoft.com/v7.0/search?{}"
    SECRET_KEY = "your secret key"
    
    @classmethod
    def _query(cls, query, count, mkt="zh-CN"):
        params = {
            "q": query,
            "textDecorations":False,
            "mkt":"en-US",
            "count": count,
            "SafeSearch": "Strict"
        }
        
        url = BingSearch.BASE_PATH.format(urlencode(params))
        # print(url)
        rsp = requests.get(url, headers={'Ocp-Apim-Subscription-Key': BingSearch.SECRET_KEY})
        if rsp is None or rsp.status_code != 200:
            return False, "http request failed"
        
        return True, rsp.json()
    
    @classmethod
    def search(cls, query: str, count=10, db_handler=None):
        ret, jdata = cls._query(query, count)
        if ret == False or 'webPages' not in jdata or 'value' not in jdata['webPages']:
            # logging.error("[search]-[error] query failed, query:{}, msg:{}".format(query, jdata))
            return []
        
        # print(json.dumps(jdata, ensure_ascii=False))
        return jdata['webPages']['value']
    
if __name__ == '__main__':
    query = ''
    if len(sys.argv) >= 2: query=sys.argv[1]
    BingSearch.search(query)
