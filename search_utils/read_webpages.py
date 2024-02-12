#!/usr/bin/env python3
import argparse
import base64
import json
import os
from urllib.parse import unquote
import requests


def url_to_unicode(text: str) -> str:
    decoded_string = unquote(text, 'utf-8')
    return decoded_string


def parse_url(
        url: str,
        service_url: str = None,
        need_links: bool = False,
        enable_render: bool = True,
        extract_level: int = 1,
        timeout: int = 30,
) -> str:
    """
    Parse url into html text or raw text.

    Args:
        url (str): url to parse
        service_url (str, optional): service url. Defaults to None.
        enable_render (bool, optional): enable render on url to parse. Defaults to True.
        extract_level (int, optional): extract level. Defaults to 1. Here are the options:
            0: extract raw text only.  1: Just remove js, css, etc on url to parse.
            2: keep all. return raw html text. Note that 2, 3 will return base64 encoded html text.
    """
    if service_url is None:
        # service_url = "http://172.16.100.225:8081/fetch"  # local url
        service_url = "http://lb-2nshjbik-dfpxqkgc3sr5jomn.clb.ap-guangzhou.tencentclb.com/fetch"
    payload = json.dumps({
        "url": url,
        "need_links": need_links,
        "enable_render": enable_render,
        "extract_level": extract_level,
    })
    print(f"parse url: {url}")
    headers = {'Content-Type': 'application/json'}
    response = requests.request("GET", service_url, headers=headers, data=payload)
    if response.status_code != 200:
        raise Exception(f"Fail to request url: {url}, code: {response.status_code}")

    response_json = json.loads(response.text)
    if response_json["status_code"] != 0:
        raise Exception(f"parse url {url} failed")
    parsed_html = response_json["content"]
    if extract_level != 0:
        try:
            parsed_html = base64.b64decode(parsed_html).decode("utf-8")
        except Exception:
            pass

    return parsed_html


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--url", type=str, required=True, help="url to parse")
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--search_method", type=str, default="vanilla_search")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    search_method = args.search_method
    url = args.url
    assert search_method in ["vanilla_search", "gpt4_rewrite_search", "without_search", "llama_rewrite_search", "gold_search"]

    if not os.path.exists(f"./user_intent_data/{dataset}/bing/{search_method}/"):
        os.makedirs(f"./user_intent_data/{dataset}/bing/{search_method}/", exist_ok=True)
    with open(f"./user_intent_data/{dataset}/bing/{search_method}/bing-{dataset}-{split}.jsonl", "w",
              encoding="utf-8") as f:
        pass
    url = "https://finance.eastmoney.com/a/202310252881476837.html"
    #  retry 3 times
    webpage = ''
    patience = 3
    while True:
        try:
            webpage = parse_url(url)
            break
        except Exception as e:
            print(f"parse url {url} failed, error: {e}")
            patience -= 1
            if patience <= 0:
                break
    print(webpage)

