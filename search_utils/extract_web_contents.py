#!/usr/bin/env python3
import argparse
import base64
import concurrent.futures
import json
from time import sleep
from urllib.parse import unquote
import requests
from tqdm import tqdm


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
    # print(f"parse url: {url}")
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


def get_web_content_from_retrieved_urls(search_results):
    """
    Args:
        search_results (list): a list of search results
    """
    if len(search_results) == 0:
        return search_results
    if isinstance(search_results[0], dict):
        for search_result in search_results:
            url = search_result["url"]
            #  use snippet as web content if url is not available
            webpage = search_result["snippet"]
            patience = 2
            while True:
                try:
                    webpage = parse_url(url, extract_level=0, enable_render=True)
                    break
                except Exception as e:
                    # print(f"parse url {url} failed, error: {e}")
                    patience -= 1
                    sleep(1)
                    if patience <= 0:
                        break
            search_result["webcontent"] = webpage
    else:
        for idx, _search_result in enumerate(search_results):
            search_results[idx] = get_web_content_from_retrieved_urls(_search_result)
    return search_results


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="webglm-qa")
    arg_parser.add_argument("--split", type=str, default="test")
    arg_parser.add_argument("--search_method", type=str, default="vanilla_search")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    search_method = args.search_method
    assert search_method in ["vanilla_search", "gpt4_rewrite_search", "without_search", "llama_rewrite_search",
                             "gold_search"]

    bing_search_file = f"./user_intent_data/{dataset}/bing/{search_method}/bing-{dataset}-{split}.jsonl"
    with open(bing_search_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f][896:]
    if args.mini_dataset:
        data_lines = data_lines[:10]
    print(f"dataset: {dataset}, split: {split}, search_method: {search_method}")

    webcontent_file = f"./user_intent_data/{dataset}/bing/{search_method}/webcontent-{dataset}-{split}-more.jsonl"
    with open(webcontent_file, "w", encoding="utf-8") as f:
        pass


    def parse_seg_data_lines(seg_data_lines, display=False):
        _iter = tqdm(enumerate(seg_data_lines), total=len(seg_data_lines), desc="webcontent") if display else enumerate(
            seg_data_lines)
        for idx, data_line in _iter:
            #  get web content
            search_results = data_line[f"{search_method}_results"]

            search_results = get_web_content_from_retrieved_urls(search_results)
            data_line[f"{search_method}_results"] = search_results
        return seg_data_lines


    para_workers = 32
    print(f"para_workers: {para_workers}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=para_workers) as executor:
        seg_size = len(data_lines) // para_workers + 1
        seg_data_lines = [data_lines[i * seg_size: (i + 1) * seg_size] for i in range(para_workers)]
        futures = []
        for idx, seg_data_line in enumerate(seg_data_lines):
            if idx == 0:
                futures.append(executor.submit(parse_seg_data_lines, seg_data_line, True))
            else:
                futures.append(executor.submit(parse_seg_data_lines, seg_data_line, False))
        for future in concurrent.futures.as_completed(futures):
            pass
        data_lines = []
        for future in futures:
            data_lines.extend(future.result())

    # for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines), desc="webcontent"):
    #     #  get web content
    #     search_results = data_line[f"{search_method}_results"]
    #
    #     search_results = get_web_content_from_retrieved_urls(search_results)
    #     data_line[f"{search_method}_results"] = search_results

    with open(webcontent_file, "a", encoding="utf-8") as f:
        for data_line in data_lines:
            f.write(json.dumps(data_line, ensure_ascii=False) + "\n")
