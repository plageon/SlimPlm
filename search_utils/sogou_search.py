# -*- coding: utf-8 -*-

import re
import sys
import json
import time
import logging
import hashlib
import requests
from urllib.parse import quote 

class SogouSearch(object):
    
    BASE_PATH = "https://api.bing.microsoft.com/v7.0/search?{}"
    SECRET_KEY = "d73c4b858f3f457a845598c4859e178b"
    
    
    # URL_FMT = 'https://m.sogou.com/commonsearch?keyword={keyword}&pid={pid}&sign={sign}&salt={salt}&cnt={cnt}'
    URL_FMT = 'https://m.sogou.com/search4baichuan?keyword={keyword}&pid={pid}&sign={sign}&salt={salt}&cnt={cnt}'
    PID = 'search_f23a6e1d25df68ec2a29cb9496b15763'
    KEY = 'c526047da4782688c328614cf00b80f2'
    
    white_vrid_lizhi_set = (
            "70167200", "70320300", "70283200", "70287700", "70284500", "70279200", "70244600",
            "70277100", "70202200", "70276900", "70273900", "81003001", "81123001", "81122001",
            "81002001", "81013001", "81121101", "81121001", "81001101", "81002101", "81122101", 
            "81012001", "81007001", "81001001", "81127001", "81013004", "81007101", "81002003",
            "81012004", "81011101", "81012101", "81122004", "81011001", "81127101", "81003003",
            "81123004", "81003004", "81001003", "81017001", "81011004", "81017004", "81017104",
            "81002004", "81001103", "81121004", "81017101", "81011104", "81007003", "80055001",
            "81001004", "81121104", "80065001", "81007004", "81127104", "81127004", "81007103",
            "81007104", "81001104", "41040102", "41040101", "41110108", "41020100", "41020101",
            "41010100", "41010300", "41040142", "41020141", "50029301", "50030401", "50028001",
            "50027401", "50029101",
            
            "81095067", "41040141", "41110148"
        )

    white_vrid_mingyi_set = (
            "11026801", "11030201", "11034301", "11024401", "11034401", "11018001", "11020901",
            "11018801", "11037501", "11024601", "11035401", "70207500", "11036001", "11036601",
            "70165100", "11018301", "11021901", "70161000", "70282200", "70161700", "70320700", 
            "70315800", "70213700", "70173800", "70180500", "70299200", "70243500", "70187200",
            "70196000", "70291700", "70202000", "70207600", "11030001", "70107000", "70207700",
            "70170800", "70201600", "70212600", "70217900", "70218200", "70220300", "70204900",
            "70297800", "70220600", "70252600", "70221000", "70228600", "70239200", "70239300",
            "70252800", "70254200", "70301700", "70284800", "70284900", "70282000", "70310500",
            "70124900", "70285000", "70285100", "70290600", "70296900", "70306900", "11026501",
            "70210600", "70310600", "70228700", "70251700", "21493101"
        )
    
    white_vrid_txzxg_set = ("70280301")
    
    white_vrid_jscj_set = ("70130702", "70130701")
    
    white_site_list =[
            '.gov.cn', # www.sasac.gov.cn www.mod.gov.cn www.fmprc.gov.cn
            'people.com', # 人民网
            'news.cn', # 新华社
            'news.china.com', # news.china.com.cn 中华网
            '.yebaike.com', # 业百科
            'zhihu.com', # 业百科
            'xinhuanet.com',
            '163.com',
            'toutiao.com',
            'weixin.qq.com'
            # 'bala.iask.sina', # m.bala.iask.sina.com.cn
        ]
    
    @classmethod
    def _gen_sign(cls, keyword, timestamp):
        md5hash = hashlib.md5((SogouSearch.PID+keyword+str(timestamp)+SogouSearch.KEY).encode('utf-8'))
        return md5hash.hexdigest()
    
    @classmethod
    def _parse_abstract_info_all(cls, query, jdata={}, topk=1):
        """wzq.tenpay.com
        """
        url, title, lizhi_title, answer_type, type, select_type, baike_answer, baike_abstract = '', '', '', '', '', '', '', ''
        content = ''
        select_type = 'unknown'
        results = []
        try:
            reg_html = re.compile('<[^>]*>')
            for page in jdata['pages']:
                vrid, url, title, lizhi_title, answer_type, type, select_type = '', '', '', '', '', '', ''
                attribute, steps, data_list = {}, {}, {}
                content = ''
                
                vrid = page.get('vrid', '')
                display = page.get('display', {})
                url = display.get('url', '')
                title = display.get('title', '')
                lizhi_title = display.get('lizhi_title', '')
                answer_type = display.get('answer_type', '')
                type = display.get('@type', '')
                longanswer = display.get('longanswer', '')
                abstract_info = display.get('abstract_info', '')
                accurContent = display.get('accurContent', '')
                baike_answer = display.get('subitem', {}).get('subdisplay', {}).get('answerinfo', {}).get('longanswer', '')
                baike_abstract = display.get('kdJsonStr', {}).get('module_list', [{}])[0].get('item_list', [{}])[0].get('data', {}).get('feature', {}).get('baike_abstract', '')
                
                json_data_group = page.get('jsonData', {}).get('display_info', {}).get('group', [])
                
                attribute = display.get('attribute', {})
                steps = display.get('steps', {})
                data_list_str = display.get('subitem', {}).get('subdisplay', {}).get('data_list', '')
                try:
                    if data_list_str: 
                        data_list = json.loads(data_list_str)
                except:
                    data_list = {}
                
                select_type = 'unknown'
                # 优先选择 立知 + 搜索精选 内容
                if vrid and vrid in SogouSearch.white_vrid_lizhi_set:
                    select_type = 'lizhi_vr'
                elif vrid and vrid in SogouSearch.white_vrid_mingyi_set:
                    select_type = 'mingyi_vr'
                elif answer_type and len(answer_type) > 0:
                    select_type = 'answer_type'
                elif url and 'baike.sogou' in url:
                    select_type = 'baike_sogou'
                elif type in ['220', '2210']:
                    select_type = 'type_220'
                elif lizhi_title and len(lizhi_title) > 0:
                    select_type = 'lizhi_title'
                elif (vrid and vrid in SogouSearch.white_vrid_txzxg_set) or len(json_data_group) > 0:
                    select_type = 'tx_zxg'
                elif (vrid and vrid in SogouSearch.white_vrid_jscj_set) or 'jinse.cn' in url:
                    select_type = 'jscj'
                
                content = ''
                if select_type == 'tx_zxg':
                    for group in json_data_group:
                        content += ("类型:" + group.get('type', '-') + "\n")
                        content += ("股票:" + group.get('name', '-') + "  代码:" + group.get('code_stock', '-') + "\n")
                        content += ("时间:" + group.get('time', '-') + "  状态:" + group.get('key_status', '-') + "\n")
                        content += ("价格:" + group.get('price', '-') + "  今开:" + group.get('number_open', '-') + "  昨收:" + group.get('number_closed', '-') + "\n")
                        content += ("最高:" + group.get('number_high', '-') + "  最低:" + group.get('number_low', '-') + "\n")
                elif select_type == 'jscj':
                    subdisplay = display.get('subitem', {}).get('subdisplay', {})
                    content += (subdisplay.get('title', {}).get('#text', '') + "\n")
                    content += ("时间:" + subdisplay.get('time', '-') + "\n")
                    content += ("价格:" + subdisplay.get('price', '-') + "\n")
                    content += ("最高:" + subdisplay.get('high', '-') + "  最低:" + subdisplay.get('low', '-') + "\n")
                elif select_type == 'mingyi_vr':
                    # content: display[subitem][subdisplay][contents][detail][content_detail]
                    # content: display[subDisplay][subitem][[content]]
                    # content: display[tab_list][tab_name + tab_summary]
                    content_detail = display.get('subitem', {}).get('subdisplay', {}).get('contents', {}).get('detail', {}).get('content_detail', '')
                    print(content_detail)
                    
                    subitem_content = ''
                    content_list = display.get('subDisplay', {}).get('subitem', [])
                    if len(content_list) > 0: subitem_content = content_list[0]['content']
                    if len(content_list) > 1: subitem_content += content_list[1]['content']
                    
                    tab_content = ''
                    tab_list = display.get('tab_list', [])
                    for idx, tab in enumerate(tab_list):
                        tab_name = tab.get('tab_name', '')
                        tab_summary = tab.get('tab_summary', '')
                        if tab_name: tab_content += (tab_name + "\n")
                        if tab_summary: tab_content += (tab_summary + "\n")
                        if idx >= 6 or len(tab_content) > 500: break
                        
                    if content_detail: content += (content_detail + "\n")
                    if subitem_content: content += (subitem_content + "\n")
                    if tab_content: content += (tab_content + "\n")
                    content = content[:500]
                else:
                    if lizhi_title:  content = lizhi_title + '\n'
                    elif title: content = lizhi_title + '\n'
                    
                    if longanswer: content += (longanswer + '\n')
                    elif abstract_info: content += (abstract_info + '\n')
                    elif accurContent: content += (accurContent + '\n')
                    elif baike_answer: content += (baike_answer + '\n')
                    elif baike_abstract: content += (baike_abstract + '\n')
                    
                    if attribute and isinstance(attribute, dict):
                        name = attribute.get('@name', '')
                        text = attribute.get('element', {}).get('answer', {}).get('@text', '')
                        desc = attribute.get('element', {}).get('desc', '')
                        if name and text:
                            content += (name + '\n' + text + '\n')
                        if desc: content += (desc + '\n')
                    elif steps and isinstance(steps, dict):
                        for st in steps.get('step', []):
                            text = st.get('step_text', '')
                            if text: content += (text + '\n')
                    elif data_list and isinstance(data_list, dict):
                        for entity in data_list.get('entity_list', []):
                            entity_name = entity.get('entity_name', '')
                            feature_list = entity.get('feature_label_list', [])
                            desc = entity.get('desc', '')
                            
                            if entity_name: content += (entity_name + '\n')
                            if feature_list and len(feature_list) > 0: content += (feature_list[0] + '\n')
                            if desc: content += (desc + '\n')   
                
                logging.info("Titile:{}".format(title))
                if content:
                    results.append({
                        'query': query,
                        'url': url,
                        'select_type': select_type,
                        'snippet': reg_html.sub('', content).replace('\\\\', '')
                    })
                    if len(results) >= topk:
                        return results
        except Exception as e:
            print(e) 
            return results
        return results
    
    @classmethod
    def _parse_abstract_info(cls, query, jdata={}, topk=1):      
        url, title, lizhi_title, answer_type, type, select_type, baike_answer, baike_abstract = '', '', '', '', '', '', '', ''
        content = ''
        select_type = 'unknown'
        results = []
        try:
            reg_html = re.compile('<[^>]*>')
            for page in jdata['pages']:
                vrid, url, title, lizhi_title, answer_type, type, select_type = '', '', '', '', '', '', ''
                attribute, steps, data_list = {}, {}, {}
                content = ''
                
                vrid = page.get('vrid', '')
                display = page.get('display', {})
                url = display.get('url', '')
                title = display.get('title', '')
                lizhi_title = display.get('lizhi_title', '')
                answer_type = display.get('answer_type', '')
                type = display.get('@type', '')
                longanswer = display.get('longanswer', '')
                abstract_info = display.get('abstract_info', '')
                accurContent = display.get('accurContent', '')
                baike_answer = display.get('subitem', {}).get('subdisplay', {}).get('answerinfo', {}).get('longanswer', '')
                baike_abstract = display.get('kdJsonStr', {}).get('module_list', [{}])[0].get('item_list', [{}])[0].get('data', {}).get('feature', {}).get('baike_abstract', '')
                
                attribute = display.get('attribute', {})
                steps = display.get('steps', {})
                data_list_str = display.get('subitem', {}).get('subdisplay', {}).get('data_list', '')
                try:
                    if data_list_str: 
                        data_list = json.loads(data_list_str)
                except:
                    data_list = {}
                
                select_type = 'unknown'
                # 优先选择 立知 + 搜索精选 内容
                if vrid and vrid in SogouSearch.white_vrid_lizhi_set:
                    select_type = 'lizhi_vr'
                elif vrid and vrid in SogouSearch.white_vrid_mingyi_set:
                    select_type = 'mingyi_vr'
                elif answer_type and len(answer_type) > 0:
                    select_type = 'answer_type'
                elif url and 'baike.sogou' in url:
                    select_type = 'baike_sogou'
                elif type in ['220', '2210']:
                    select_type = 'type_220'
                elif lizhi_title and len(lizhi_title) > 0:
                    select_type = 'lizhi_title'
                
                # 筛选优质内容
                if select_type == 'unknown' and url and isinstance(url, str) and len(url) > 0:
                    for site in SogouSearch.white_site_list:
                        if site in url:
                            select_type = 'wihte_site'
                            break
                
                content = ''
                if select_type == 'mingyi_vr':
                    # content: display[subitem][subdisplay][contents][detail][content_detail]
                    # content: display[subDisplay][subitem][[content]]
                    # content: display[tab_list][tab_name + tab_summary]
                    content_detail = display.get('subitem', {}).get('subdisplay', {}).get('contents', {}).get('detail', {}).get('content_detail', '')
                    print(content_detail)
                    
                    subitem_content = ''
                    content_list = display.get('subDisplay', {}).get('subitem', [])
                    if len(content_list) > 0: subitem_content = content_list[0]['content']
                    if len(content_list) > 1: subitem_content += content_list[1]['content']
                    
                    tab_content = ''
                    tab_list = display.get('tab_list', [])
                    for idx, tab in enumerate(tab_list):
                        tab_name = tab.get('tab_name', '')
                        tab_summary = tab.get('tab_summary', '')
                        if tab_name: tab_content += (tab_name + "\n")
                        if tab_summary: tab_content += (tab_summary + "\n")
                        if idx >= 6 or len(tab_content) > 500: break
                        
                    if content_detail: content += (content_detail + "\n")
                    if subitem_content: content += (subitem_content + "\n")
                    if tab_content: content += (tab_content + "\n")
                    content = content[:500]
                elif select_type != 'unknown':
                    if lizhi_title:  content = lizhi_title + '\n'
                    elif title: content = lizhi_title + '\n'
                    
                    if longanswer: content += (longanswer + '\n')
                    elif abstract_info: content += (abstract_info + '\n')
                    elif accurContent: content += (accurContent + '\n')
                    elif baike_answer: content += (baike_answer + '\n')
                    elif baike_abstract: content += (baike_abstract + '\n')
                    
                    if attribute and isinstance(attribute, dict):
                        name = attribute.get('@name', '')
                        text = attribute.get('element', {}).get('answer', {}).get('@text', '')
                        desc = attribute.get('element', {}).get('desc', '')
                        if name and text:
                            content += (name + '\n' + text + '\n')
                        if desc: content += (desc + '\n')
                    elif steps and isinstance(steps, dict):
                        for st in steps.get('step', []):
                            text = st.get('step_text', '')
                            if text: content += (text + '\n')
                    elif data_list and isinstance(data_list, dict):
                        for entity in data_list.get('entity_list', []):
                            entity_name = entity.get('entity_name', '')
                            feature_list = entity.get('feature_label_list', [])
                            desc = entity.get('desc', '')
                            
                            if entity_name: content += (entity_name + '\n')
                            if feature_list and len(feature_list) > 0: content += (feature_list[0] + '\n')
                            if desc: content += (desc + '\n')   

                if content:
                    results.append({
                        'query': query,
                        'url': url,
                        'select_type': select_type,
                        'snippet': reg_html.sub('', content).replace('\\\\', '')
                    })
                    if len(results) >= topk:
                        return results
        except Exception as e:
            print(e) 
            return results
        return results
    
    @classmethod
    def search(cls, query: str, topk: int):
        keyword = query.lstrip()
        time_s = int(time.time()*1000)
        url = SogouSearch.URL_FMT.format(
            keyword=quote(keyword),
            pid=SogouSearch.PID,
            sign=cls._gen_sign(keyword, time_s),
            salt=time_s,
            cnt=10
        )
        
        print(url) # http_cli = HttpClient()
        rsp = requests.get(url)
        if rsp.status_code != 200:
            logging.error("[search]-[failed] http request failed, {}".format(rsp.status_code))
            return []
        
        # logging.info("Response:{}".format(json.dumps(rsp.json(), ensure_ascii=False)))
        print("Response:{}", json.dumps(rsp.json(), ensure_ascii=False))
        return cls._parse_abstract_info_all(query, rsp.json(), topk=topk)

        
if __name__ == '__main__':
    logging.basicConfig(filename="search_app.log", level=logging.INFO)
    query='百川智能融资规模'
    if len(sys.argv) >= 2: query=sys.argv[1]
    
    #results = SogouSearch.search('李玟 site:baike.baidu.com', topk=3)
    results = SogouSearch.search(query, topk=3)
    print(json.dumps(results, ensure_ascii=False))
