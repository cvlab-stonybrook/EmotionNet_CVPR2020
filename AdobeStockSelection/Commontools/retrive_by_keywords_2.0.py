import requests, json, math, re, sys, argparse

class TextQueryMeta(object):

    def __init__(self):
        self.qps_url = 'https://qps-uw2.dev.adobesearch.io'
        self.es_url = 'https://sar-es-552-c2-uw2.dev.adobesearch.io/search_sar_1128/'
        self.MAX_ATTEMPTS = 10

    def qps(self, queries):
        payload = {'text': ' '.join(queries), 'locale': 'en_US', 'sort': 'ksvm_rank_v10'}
        attempts = 0
        while attempts < self.MAX_ATTEMPTS:
            try:
                resp = requests.get(self.qps_url, params=payload)
                if resp.status_code == requests.codes.ok:
                    return resp.text
            except:
                attempts += 1
            else:
                attempts += 1
                resp.close()
        raise ConnectionError('pqs query failed')

    def es_request(self, qps_resp, media, size, tag_field):
        es_body = self._create_es_body(qps_resp, size, tag_field)
        attempts = 0
        while attempts < self.MAX_ATTEMPTS:
            try:
                resp=requests.post(self.es_url+'/_search', data=es_body)
                if resp.status_code == requests.codes.ok:
                    return self._build_resps(resp.json(), tag_field, media)
            except:
                attempts += 1
            else:
                attempts += 1
                resp.close()
        raise ConnectionError('elasticseach query failed')

    
    def _create_es_body(self, qps_resp, size, tag_field):
        query = json.loads(qps_resp)["query"]["query"]["filtered"]["query"]
        for r in query['function_score']['functions']:
            if 'filter' not in r:
                continue
            if 'and' not in r['filter']:
                continue
            if 'and' in r['filter']['and'][0]:
                andQ = r['filter']['and'][0]['and']['queries']
                r['filter']['bool'] = {'filter': andQ}
                del r['filter']['and']
            else:
                andQ = r['filter']['and']
                r['filter']['bool'] = {'filter': andQ}
                del r['filter']['and']
        query = json.dumps(query)
        replacements = [
                ["boost_factor", "weight"],
                ["k.eksr", "k.eksrr"],
                ["k.okgnbv", "k.okgnbvr"],
                ["k.kgnbv", "k.kgnbvr"],
                ["r.us.srs","r.us.srrs"]
            ]
        for r in replacements:
            query = query.replace(r[0], r[1])
        es_body = {
            "query": json.loads(query),
            "_source":[
                "cid",
                "id32",
                "k."+tag_field,
                "gv"
            ],
            "from":0,
            "size":size
        }
        return json.dumps(es_body)

    def _build_resps(self, es_resp, tag_field, media):
        rst = []
        for hit in es_resp["hits"]["hits"]:
            if "cid" not in hit["_source"] or "id32" not in hit["_source"]: continue
            if "k" not in hit["_source"] or tag_field not in hit["_source"]["k"]: continue
            if "gv" not in hit["_source"]: continue

            
            cid = hit["_source"]["cid"]
            id32 = hit["_source"]["id32"]
            url = self._make_url_from_ftcdn(cid, id32, media)
            tags = hit["_source"]["k"][tag_field]
            title = hit["_source"]["gv"]
            resp = {
                "cid": cid,
                "id32": id32,
                "url": url,
                "tags": tags,
                "title": title
            }
            rst.append(resp)
        return rst

    def _make_url_from_ftcdn(self, cid, id32, media):
        if media == 'image':
            return 'https://as2.ftcdn.net/jpg/{}/{}_F_{}_{}.jpg'.format(
                '/'.join(re.findall('..', '{:0>10}'.format(cid))[0:4]),
                1000,
                cid,
                id32
            )
        elif media == 'video':
            return 'https://v.ftcdn.net/{}/{}_F_{}_{}_ST.mp4'.format(
                '/'.join(re.findall('..', '{:0>10}'.format(cid))[0:4]),
                700,
                cid,
                id32
            )
        else:
            raise ValueError('wrong media')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--media', default='image', choices=['image', 'video'], type=str)
    parser.add_argument('--size', default=10000, type=int)
    parser.add_argument('--tag_field', default='eksrr', choices=['oksrr', 'eksrr'], type=str)
    parser.add_argument('--queries', required=True, nargs='+', type=str, help='queries separated by whilte space')
    parser.add_argument('--outf', required=True, type=str, help='output filename')
    opt = parser.parse_args()

    textQueryMeta = TextQueryMeta()
    qps_resp = textQueryMeta.qps(queries=opt.queries)
    resps = textQueryMeta.es_request(qps_resp=qps_resp, media=opt.media, size=opt.size, tag_field=opt.tag_field)
    # print(resps)
    with open(opt.outf, 'w') as outf:
        for resp in resps:
            outf.write(json.dumps(resp)+"\n")