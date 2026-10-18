"""学習曲線を測るための「取り置き文書」を、巡回とは別に取得して保存する(約20ページ、礼儀正しく)。"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import forager
URLS = """https://en.wikipedia.org/wiki/Photosynthesis
https://en.wikipedia.org/wiki/Roman_Empire
https://en.wikipedia.org/wiki/Jazz
https://en.wikipedia.org/wiki/Black_hole
https://simple.wikipedia.org/wiki/Volcano
https://simple.wikipedia.org/wiki/Bicycle
https://ja.wikipedia.org/wiki/%E5%AF%8C%E5%A3%AB%E5%B1%B1
https://ja.wikipedia.org/wiki/%E5%AF%BF%E5%8F%B8
https://de.wikipedia.org/wiki/Photosynthese
https://de.wikipedia.org/wiki/Fu%C3%9Fball
https://fr.wikipedia.org/wiki/R%C3%A9volution_fran%C3%A7aise
https://fr.wikipedia.org/wiki/Fromage
https://es.wikipedia.org/wiki/F%C3%BAtbol
https://es.wikipedia.org/wiki/Don_Quijote_de_la_Mancha
https://zh.wikipedia.org/wiki/%E5%9C%B0%E7%90%83
https://ru.wikipedia.org/wiki/%D0%9C%D0%BE%D1%81%D0%BA%D0%B2%D0%B0
https://en.wikipedia.org/wiki/Python_(programming_language)
https://docs.python.org/3/library/json.html
https://docs.python.org/3/tutorial/classes.html
https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array
https://en.wikiquote.org/wiki/Albert_Einstein
https://en.wikibooks.org/wiki/Cookbook:Pancake""".split()
g = forager.Guard(os.environ.get('FORAGER_CONTACT') or 'research-prototype-no-contact')
out = []
for u in URLS:
    r = g.fetch(u)
    if r.get('status') == 200:
        text = forager.extract(r['url'], r['body'], r['charset'], r['ctype'])[0]
        out.append({'url': u, 'text': text})
        print('ok  %6d字 %s' % (len(text), u))
    else:
        print('NG', r.get('status'), r.get('error'), u)
    time.sleep(3)
json.dump(out, open(os.path.join(HERE, 'heldout.json'), 'w'), ensure_ascii=False)
print(len(out), 'ページ保存')
