import re, json, sys, urllib.request, html, os, time, unicodedata
UA={"User-Agent":"Mozilla/5.0"}
def get(u): return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=60).read()
page=get("https://tubitak.gov.tr/tr/mevzuat").decode("utf-8")
art=page[page.find("<article"):page.rfind("</article>")]
# anchors in document order; also capture following "Güncelleme tarihi"
links=[]; seen=set()
ms=list(re.finditer(r'<a[^>]+href="([^"]+\.pdf)"[^>]*>(.*?)</a>',art,re.S|re.I))
for k,m in enumerate(ms):
    h=html.unescape(m.group(1)); h=h if h.startswith("http") else "https://tubitak.gov.tr"+h
    t=re.sub(r"\s+"," ",re.sub("<[^>]+>","",html.unescape(m.group(2)))).strip()
    if not t or h in seen: continue
    seen.add(h)
    tail=art[m.end():ms[k+1].start() if k+1<len(ms) else m.end()+400]
    g=re.search(r"(Güncelleme|Mülga) tarihi\s*:\s*([\d.]+)",re.sub("<[^>]+>"," ",tail))
    links.append({"baslik":t,"url":h,"tarih_turu":g.group(1) if g else None,"tarih":g.group(2) if g else None})
print("found",len(links)); 
SEL=[5,9,11,13,23,30,31,32,33,34,35,37,40,42,43,44,45,47,48,49,50,56,57,63,64,65,66,67,68,69,70,73,79,84,85,86,87,88,92,94,95,96,97,98,100,101,102,103,105,112,113,116,118,119,120,127,129,130,131,135]
assert len(links)==136, len(links)
assert "1501" in links[30]["baslik"] and "1814" in links[135]["baslik"] and "İtiraz" in links[116]["baslik"], [links[i]["baslik"] for i in (30,116,135)]
out=sys.argv[1]; os.makedirs(out,exist_ok=True)
man=[]
for i in SEL:
    d=links[i]; fn=f"{i:03d}_"+os.path.basename(d["url"])
    data=get(d["url"]); assert data[:4]==b"%PDF",fn
    open(os.path.join(out,fn),"wb").write(data)
    pk=re.search(r"\b(1\d{3}|3001|4\d{3}|5\d{3})\b",d["baslik"])
    man.append({**d,"dosya":fn,"boyut":len(data),"program_kodu":pk.group(1) if pk else None,"kaynak_sayfa":"https://tubitak.gov.tr/tr/mevzuat","indirme_tarihi":"2026-10-15"})
    time.sleep(0.5)
json.dump({"kaynak":"https://tubitak.gov.tr/tr/mevzuat","secim":"başvuru süreçleriyle ilgili 60 doküman (136 içinden)","dokumanlar":man},open(os.path.join(out,"manifest.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("indirildi",len(man),"toplam MB",round(sum(x["boyut"] for x in man)/1e6,1))
