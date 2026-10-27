import re, json, sys, os, time, html, urllib.request, urllib.parse
UA={"User-Agent":"Mozilla/5.0"}
S=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.join(S, "..", "..", "data", "raw")
MODE=sys.argv[1]  # plan | download
def get(u):
    return urllib.request.urlopen(urllib.request.Request(urllib.parse.quote(u,safe=":/?=&%#"),headers=UA),timeout=90).read()
mev=json.load(open(f"{ROOT}/tubitak_mevzuat/manifest.json",encoding="utf-8"))["dokumanlar"]
mev_names={os.path.basename(d["url"]).lower() for d in mev}
urls=[u.strip() for u in open(f"{S}/prog_urls.txt") if u.strip()]
pages=[]; files={}; erisilemeyen=[]
for u in urls:
    try: raw=get(u).decode("utf-8","replace"); time.sleep(0.3)
    except Exception as e: erisilemeyen.append({"url":u,"hata":str(e)}); continue
    a=raw[raw.find("<article"):raw.rfind("</article>")+10]
    title=html.unescape(re.search(r"<title>(.*?)</title>",raw,re.S).group(1)).split("|")[0].strip()
    slug=u.rstrip("/").split("/destekler/")[1].replace("/","__")
    pages.append({"url":u,"baslik":title,"slug":slug,"html_boyut":len(a),"_html":a})
    for m in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',a,re.S):
        h=html.unescape(m.group(1)); 
        if "/sites/default/files/" not in h: continue
        h=h if h.startswith("http") else "https://tubitak.gov.tr"+h
        t=re.sub(r"\s+"," ",re.sub("<[^>]+>","",html.unescape(m.group(2)))).strip()
        ext=os.path.splitext(urllib.parse.urlparse(h).path)[1].lower()
        name=os.path.basename(urllib.parse.unquote(urllib.parse.urlparse(h).path))
        key=f"{t} {name}".lower()
        if ext in (".doc",".docx",".odt",".rtf"): why="form_sablonu"
        elif ext not in (".pdf",".xlsx",".xls"): why="diger_format"
        elif re.search(r"sunum|sunu\b|presentation|sunumlar",key): why="sunum"
        elif re.search(r"form",key) and not re.search(r"kılavuz|kilavuz|rehber",key): why="form"
        elif name.lower() in mev_names: why="mevzuat_setinde_var"
        else: why=None
        f=files.setdefault(h,{"url":h,"baslik":t,"ext":ext,"dosya_adi":name,"sayfalar":[],"haric":why})
        if slug not in f["sayfalar"]: f["sayfalar"].append(slug)
keep=[f for f in files.values() if not f["haric"]]
for f in keep:
    try:
        r=urllib.request.urlopen(urllib.request.Request(urllib.parse.quote(f["url"],safe=":/?=&%#"),headers=UA,method="HEAD"),timeout=60)
        f["boyut"]=int(r.headers.get("Content-Length") or 0)
    except Exception as e: f["boyut"]=None; f["hata"]=str(e)
    time.sleep(0.2)
from collections import Counter
print("erisilemeyen:",erisilemeyen); print("sayfa:",len(pages),"| sayfa HTML toplam KB:",sum(p["html_boyut"] for p in pages)//1024)
print("link toplam:",len(files),"| haric:",Counter(f["haric"] for f in files.values() if f["haric"]))
print("alinacak:",len(keep),"| ext:",Counter(f["ext"] for f in keep),"| MB:",round(sum(f["boyut"] or 0 for f in keep)/1e6,1),"| hata:",sum(1 for f in keep if f.get("hata")))
big=sorted(keep,key=lambda f:-(f["boyut"] or 0))[:12]
for f in big: print(f"  {(f['boyut'] or 0)/1e6:5.1f}MB  {f['baslik'][:80]}")
json.dump({"pages":[{k:v for k,v in p.items() if k!='_html'} for p in pages],"files":list(files.values())},open(f"{S}/program_plan.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
if MODE=="download":
    out=f"{ROOT}/tubitak_program_sayfalari"; os.makedirs(f"{out}/html",exist_ok=True); os.makedirs(f"{out}/dosyalar",exist_ok=True)
    for p in pages: open(f"{out}/html/{p['slug']}.html","w",encoding="utf-8").write(p["_html"])
    for f in keep:
        if f.get("hata"): continue
        data=get(f["url"]); fn=f["url"].split("/sites/default/files/")[1].replace("/","__")
        open(f"{out}/dosyalar/{fn}","wb").write(data); f["dosya"]=fn; f["boyut"]=len(data); time.sleep(0.3)
    json.dump({"kaynak":"tubitak.gov.tr/tr/destekler program sayfaları (sitemap'ten, cagri-* arşivi hariç)","indirme_tarihi":"2026-10-15",
               "haric_tutma_kurallari":"doc/docx form şablonları, sunumlar, 'form' içeren ve kılavuz olmayan dosyalar, mevzuat setinde zaten olanlar",
               "sayfalar":[{**{k:v for k,v in p.items() if k!='_html'},"html_dosya":f"html/{p['slug']}.html"} for p in pages],
               "dosyalar":keep,"erisilemeyen_sayfalar":erisilemeyen,"haric_tutulanlar":[f for f in files.values() if f["haric"]]},
              open(f"{out}/manifest.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
    print("INDIRILDI", len([f for f in keep if f.get('dosya')]), "dosya,", round(sum(f['boyut'] or 0 for f in keep if f.get('dosya'))/1e6,1),"MB")
