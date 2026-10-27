import json, re, os, time, hashlib, html, urllib.request, urllib.parse, logging
logging.getLogger("pypdf").setLevel(logging.ERROR)
from pypdf import PdfReader
S=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(S, "..", "..", "data", "raw", "tubitak_program_sayfalari")
UA={"User-Agent":"Mozilla/5.0"}
def get(u): return urllib.request.urlopen(urllib.request.Request(urllib.parse.quote(u,safe=":/?=&%#"),headers=UA),timeout=120).read()
plan=json.load(open(f"{S}/program_plan.json",encoding="utf-8"))
sel=json.load(open(f"{S}/program_selected.json",encoding="utf-8"))
EXCL=r"kapanmış çağrı|tech-investr-2018|\b2019\b|katip çelebi|2018 yılı çağrısı ile destek verilen|uygulayıcı kuruluşların listesi|faaliyet gösteren kuruluşlar|faaliyet raporu ek-2|e-imza|senet hazırlama"
dropped=[]; todo=[]
for f in sel:
    if re.search(EXCL,(f["baslik"] or "").lower()): f["haric"]="eski_cagri_veya_liste"; dropped.append(f)
    else: todo.append(f)
os.makedirs(f"{OUT}/html",exist_ok=True); os.makedirs(f"{OUT}/dosyalar",exist_ok=True)
# 1) sayfa HTML'leri (article bölümü)
pages=[]
for p in plan["pages"]:
    raw=get(p["url"]).decode("utf-8","replace"); time.sleep(0.3)
    a=raw[raw.find("<article"):raw.rfind("</article>")+10]
    open(f"{OUT}/html/{p['slug']}.html","w",encoding="utf-8").write(a)
    pages.append({**p,"html_dosya":f"html/{p['slug']}.html","html_boyut":len(a)})
# 2) dosyalar
seen={}; kept=[]
def first_text(path):
    try: return re.sub(r"\s+"," ",(PdfReader(path).pages[0].extract_text() or ""))[:600]
    except Exception as e: return f"<okunamadi: {e}>"
for f in todo:
    try: data=get(f["url"]); time.sleep(0.3)
    except Exception as e: f["haric"]=f"indirme_hatasi: {e}"; dropped.append(f); continue
    h=hashlib.sha256(data).hexdigest()
    if h in seen:
        seen[h]["sayfalar"]=sorted(set(seen[h]["sayfalar"]+f["sayfalar"])); seen[h].setdefault("ayni_icerikli_urller",[]).append(f["url"]); continue
    fn=f["url"].split("/sites/default/files/")[1].replace("/","__")
    path=f"{OUT}/dosyalar/{fn}"; open(path,"wb").write(data)
    f.update(dosya=f"dosyalar/{fn}",boyut=len(data),sha256=h)
    if f["ext"]==".pdf":
        f["ilk_sayfa"]=first_text(path)
        if re.fullmatch(r"tıklayınız\.?|sıkça sorulan sorular için tıklayınız\.?|çağrı duyurusu için tıklayınız\.?",(f["baslik"] or "").strip().lower()):
            f["baslik_belirsiz"]=True
    seen[h]=f; kept.append(f)
json.dump({"kaynak":"tubitak.gov.tr/tr/destekler program sayfaları (sitemap; cagri-* arşivi hariç)","indirme_tarihi":"2026-10-15",
  "secim_kurali":"PDF/XLSX içinden yalnız çağrı metni, duyuru, SSS ve başvuru/hazırlama kılavuzları; görünürlük/kimlik rehberleri, form şablonları, sunumlar, eski kapanmış çağrılar, listeler hariç; aynı içerik sha256 ile tekilleştirildi",
  "sayfalar":pages,"dosyalar":kept,"elenenler":dropped,"erisilemeyen_sayfalar":[
   "https://tubitak.gov.tr/tr/destekler/akademik/ulusal-destek-programlari/3005-sosyal-ve-beseri-bilimlerde-yenilikci-cozumler-arastirma-projeleri-destek-programi",
   "https://tubitak.gov.tr/tr/destekler/sanayi/ulusal-destek-programlari/1512-girisimcilik-destek-programi-bigg",
   "https://tubitak.gov.tr/tr/destekler/sanayi/ulusal-destek-programlari/1607-bigg-kobi-mentor-arayuzu"]},
  open(f"{OUT}/manifest.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("sayfa",len(pages),"| dosya",len(kept),round(sum(f['boyut'] for f in kept)/1e6,1),"MB | elenen",len(dropped))
for f in kept:
    if f.get("baslik_belirsiz"): print("BELIRSIZ",f["dosya"][:70],"::",f["ilk_sayfa"][:160])
