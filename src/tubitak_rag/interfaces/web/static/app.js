/* Sohbet paneli davranışları: gönderilen soru hemen akışta görünür, cevap gelince yerine
   sunucunun turu eklenir; atıf → kaynak, Enter ile gönderme, "Yeni sohbet", yapılandırma
   seçilince yöntem satırı. Sayfa JavaScript olmadan da çalışır (form /?soru=… yükler). */
(function () {
  "use strict";
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var emptyState = null; // "Yeni sohbet" için boş durumun kopyası

  function $(selector, root) {
    return (root || document).querySelector(selector);
  }

  function openSource(id, moveFocus) {
    var item = document.getElementById(id);
    if (!item || !(item.classList.contains("kaynak") || item.classList.contains("bolum"))) {
      return false;
    }
    var details = item.querySelector("details");
    if (details) details.open = true;
    item.classList.remove("hedef");
    void item.offsetWidth; // vurgu animasyonunu yeniden başlat
    item.classList.add("hedef");
    item.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "nearest" });
    if (moveFocus) {
      var target = item.querySelector("a, summary");
      if (target) target.focus({ preventScroll: true });
    }
    return true;
  }

  // --- bekleyen tur: soru anında görünür ---
  function pendingTurn(question) {
    var turn = document.createElement("section");
    turn.className = "tur bekliyor";
    turn.id = "bekleyen-tur";
    turn.setAttribute("aria-label", "Soru ve cevap");
    var ask = document.createElement("div");
    ask.className = "mesaj soru-mesaji";
    var who = document.createElement("p");
    who.className = "mesaj-kim";
    who.textContent = "Soru";
    var text = document.createElement("p");
    text.className = "soru-metni";
    text.textContent = question;
    ask.append(who, text);
    var reply = document.createElement("div");
    reply.className = "mesaj cevap-mesaji";
    var status = document.createElement("p");
    status.className = "yukleniyor";
    status.setAttribute("role", "status");
    status.textContent = "Kaynaklar aranıyor, cevap yazılıyor…";
    reply.append(status);
    turn.append(ask, reply);
    return turn;
  }

  function removePending() {
    var pending = document.getElementById("bekleyen-tur");
    if (pending) pending.remove();
  }

  // İsteği başlatan öğe: swap olaylarında detail.elt hedeftir, requestConfig.elt formdur.
  function isAskForm(event) {
    var config = event.detail.requestConfig;
    var elt = config && config.elt ? config.elt : event.detail.elt;
    return Boolean(elt) && elt.id === "soru-formu";
  }

  document.addEventListener("htmx:beforeRequest", function (event) {
    if (!isAskForm(event)) return;
    var question = $("#soru").value.trim();
    var thread = $("#sohbet");
    var empty = $(".bos-durum", thread);
    if (empty) empty.remove();
    removePending();
    if (question) {
      var turn = pendingTurn(question);
      thread.append(turn);
      turn.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    }
  });

  document.addEventListener("htmx:beforeSwap", function (event) {
    if (isAskForm(event)) removePending();
  });

  document.addEventListener("htmx:afterRequest", function (event) {
    if (!isAskForm(event) || !event.detail.successful) return;
    var box = $("#soru");
    box.value = "";
    box.focus({ preventScroll: true });
  });

  function failPending() {
    var status = $("#bekleyen-tur .yukleniyor");
    if (!status) return;
    status.textContent = "Sunucuya ulaşılamadı. Bağlantıyı kontrol edip yeniden deneyin.";
    status.className = "form-hatasi";
    status.setAttribute("role", "alert");
    document.getElementById("bekleyen-tur").removeAttribute("id");
  }
  document.addEventListener("htmx:sendError", function (event) {
    if (isAskForm(event)) failPending();
  });
  document.addEventListener("htmx:responseError", function (event) {
    if (isAskForm(event)) failPending();
  });

  document.addEventListener("htmx:afterSwap", function (event) {
    if (event.detail.target.id !== "sohbet") return;
    var turns = event.detail.target.querySelectorAll(".tur");
    var last = turns[turns.length - 1];
    if (!last) return;
    // Odak yazma alanında kalır (sonraki soru); yeni tur role="log" ile duyurulur.
    last.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  });

  // --- tıklamalar: atıf, örnek soru, yeni sohbet ---
  document.addEventListener("click", function (event) {
    if (!event.target.closest) return;
    var cite = event.target.closest("a.atif");
    if (cite) {
      var id = cite.getAttribute("href").slice(1);
      if (openSource(id, false)) event.preventDefault();
      return;
    }
    var example = event.target.closest("a[data-ornek]");
    var form = $("#soru-formu");
    if (example && form && window.htmx) {
      event.preventDefault();
      $("#soru").value = example.getAttribute("data-ornek");
      form.requestSubmit();
      return;
    }
    // ?soru= ile açılan sayfada boş durum kopyası yok: bağlantı "/" adresini yükler.
    var reset = event.target.closest("[data-yeni-sohbet]");
    if (reset && emptyState && $("#sohbet")) {
      event.preventDefault();
      var thread = $("#sohbet");
      thread.replaceChildren(emptyState.cloneNode(true));
      history.replaceState(null, "", "/");
      document.title = "Sohbet · TÜBİTAK Mevzuat Rehberi";
      $("#soru").value = "";
      $("#soru").focus();
    }
  });

  // Enter gönderir, Shift+Enter yeni satır (Ctrl/Cmd+Enter da gönderir).
  document.addEventListener("keydown", function (event) {
    if (event.target.id !== "soru" || event.key !== "Enter" || event.isComposing) return;
    if (event.shiftKey) return;
    event.preventDefault();
    if (event.target.value.trim()) event.target.form.requestSubmit();
  });

  // Seçilen yapılandırmanın yöntem satırını göster.
  document.addEventListener("change", function (event) {
    if (event.target.id !== "yapilandirma") return;
    var chosen = event.target.value;
    document.querySelectorAll(".yontem[data-yapilandirma]").forEach(function (line) {
      line.hidden = line.getAttribute("data-yapilandirma") !== chosen;
    });
  });

  window.addEventListener("DOMContentLoaded", function () {
    var empty = $("#sohbet .bos-durum");
    if (empty) emptyState = empty.cloneNode(true);
    // Tarayıcı geri gelince seçici eski değerde kalabilir: yöntem satırını eşitle.
    var select = $("#yapilandirma");
    if (select) select.dispatchEvent(new Event("change", { bubbles: true }));
    // #kaynak-… (cevaptaki kaynak) ya da #bolum-… (belge önizlemesindeki bölüm) açılır.
    if (/^#(kaynak|bolum)-/.test(location.hash)) {
      openSource(decodeURIComponent(location.hash.slice(1)), false);
    }
  });
})();
