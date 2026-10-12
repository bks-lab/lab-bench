"""Builds the C9 gold audit form (plan/c9.md, Data; notes of 2026-10-12).

    python bench/c9/audit.py [--out private/c9/gold-audit.html]

One self-contained HTML page with the 40 audit documents of
cases/c9/manifest.json. Every gold span of the seven types is highlighted and
gets a verdict (correct, not personal data, wrong boundary, wrong type);
missed spans are added per document. Spans with a dropped Gretel label are
shown greyed, so a mislabelled date of birth can be caught. The page saves
its state in the browser and exports private/c9/gold-audit.json, the file
bench/c9/score_c9.py reads. It shows no arm output: the audit stays blind.
The page lives under private/ (gitignored) because it will hold a person's
judgments; the documents themselves are committed in cases/c9/.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import c9lib  # noqa: E402

PAGE = r"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>C9 Gold-Audit</title>
<style>
:root{--bg:#fbfbf8;--fg:#1d1d1b;--muted:#6b6b66;--card:#ffffff;--line:#deded8;--hl:#ffe9a8;--hl-ok:#cdeccf;--hl-bad:#f6c9c4;--drop:#ececec;--accent:#2f5d8a}
@media (prefers-color-scheme: dark){:root{--bg:#171716;--fg:#ecece8;--muted:#a3a39c;--card:#22221f;--line:#3a3a35;--hl:#6b5a1f;--hl-ok:#2d5a33;--hl-bad:#6e2f29;--drop:#33332f;--accent:#8fb6dd}}
body{background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif;margin:0;padding:16px}
main{max-width:980px;margin:0 auto}
h1{font-size:1.5rem;margin:.2em 0}
.intro{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.doc{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:18px 0}
.doc h2{font-size:1.05rem;margin:0 0 6px}
.text{white-space:pre-wrap;font-family:ui-serif,Georgia,serif;border-left:3px solid var(--line);padding-left:12px;margin:8px 0 12px;overflow-wrap:anywhere}
mark{background:var(--hl);color:inherit;padding:0 2px;border-radius:3px}
mark.ok{background:var(--hl-ok)}mark.bad{background:var(--hl-bad)}
mark sup{font-size:.7em;color:var(--muted);margin-left:2px}
span.drop{background:var(--drop);border-radius:3px}
span.drop sup{font-size:.7em;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:.93rem}
td,th{border-bottom:1px solid var(--line);padding:5px 6px;text-align:left;vertical-align:top}
td.t{font-family:ui-monospace,monospace;overflow-wrap:anywhere}
.v label{margin-right:10px;white-space:nowrap}
input[type=text],select{font:inherit;padding:3px 6px;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--fg);max-width:100%}
.muted{color:var(--muted);font-size:.9rem}
.missed div{margin:4px 0;display:flex;gap:6px;flex-wrap:wrap}
.done{color:var(--accent);font-weight:600}
</style></head><body><main>
<h1>C9 Gold-Audit: 40 Dokumente</h1>
<div class="intro">
<p>Die Gold-Markierungen des Datensatzes hat selbst ein Sprachmodell erzeugt. Dieses Blatt prüft sie an 40 zufällig gezogenen Dokumenten (Ziehung festgelegt in <code>cases/c9/manifest.json</code>). Es zeigt keine Ergebnisse der getesteten Modelle, bitte vor dem Ende des Audits auch nicht in <code>results/c9/</code> schauen.</p>
<p><b>Je markierter Stelle</b> (gelb, Typ hochgestellt) eine Wahl: <i>richtig</i>; <i>kein Personenbezug</i> (z. B. „Ihnen“, ein Platzhalter wie „[E-Mail]“, eine Firma); <i>Grenze falsch</i> (dann den richtigen Wortlaut eintragen, so wie er im Text steht); <i>Typ falsch</i> (dann den richtigen Typ wählen). Ohne Wahl gilt die Stelle als richtig.</p>
<p><b>Je Dokument</b> unten unter „Übersehen“ jede personenbezogene Angabe eintragen, die nicht gelb markiert ist: Typ wählen, Wortlaut genau wie im Text. Grau hinterlegt sind Stellen mit einem Label, das nicht zählt (Datum, Firma, Uhrzeit, Bankcode, Passwort); ist darunter ein Geburtsdatum oder Ähnliches, gehört es nach „Übersehen“.</p>
<p>Typen: Person (Name), Adresse (Anschrift, Standort), E-Mail, Telefon, Konto/IBAN/Karte, Kennnummer (Steuer-ID, SV-Nummer, Kunden-, Personal-, Pass-, Führerscheinnummer, Benutzername, IP-Adresse), Geburtsdatum.</p>
<p>Der Stand wird im Browser gespeichert. Am Ende „JSON exportieren“ und die Datei als <code>private/c9/gold-audit.json</code> im Repository ablegen. Schwelle laut Plan: liegt die Präzision oder der Recall des Golds unter 0,90, gilt C9 als „gold noisy“ und es gibt keine Rangfolge.</p>
</div>
<div class="bar"><span id="progress"></span><button class="primary" id="export">JSON exportieren</button><button id="reset">Alles zurücksetzen</button><span class="muted" id="saved"></span></div>
<div id="docs"></div>
</main>
<script>
const DATA = __DATA__;
const TYPES = DATA.types, TYPE_DE = DATA.type_de;
const KEY = 'c9-gold-audit-v1';
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) { state = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); document.getElementById('saved').textContent = 'gespeichert ' + new Date().toLocaleTimeString(); } catch (e) { document.getElementById('saved').textContent = 'Browser-Speicher nicht verfügbar: vor dem Schließen exportieren'; } progress(); }
function st(id) { return state[id] ||= { spans: {}, missed: [], seen: false }; }
function esc(s) { return s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
function opt(t, sel) { return TYPES.map(x => `<option value="${x}"${x === sel ? ' selected' : ''}>${TYPE_DE[x]}</option>`).join(''); }
function renderText(d) {
  const marks = [];
  d.gold.forEach((g, i) => marks.push({ s: g[0], e: g[1], kind: 'gold', i, t: g[2] }));
  d.dropped.forEach(g => marks.push({ s: g[0], e: g[1], kind: 'drop', t: g[2] }));
  marks.sort((a, b) => a.s - b.s || b.e - a.e);
  let out = '', pos = 0;
  for (const m of marks) {
    if (m.s < pos) continue;
    out += esc(d.text.slice(pos, m.s));
    const inner = esc(d.text.slice(m.s, m.e));
    if (m.kind === 'gold') {
      const v = (st(d.id).spans[m.i] || {}).verdict || 'ok';
      out += `<mark id="m-${d.id}-${m.i}" class="${v === 'ok' ? '' : 'bad'}">${inner}<sup>${m.i + 1} ${TYPE_DE[m.t]}</sup></mark>`;
    } else out += `<span class="drop">${inner}<sup>${esc(m.t)}</sup></span>`;
    pos = m.e;
  }
  return out + esc(d.text.slice(pos));
}
function renderDoc(d) {
  const s = st(d.id);
  const rows = d.gold.map((g, i) => {
    const v = s.spans[i] || { verdict: 'ok' };
    const name = `v-${d.id}-${i}`;
    const r = (val, lab) => `<label><input type="radio" name="${name}" value="${val}"${v.verdict === val ? ' checked' : ''}> ${lab}</label>`;
    return `<tr><td>${i + 1}</td><td class="t">${esc(d.text.slice(g[0], g[1]))}</td><td>${TYPE_DE[g[2]]}</td><td class="v" data-doc="${d.id}" data-i="${i}">
      ${r('ok', 'richtig')}${r('not_pii', 'kein Personenbezug')}${r('boundary', 'Grenze falsch')}${r('type', 'Typ falsch')}
      <span class="extra">${v.verdict === 'boundary' ? `<input type="text" placeholder="richtiger Wortlaut" value="${esc(v.text || '')}">` : ''}${v.verdict === 'type' ? `<select>${opt(null, v.type)}</select>` : ''}</span></td></tr>`;
  }).join('');
  const missed = s.missed.map((m, j) => `<div data-j="${j}"><select>${opt(null, m.type)}</select><input type="text" size="40" placeholder="Wortlaut wie im Text" value="${esc(m.text || '')}"><button data-del="${j}">entfernen</button></div>`).join('');
  return `<section class="doc" id="doc-${d.id}"><h2>${d.n}. Dokument ${d.id} <span class="muted">${esc(d.type)}</span> ${s.seen ? '<span class="done">geprüft</span>' : ''}</h2>
    <div class="text">${renderText(d)}</div>
    ${d.gold.length ? `<table><tr><th>#</th><th>Stelle</th><th>Typ</th><th>Urteil</th></tr>${rows}</table>` : '<p class="muted">Keine Gold-Stelle der sieben Typen.</p>'}
    <h3 style="font-size:1rem">Übersehen</h3><div class="missed" data-doc="${d.id}">${missed}</div>
    <button data-add="${d.id}">Übersehene Stelle hinzufügen</button>
    <label style="margin-left:12px"><input type="checkbox" data-seen="${d.id}"${s.seen ? ' checked' : ''}> Dokument fertig geprüft</label></section>`;
}
function rerender(id) { const d = DATA.docs.find(x => x.id === id); document.getElementById('doc-' + id).outerHTML = renderDoc(d); }
function progress() { const n = DATA.docs.filter(d => st(d.id).seen).length; document.getElementById('progress').textContent = `${n} von ${DATA.docs.length} Dokumenten geprüft`; }
document.getElementById('docs').innerHTML = DATA.docs.map(renderDoc).join('');
progress();
document.addEventListener('change', e => {
  const t = e.target;
  const v = t.closest('.v');
  if (v) {
    const s = st(v.dataset.doc), i = v.dataset.i;
    if (t.type === 'radio') { s.spans[i] = { verdict: t.value }; if (t.value === 'type') s.spans[i].type = TYPES[0]; rerender(v.dataset.doc); }
    else if (t.tagName === 'SELECT') s.spans[i].type = t.value;
    save(); return;
  }
  const m = t.closest('.missed');
  if (m) { const s = st(m.dataset.doc), j = t.parentElement.dataset.j; if (t.tagName === 'SELECT') s.missed[j].type = t.value; save(); return; }
  if (t.dataset.seen) { st(t.dataset.seen).seen = t.checked; rerender(t.dataset.seen); save(); }
});
document.addEventListener('input', e => {
  const t = e.target;
  const v = t.closest('.v');
  if (v && t.type === 'text') { st(v.dataset.doc).spans[v.dataset.i].text = t.value; save(); return; }
  const m = t.closest('.missed');
  if (m && t.type === 'text') { st(m.dataset.doc).missed[t.parentElement.dataset.j].text = t.value; save(); }
});
document.addEventListener('click', e => {
  const t = e.target;
  if (t.dataset.add) { st(t.dataset.add).missed.push({ type: TYPES[0], text: '' }); rerender(t.dataset.add); save(); }
  else if (t.dataset.del !== undefined) { const id = t.closest('.missed').dataset.doc; st(id).missed.splice(Number(t.dataset.del), 1); rerender(id); save(); }
});
document.getElementById('export').onclick = () => {
  const out = { design: 'plan/c9.md', form: 'bench/c9/audit.py', docs_sha256: DATA.docs_sha256, audited_at: new Date().toISOString(), docs: {} };
  for (const d of DATA.docs) { const s = st(d.id); out.docs[d.id] = { seen: s.seen, spans: s.spans, missed: s.missed.filter(m => m.text && m.text.trim()) }; }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], { type: 'application/json' }));
  a.download = 'gold-audit.json'; a.click();
};
document.getElementById('reset').onclick = () => { if (confirm('Alle Eingaben löschen?')) { state = {}; save(); document.getElementById('docs').innerHTML = DATA.docs.map(renderDoc).join(''); } };
</script></body></html>
"""

TYPE_DE = {"person": "Person", "address": "Adresse", "email": "E-Mail", "phone": "Telefon",
           "iban_account": "Konto/IBAN/Karte", "id_number": "Kennnummer", "date_of_birth": "Geburtsdatum"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "private", "c9", "gold-audit.html"))
    a = ap.parse_args()
    man, docs, labels = c9lib.load(ROOT)
    by_id = {d["id"]: d for d in docs}
    out = []
    for n, did in enumerate(man["audit"]["ids"], 1):
        d = by_id[did]
        dropped = []
        for s in d["spans"]:
            if s["label"] in labels["dropped"]:
                b = c9lib.norm(d["text"], s["start"], s["end"])
                if b:
                    dropped.append([b[0], b[1], s["label"]])
        out.append({"n": n, "id": did, "type": d["document_type"], "text": d["text"],
                    "gold": [list(g) for g in c9lib.gold(d, labels)], "dropped": dropped})
    data = {"types": labels["types"], "type_de": TYPE_DE, "docs_sha256": man["docs_sha256"], "docs": out}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    page = PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    open(a.out, "w", encoding="utf-8").write(page)
    spans = sum(len(x["gold"]) for x in out)
    print(f"{a.out}: {len(out)} documents, {spans} gold spans to judge")


if __name__ == "__main__":
    main()
