# C3 method: instruction and schema, fixed before the first arm

Written 2026-10-09, before any C3 arm ran. Design: `plan/c3.md`. Every arm
gets the same instruction below, verbatim, with no per-model change and no
few-shot example. `bench/c3/run_c3.py` reads it from this file (the text
between the two `text` fences of the section "Instruction"), so the file
and the run cannot drift apart.

## Instruction

```text
Du bekommst eine Rechnung. Lies die folgenden 12 Felder aus und gib sie als JSON-Objekt mit genau diesen Schlüsseln zurück:

invoice_number: Rechnungsnummer
issue_date: Rechnungsdatum, im Format JJJJ-MM-TT
due_date: Fälligkeitsdatum, im Format JJJJ-MM-TT
seller_name: Name des Verkäufers (Firmenname)
seller_vat_id: Umsatzsteuer-Identifikationsnummer des Verkäufers
buyer_name: Name des Käufers
iban: IBAN des Kontos, auf das der Käufer überweisen soll
net_total: Gesamtsumme ohne Umsatzsteuer
vat_total: Summe der Umsatzsteuer
gross_total: Gesamtsumme mit Umsatzsteuer
amount_due: fälliger Betrag
currency: Währung als dreistelliger Code, zum Beispiel EUR

Regeln:
- Beträge als Zahl mit Punkt als Dezimaltrennzeichen und zwei Nachkommastellen, ohne Tausendertrennzeichen und ohne Währungszeichen, zum Beispiel 1234.56.
- Übernimm alle anderen Werte so, wie sie auf der Rechnung stehen.
- Steht ein Feld nicht auf der Rechnung, gib null zurück.
- Gib nur das JSON-Objekt zurück, ohne weiteren Text.
```

## How the invoice is attached

- Vision arms: the instruction is the text of one user message, and every
  page of the invoice is attached to that message as a PNG image, in page
  order.
- Text arms (`docling+...` and the reference arm `pdftext+qwen3.8-27b`):
  one user message, the instruction, then a blank line, then
  `Rechnung:` and the invoice text (docling Markdown of all pages joined by
  a blank line, or the `pdftotext -layout` text of the PDF).

## Output schema (Ollama `format`)

```json
{
  "type": "object",
  "properties": {
    "invoice_number": {"type": ["string", "null"]},
    "issue_date": {"type": ["string", "null"]},
    "due_date": {"type": ["string", "null"]},
    "seller_name": {"type": ["string", "null"]},
    "seller_vat_id": {"type": ["string", "null"]},
    "buyer_name": {"type": ["string", "null"]},
    "iban": {"type": ["string", "null"]},
    "net_total": {"type": ["string", "null"]},
    "vat_total": {"type": ["string", "null"]},
    "gross_total": {"type": ["string", "null"]},
    "amount_due": {"type": ["string", "null"]},
    "currency": {"type": ["string", "null"]}
  },
  "required": ["invoice_number", "issue_date", "due_date", "seller_name",
               "seller_vat_id", "buyer_name", "iban", "net_total",
               "vat_total", "gross_total", "amount_due", "currency"]
}
```

## Options

`temperature` 0, `seed` 1, `num_ctx` 32768 for every arm (see the dated
section of `plan/c3.md` for why not 8192), `stream` false, `think` false
for every model whose `/api/show` capabilities list `thinking` (as in the
earlier local arms, which switch thinking off in the template). A call whose
`prompt_eval_count` reaches `num_ctx` minus 64 is flagged `truncated` and
scored as returned. Three warm-up calls on the first three invoices run
before the timed pass and are not scored. One retry on an HTTP or JSON
error; a second failure scores all 12 cells of that invoice as wrong.

## Scoring

`bench/c3/fields.py`: exact match per cell after the normalisation of the
field table in `plan/c3.md` (money to a decimal with two places, dates to
ISO, names whitespace collapsed and casefolded, VAT id, IBAN and currency
uppercase without spaces, empty string and the string `null` read as null).
