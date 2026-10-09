# C3 render check

Every non-null gold value of `reference/c3/gold.jsonl` searched in the
`pdftotext -layout` output of its rendered PDF (`cases/c3/text/`).

Result: 391 of 391 non-null gold values found (100.0 %). Gate: 100 %, PASSED.

| field | found | non-null |
|---|---|---|
| invoice_number | 34 | 34 |
| issue_date | 34 | 34 |
| due_date | 24 | 24 |
| seller_name | 34 | 34 |
| seller_vat_id | 33 | 33 |
| buyer_name | 34 | 34 |
| iban | 28 | 28 |
| net_total | 34 | 34 |
| vat_total | 34 | 34 |
| gross_total | 34 | 34 |
| amount_due | 34 | 34 |
| currency | 34 | 34 |
