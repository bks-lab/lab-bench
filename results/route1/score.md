# route1: arms against the gold labels

Run 1 of each arm. Base for McNemar: `jev`. Unanswered rows count as wrong.

## massive scenario, de

| arm | n | answered | acc | macro-F1 | base | skill | only jev right | only arm right | p (McNemar) | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| jev | 2974 | 2974 | 72.8 % | 73.3 % | 13.5 % | 0.686 | | | | 232 |
| gliner2.5-decide (English model, extra) | 2974 | 2974 | 49.0 % | 46.3 % | 13.5 % | 0.410 | 856 | 147 | 0.000 | 26 |
| gliner2.5-decide-1b (English model, extra) | 2974 | 2974 | 41.1 % | 41.1 % | 13.5 % | 0.319 | 1149 | 205 | 0.000 | 17 |
| gliner2.5-decide-1b-intext (English model, extra) | 2974 | 2974 | 40.2 % | 39.7 % | 13.5 % | 0.308 | 1183 | 212 | 0.000 | 18 |
| gliner2.5-decide-intext (English model, extra) | 2974 | 2974 | 35.9 % | 33.7 % | 13.5 % | 0.259 | 1182 | 85 | 0.000 | 26 |
| gliner2.5-multi-decide | 2974 | 2974 | 48.5 % | 46.4 % | 13.5 % | 0.404 | 912 | 188 | 0.000 | 14 |
| gliner2.5-multi-decide-intext | 2974 | 2974 | 46.7 % | 46.9 % | 13.5 % | 0.384 | 957 | 181 | 0.000 | 14 |
| winnow-12b | 2974 | 2974 | 71.8 % | 72.3 % | 13.5 % | 0.674 | 192 | 161 | 0.110 | 89 |

## massive intent, de

| arm | n | answered | acc | macro-F1 | base | skill | only jev right | only arm right | p (McNemar) | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| jev | 2974 | 2974 | 91.8 % | 87.2 % | 56.1 % | 0.812 | | | | 233 |
| gliner2.5-decide (English model, extra) | 2974 | 2974 | 70.0 % | 64.8 % | 56.1 % | 0.316 | 709 | 61 | 0.000 | 26 |
| gliner2.5-decide-1b (English model, extra) | 2974 | 2974 | 61.6 % | 53.2 % | 56.1 % | 0.125 | 967 | 69 | 0.000 | 14 |
| gliner2.5-decide-1b-intext (English model, extra) | 2974 | 2974 | 56.8 % | 46.3 % | 56.1 % | 0.017 | 1112 | 73 | 0.000 | 17 |
| gliner2.5-decide-intext (English model, extra) | 2974 | 2974 | 62.9 % | 57.5 % | 56.1 % | 0.156 | 931 | 74 | 0.000 | 26 |
| gliner2.5-multi-decide | 2974 | 2974 | 63.8 % | 57.2 % | 56.1 % | 0.175 | 899 | 67 | 0.000 | 13 |
| gliner2.5-multi-decide-intext | 2974 | 2974 | 63.3 % | 55.3 % | 56.1 % | 0.165 | 916 | 71 | 0.000 | 14 |
| winnow-12b | 2974 | 2973 | 90.3 % | 83.4 % | 56.1 % | 0.779 | 115 | 71 | 0.002 | 85 |

## massive intent, 2+ options, de

| arm | n | answered | acc | macro-F1 | base | skill | only jev right | only arm right | p (McNemar) | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| jev | 2622 | 2622 | 90.7 % | 86.5 % | 50.2 % | 0.812 | | | | 234 |
| gliner2.5-decide (English model, extra) | 2622 | 2622 | 65.9 % | 62.9 % | 50.2 % | 0.316 | 709 | 61 | 0.000 | 26 |
| gliner2.5-decide-1b (English model, extra) | 2622 | 2622 | 56.4 % | 50.7 % | 50.2 % | 0.125 | 967 | 69 | 0.000 | 14 |
| gliner2.5-decide-1b-intext (English model, extra) | 2622 | 2622 | 51.0 % | 43.4 % | 50.2 % | 0.017 | 1112 | 73 | 0.000 | 18 |
| gliner2.5-decide-intext (English model, extra) | 2622 | 2622 | 58.0 % | 55.2 % | 50.2 % | 0.156 | 931 | 74 | 0.000 | 26 |
| gliner2.5-multi-decide | 2622 | 2622 | 58.9 % | 54.9 % | 50.2 % | 0.175 | 899 | 67 | 0.000 | 13 |
| gliner2.5-multi-decide-intext | 2622 | 2622 | 58.4 % | 52.9 % | 50.2 % | 0.165 | 916 | 71 | 0.000 | 14 |
| winnow-12b | 2622 | 2621 | 89.0 % | 82.5 % | 50.2 % | 0.779 | 115 | 71 | 0.002 | 84 |

## massive scenario, en

| arm | n | answered | acc | macro-F1 | base | skill | only jev right | only arm right | p (McNemar) | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| jev | 2974 | 2974 | 74.1 % | 74.2 % | 13.5 % | 0.701 | | | | 231 |
| gliner2.5-decide | 2974 | 2974 | 60.9 % | 60.2 % | 13.5 % | 0.547 | 543 | 149 | 0.000 | 26 |
| gliner2.5-decide-1b | 2974 | 2974 | 53.3 % | 56.8 % | 13.5 % | 0.460 | 786 | 167 | 0.000 | 18 |
| gliner2.5-decide-1b-intext | 2974 | 2974 | 52.3 % | 56.6 % | 13.5 % | 0.449 | 817 | 169 | 0.000 | 18 |
| gliner2.5-decide-intext | 2974 | 2974 | 49.8 % | 51.9 % | 13.5 % | 0.420 | 813 | 91 | 0.000 | 26 |
| gliner2.5-multi-decide | 2974 | 2974 | 52.0 % | 50.3 % | 13.5 % | 0.445 | 844 | 186 | 0.000 | 14 |
| gliner2.5-multi-decide-intext | 2974 | 2974 | 50.1 % | 51.1 % | 13.5 % | 0.423 | 886 | 172 | 0.000 | 14 |
| winnow-12b | 2974 | 2974 | 70.6 % | 71.8 % | 13.5 % | 0.661 | 237 | 134 | 0.000 | 92 |

## massive intent, en

| arm | n | answered | acc | macro-F1 | base | skill | only jev right | only arm right | p (McNemar) | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| jev | 2974 | 2974 | 92.9 % | 89.0 % | 56.1 % | 0.838 | | | | 230 |
| gliner2.5-decide | 2974 | 2974 | 80.9 % | 75.6 % | 56.1 % | 0.565 | 428 | 71 | 0.000 | 26 |
| gliner2.5-decide-1b | 2974 | 2974 | 75.7 % | 72.1 % | 56.1 % | 0.446 | 572 | 60 | 0.000 | 14 |
| gliner2.5-decide-1b-intext | 2974 | 2974 | 71.2 % | 67.1 % | 56.1 % | 0.345 | 704 | 59 | 0.000 | 15 |
| gliner2.5-decide-intext | 2974 | 2974 | 74.0 % | 70.0 % | 56.1 % | 0.407 | 631 | 68 | 0.000 | 26 |
| gliner2.5-multi-decide | 2974 | 2974 | 68.0 % | 62.2 % | 56.1 % | 0.270 | 812 | 70 | 0.000 | 13 |
| gliner2.5-multi-decide-intext | 2974 | 2974 | 68.3 % | 63.9 % | 56.1 % | 0.279 | 791 | 60 | 0.000 | 14 |
| winnow-12b | 2974 | 2974 | 92.6 % | 87.1 % | 56.1 % | 0.832 | 82 | 74 | 0.575 | 88 |

## massive intent, 2+ options, en

| arm | n | answered | acc | macro-F1 | base | skill | only jev right | only arm right | p (McNemar) | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| jev | 2622 | 2622 | 92.0 % | 88.4 % | 50.2 % | 0.838 | | | | 230 |
| gliner2.5-decide | 2622 | 2622 | 78.3 % | 74.3 % | 50.2 % | 0.565 | 428 | 71 | 0.000 | 26 |
| gliner2.5-decide-1b | 2622 | 2622 | 72.4 % | 70.6 % | 50.2 % | 0.446 | 572 | 60 | 0.000 | 14 |
| gliner2.5-decide-1b-intext | 2622 | 2622 | 67.4 % | 65.3 % | 50.2 % | 0.345 | 704 | 59 | 0.000 | 15 |
| gliner2.5-decide-intext | 2622 | 2622 | 70.5 % | 68.4 % | 50.2 % | 0.407 | 631 | 68 | 0.000 | 26 |
| gliner2.5-multi-decide | 2622 | 2622 | 63.7 % | 60.2 % | 50.2 % | 0.270 | 812 | 70 | 0.000 | 13 |
| gliner2.5-multi-decide-intext | 2622 | 2622 | 64.1 % | 62.0 % | 50.2 % | 0.279 | 791 | 60 | 0.000 | 14 |
| winnow-12b | 2622 | 2622 | 91.6 % | 86.4 % | 50.2 % | 0.832 | 82 | 74 | 0.575 | 88 |

## fastdec, all single-label tasks, en

| arm | n | answered | acc | macro-F1 | base | skill | only jev right | only arm right | p (McNemar) | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| jev | 2600 | 2600 | 63.0 % | 57.0 % | 30.4 % | 0.469 | | | | 240 |
| gliner2.5-decide | 2600 | 2600 | 63.3 % | 59.2 % | 30.4 % | 0.473 | 299 | 307 | 0.776 | 26 |
| gliner2.5-decide-1b | 2600 | 2600 | 63.5 % | 58.8 % | 30.4 % | 0.475 | 328 | 339 | 0.699 | 27 |
| gliner2.5-decide-1b-intext | 2600 | 2600 | 64.6 % | 61.1 % | 30.4 % | 0.492 | 308 | 349 | 0.119 | 26 |
| gliner2.5-decide-intext | 2600 | 2600 | 63.6 % | 60.2 % | 30.4 % | 0.477 | 317 | 332 | 0.583 | 26 |
| gliner2.5-multi-decide | 2600 | 2600 | 58.0 % | 54.4 % | 30.4 % | 0.396 | 449 | 317 | 0.000 | 14 |
| gliner2.5-multi-decide-intext | 2600 | 2600 | 58.2 % | 55.1 % | 30.4 % | 0.400 | 447 | 322 | 0.000 | 14 |
| winnow-12b | 2600 | 2500 | 61.2 % | 47.1 % | 30.4 % | 0.443 | 180 | 132 | 0.008 | 120 |

## fastdec, tasks with 26 options or fewer, en

| arm | n | answered | acc | macro-F1 | base | skill | only jev right | only arm right | p (McNemar) | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| jev | 2500 | 2500 | 63.0 % | 55.9 % | 31.2 % | 0.461 | | | | 240 |
| gliner2.5-decide | 2500 | 2500 | 63.3 % | 59.4 % | 31.2 % | 0.466 | 293 | 301 | 0.774 | 26 |
| gliner2.5-decide-1b | 2500 | 2500 | 63.4 % | 59.1 % | 31.2 % | 0.468 | 322 | 334 | 0.668 | 26 |
| gliner2.5-decide-1b-intext | 2500 | 2500 | 64.4 % | 60.4 % | 31.2 % | 0.482 | 305 | 341 | 0.168 | 26 |
| gliner2.5-decide-intext | 2500 | 2500 | 63.4 % | 60.0 % | 31.2 % | 0.468 | 313 | 325 | 0.663 | 26 |
| gliner2.5-multi-decide | 2500 | 2500 | 57.7 % | 53.4 % | 31.2 % | 0.385 | 440 | 309 | 0.000 | 14 |
| gliner2.5-multi-decide-intext | 2500 | 2500 | 58.0 % | 54.3 % | 31.2 % | 0.389 | 439 | 315 | 0.000 | 14 |
| winnow-12b | 2500 | 2500 | 63.6 % | 55.3 % | 31.2 % | 0.471 | 115 | 132 | 0.309 | 120 |

## fastdec per task, en (accuracy; base in the last column)

| task | jev | gliner2.5-decide | gliner2.5-decide-1b | gliner2.5-decide-1b-intext | gliner2.5-decide-intext | gliner2.5-multi-decide | gliner2.5-multi-decide-intext | winnow-12b | base |
|---|---|---|---|---|---|---|---|---|---|
| agent_handoff:should_handoff | 76.0 % | 74.0 % | 65.0 % | 72.0 % | 72.0 % | 55.0 % | 57.0 % | 78.0 % | 54.0 % |
| banking_intent:intent | 55.0 % | 62.0 % | 61.0 % | 62.0 % | 63.0 % | 48.0 % | 51.0 % | 49.0 % | 13.0 % |
| benefits_request:asking_status | 59.0 % | 57.0 % | 58.0 % | 55.0 % | 60.0 % | 61.0 % | 59.0 % | 61.0 % | 54.0 % |
| benefits_request:program | 90.0 % | 82.0 % | 87.0 % | 88.0 % | 84.0 % | 79.0 % | 80.0 % | 92.0 % | 15.0 % |
| clinic_request:request | 53.0 % | 54.0 % | 55.0 % | 55.0 % | 49.0 % | 45.0 % | 46.0 % | 51.0 % | 13.0 % |
| clinic_request:urgent | 76.0 % | 69.0 % | 70.0 % | 73.0 % | 66.0 % | 74.0 % | 58.0 % | 75.0 % | 56.0 % |
| document_type:doc_type | 78.0 % | 80.0 % | 78.0 % | 78.0 % | 81.0 % | 70.0 % | 69.0 % | 78.0 % | 16.0 % |
| email_triage:action | 53.0 % | 56.0 % | 56.0 % | 57.0 % | 55.0 % | 51.0 % | 55.0 % | 53.0 % | 30.0 % |
| email_triage:category | 42.0 % | 46.0 % | 50.0 % | 50.0 % | 48.0 % | 48.0 % | 47.0 % | 40.0 % | 18.0 % |
| email_triage:is_phishing | 56.0 % | 59.0 % | 56.0 % | 60.0 % | 68.0 % | 55.0 % | 54.0 % | 57.0 % | 53.0 % |
| email_triage:needs_reply | 65.0 % | 66.0 % | 62.0 % | 70.0 % | 68.0 % | 57.0 % | 62.0 % | 65.0 % | 53.0 % |
| news_topic:topic | 59.0 % | 65.0 % | 63.0 % | 60.0 % | 66.0 % | 57.0 % | 58.0 % | 65.0 % | 14.0 % |
| paper_field:field | 39.0 % | 41.0 % | 48.0 % | 51.0 % | 43.0 % | 38.0 % | 37.0 % | 40.0 % | 16.0 % |
| product_feedback:feedback_type | 57.0 % | 70.0 % | 73.0 % | 73.0 % | 67.0 % | 63.0 % | 62.0 % | 58.0 % | 32.0 % |
| restaurant_review:sentiment | 80.0 % | 80.0 % | 79.0 % | 83.0 % | 84.0 % | 73.0 % | 74.0 % | 80.0 % | 36.0 % |
| review_sentiment:sentiment | 84.0 % | 83.0 % | 82.0 % | 84.0 % | 86.0 % | 77.0 % | 76.0 % | 87.0 % | 36.0 % |
| screen_tags:format | 80.0 % | 75.0 % | 78.0 % | 78.0 % | 76.0 % | 78.0 % | 80.0 % | 81.0 % | 58.0 % |
| sports_recap:result | 90.0 % | 85.0 % | 88.0 % | 86.0 % | 85.0 % | 66.0 % | 68.0 % | 92.0 % | 31.0 % |
| sports_recap:sport | 75.0 % | 63.0 % | 58.0 % | 54.0 % | 62.0 % | 53.0 % | 53.0 % | 77.0 % | 16.0 % |
| sports_recap:upset | 70.0 % | 64.0 % | 58.0 % | 56.0 % | 51.0 % | 55.0 % | 54.0 % | 68.0 % | 53.0 % |
| support_intent:intent | 65.0 % | 65.0 % | 64.0 % | 70.0 % | 68.0 % | 64.0 % | 64.0 % | 0.0 % | 9.0 % |
| support_topic:topic | 42.0 % | 49.0 % | 44.0 % | 43.0 % | 47.0 % | 47.0 % | 50.0 % | 41.0 % | 11.0 % |
| ticket_route:contains_pii | 69.0 % | 51.0 % | 69.0 % | 64.0 % | 53.0 % | 56.0 % | 59.0 % | 80.0 % | 52.0 % |
| ticket_route:queue | 37.0 % | 51.0 % | 53.0 % | 54.0 % | 55.0 % | 56.0 % | 56.0 % | 29.0 % | 11.0 % |
| ticket_route:urgency | 32.0 % | 39.0 % | 34.0 % | 41.0 % | 34.0 % | 28.0 % | 29.0 % | 38.0 % | 27.0 % |
| travel_request:intent | 57.0 % | 61.0 % | 61.0 % | 63.0 % | 63.0 % | 53.0 % | 56.0 % | 56.0 % | 13.0 % |

## Jev cost

| source | lang | requests | median input tokens | USD per 1,000 requests |
|---|---|---|---|---|
| massive | de | 2974 | 894 | 0.0375 |
| massive | en | 2974 | 888 | 0.0373 |
| fastdec | en | 1700 | 633 | 0.0266 |

USD 0.042 per million input tokens, output free (docs.typesafe.ai/models, read 2026-10-02).
