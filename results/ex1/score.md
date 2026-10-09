# ex1: slot extraction against MASSIVE gold

Run 1 of each arm. Jev: not applicable (no extraction primitive).

## de

| arm | utterances | gold slots | predicted | P | R | F1 | exact frame | errors | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| no extraction (baseline) | 1000 | 941 | 0 | n/a | 0.0 % | 0.0 % | 33.5 % | | |
| gliner2.5-multi | 1000 | 941 | 604 | 28.8 % | 18.5 % | 22.5 % | 27.5 % | 0 | 19 |
| winnow-12b | 1000 | 941 | 714 | 52.2 % | 39.6 % | 45.1 % | 45.1 % | 0 | 389 |

## en

| arm | utterances | gold slots | predicted | P | R | F1 | exact frame | errors | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| no extraction (baseline) | 1000 | 937 | 0 | n/a | 0.0 % | 0.0 % | 33.5 % | | |
| gliner2-large | 1000 | 937 | 1491 | 31.0 % | 49.3 % | 38.1 % | 22.4 % | 0 | 30 |
| gliner2.5-multi | 1000 | 937 | 664 | 28.8 % | 20.4 % | 23.9 % | 25.6 % | 0 | 19 |
| winnow-12b | 1000 | 937 | 857 | 51.2 % | 46.9 % | 48.9 % | 44.9 % | 0 | 401 |

## F1 per slot type, de

| slot type | gold | gliner2.5-multi | winnow-12b |
|---|---|---|---|
| date | 137 | 30.5 % | 42.4 % |
| event_name | 87 | 52.2 % | 31.9 % |
| place_name | 86 | 0.0 % | 73.8 % |
| person | 77 | 2.4 % | 79.5 % |
| time | 64 | 35.0 % | 61.5 % |
| media_type | 42 | 0.0 % | 0.0 % |
| business_name | 27 | 37.8 % | 54.2 % |
| food_type | 25 | 10.8 % | 42.6 % |
| weather_descriptor | 25 | 13.6 % | 28.6 % |
| house_place | 23 | 37.5 % | 26.7 % |
| timeofday | 22 | 0.0 % | 40.0 % |
| artist_name | 21 | 41.9 % | 74.4 % |
| list_name | 20 | 0.0 % | 6.7 % |
| relation | 20 | 0.0 % | 16.7 % |
| transport_type | 20 | 0.0 % | 33.3 % |
| player_setting | 19 | 0.0 % | 0.0 % |
| device_type | 18 | 29.6 % | 16.7 % |
| currency_name | 17 | 60.0 % | 36.4 % |
| definition_word | 17 | 53.3 % | 53.7 % |
| music_genre | 16 | 52.9 % | 66.7 % |
| news_topic | 15 | 12.5 % | 38.3 % |
| radio_name | 15 | 8.7 % | 34.8 % |
| business_type | 14 | 14.3 % | 40.0 % |
| audiobook_name | 11 | 0.0 % | 66.7 % |
| podcast_descriptor | 11 | 0.0 % | 0.0 % |
| color_type | 10 | 38.1 % | 72.7 % |
| general_frequency | 9 | 25.0 % | 77.8 % |
| song_name | 8 | 0.0 % | 50.0 % |
| email_address | 7 | 0.0 % | 61.5 % |
| podcast_name | 7 | 0.0 % | 40.0 % |
| game_name | 5 | 18.2 % | 60.0 % |
| order_type | 5 | 0.0 % | 20.0 % |
| personal_info | 5 | 0.0 % | 25.0 % |
| audiobook_author | 4 | 66.7 % | 75.0 % |
| meal_type | 4 | 0.0 % | 61.5 % |
| playlist_name | 4 | 18.2 % | 0.0 % |
| cooking_type | 3 | 0.0 % | 12.5 % |
| joke_type | 3 | 0.0 % | 0.0 % |
| music_descriptor | 3 | 0.0 % | 66.7 % |
| coffee_type | 2 | 0.0 % | 0.0 % |
| email_folder | 2 | 50.0 % | 0.0 % |
| ingredient | 2 | 0.0 % | 22.2 % |
| movie_type | 2 | 0.0 % | 50.0 % |
| time_zone | 2 | 0.0 % | 40.0 % |
| change_amount | 1 | 0.0 % | 0.0 % |
| drink_type | 1 | 0.0 % | 50.0 % |
| movie_name | 1 | 0.0 % | 0.0 % |
| transport_agency | 1 | 0.0 % | 0.0 % |
| transport_name | 1 | 0.0 % | 13.3 % |
| alarm_type | 0 | 0.0 % | 0.0 % |
| app_name | 0 | 0.0 % | 0.0 % |
| music_album | 0 |  | 0.0 % |

## F1 per slot type, en

| slot type | gold | gliner2-large | gliner2.5-multi | winnow-12b |
|---|---|---|---|---|
| date | 137 | 73.5 % | 48.3 % | 46.7 % |
| place_name | 86 | 42.0 % | 0.0 % | 80.7 % |
| event_name | 84 | 53.1 % | 43.5 % | 47.2 % |
| person | 77 | 45.1 % | 4.8 % | 72.4 % |
| time | 64 | 38.5 % | 11.1 % | 39.1 % |
| media_type | 42 | 10.0 % | 0.0 % | 0.0 % |
| business_name | 27 | 43.8 % | 33.7 % | 59.3 % |
| weather_descriptor | 27 | 28.2 % | 11.8 % | 26.4 % |
| food_type | 25 | 54.2 % | 0.0 % | 64.3 % |
| house_place | 23 | 35.1 % | 31.3 % | 34.3 % |
| timeofday | 22 | 44.4 % | 0.0 % | 41.4 % |
| artist_name | 21 | 43.8 % | 45.9 % | 69.8 % |
| list_name | 20 | 4.8 % | 0.0 % | 12.5 % |
| relation | 20 | 27.6 % | 0.0 % | 51.9 % |
| transport_type | 20 | 35.3 % | 0.0 % | 66.7 % |
| device_type | 18 | 14.6 % | 29.4 % | 48.3 % |
| currency_name | 17 | 53.3 % | 57.8 % | 38.1 % |
| definition_word | 17 | 42.1 % | 60.0 % | 58.5 % |
| music_genre | 16 | 56.4 % | 54.5 % | 68.6 % |
| player_setting | 16 | 0.0 % | 0.0 % | 0.0 % |
| news_topic | 15 | 26.4 % | 10.8 % | 34.4 % |
| business_type | 14 | 13.3 % | 9.1 % | 45.5 % |
| radio_name | 14 | 6.1 % | 13.8 % | 32.0 % |
| audiobook_name | 11 | 20.0 % | 30.0 % | 84.2 % |
| color_type | 10 | 38.5 % | 48.0 % | 80.0 % |
| podcast_descriptor | 10 | 0.0 % | 0.0 % | 0.0 % |
| general_frequency | 9 | 26.1 % | 27.3 % | 72.7 % |
| song_name | 9 | 16.3 % | 0.0 % | 75.0 % |
| email_address | 7 | 0.0 % | 0.0 % | 71.4 % |
| podcast_name | 7 | 8.3 % | 18.2 % | 40.0 % |
| game_name | 5 | 46.2 % | 44.4 % | 46.2 % |
| order_type | 5 | 33.3 % | 0.0 % | 35.3 % |
| personal_info | 5 | 0.0 % | 20.0 % | 33.3 % |
| audiobook_author | 4 | 57.1 % | 50.0 % | 85.7 % |
| meal_type | 4 | 57.1 % | 40.0 % | 61.5 % |
| playlist_name | 4 | 42.9 % | 40.0 % | 26.7 % |
| cooking_type | 3 | 37.5 % | 25.0 % | 0.0 % |
| joke_type | 3 | 0.0 % | 0.0 % | 0.0 % |
| music_descriptor | 3 | 0.0 % | 0.0 % | 50.0 % |
| time_zone | 3 | 0.0 % | 0.0 % | 0.0 % |
| coffee_type | 2 | 0.0 % | 0.0 % | 0.0 % |
| email_folder | 2 | 80.0 % | 50.0 % | 50.0 % |
| ingredient | 2 | 28.6 % | 33.3 % | 22.2 % |
| movie_type | 2 | 50.0 % | 0.0 % | 40.0 % |
| change_amount | 1 | 0.0 % | 0.0 % | 0.0 % |
| drink_type | 1 | 40.0 % | 33.3 % | 40.0 % |
| movie_name | 1 | 9.5 % | 0.0 % | 66.7 % |
| transport_agency | 1 | 0.0 % | 0.0 % | 0.0 % |
| transport_name | 1 | 4.3 % | 0.0 % | 16.7 % |
| alarm_type | 0 | 0.0 % | 0.0 % | 0.0 % |
| app_name | 0 | 0.0 % | 0.0 % | 0.0 % |
| music_album | 0 | 0.0 % | 0.0 % | 0.0 % |
| transport_descriptor | 0 | 0.0 % |  |  |

## GLiNER threshold sensitivity

Same run, rescored from the stored candidates (call threshold 0.1). **0.5 is the primary result**: the library default, fixed before any run, and identical to the tables above. 0.3, 0.4, 0.6 are a sensitivity check read on the test data itself, not tuning; no threshold may be chosen from this table.

| lang | arm | threshold | role | predicted | P | R | F1 | exact frame |
|---|---|---|---|---|---|---|---|---|
| de | gliner2.5-multi | 0.3 | sensitivity (test data) | 768 | 25.9 % | 21.1 % | 23.3 % | 25.1 % |
| de | gliner2.5-multi | 0.4 | sensitivity (test data) | 702 | 27.1 % | 20.2 % | 23.1 % | 26.2 % |
| de | gliner2.5-multi | 0.5 | **primary** | 604 | 28.8 % | 18.5 % | 22.5 % | 27.5 % |
| de | gliner2.5-multi | 0.6 | sensitivity (test data) | 501 | 30.7 % | 16.4 % | 21.4 % | 28.8 % |
| en | gliner2-large | 0.3 | sensitivity (test data) | 1959 | 26.4 % | 55.2 % | 35.7 % | 14.0 % |
| en | gliner2-large | 0.4 | sensitivity (test data) | 1704 | 28.9 % | 52.5 % | 37.3 % | 18.3 % |
| en | gliner2-large | 0.5 | **primary** | 1491 | 31.0 % | 49.3 % | 38.1 % | 22.4 % |
| en | gliner2-large | 0.6 | sensitivity (test data) | 1274 | 34.1 % | 46.4 % | 39.3 % | 27.7 % |
| en | gliner2.5-multi | 0.3 | sensitivity (test data) | 879 | 24.0 % | 22.5 % | 23.2 % | 21.3 % |
| en | gliner2.5-multi | 0.4 | sensitivity (test data) | 775 | 26.2 % | 21.7 % | 23.7 % | 23.9 % |
| en | gliner2.5-multi | 0.5 | **primary** | 664 | 28.8 % | 20.4 % | 23.9 % | 25.6 % |
| en | gliner2.5-multi | 0.6 | sensitivity (test data) | 560 | 31.8 % | 19.0 % | 23.8 % | 27.6 % |

