# 📋 Sample Predicted Matches Summary (Test Set)
### Amazon ML Challenge 2026: Multi-Source Business Entity Resolution

This document illustrates the actual business entities resolved and matched by the pipeline from `outputs/matching_results.tsv`.

## 1. Sample Multi-Source and Single-Source Matches

| Source 1 ID | Business Name | Address | Country | Matched Entity IDs | Match Category |
|---|---|---|---|---|---|
| `S1-714132312` | Zephay Labs Inc | 2621 Cotten Road, Tyler, TX | `US` | `S3-625880872` | **Source 3 Match (n=1)** |
| `S1-106407869` | Vision Partners Corp | IA, Iowa City, 1064 Newton Rd, Unit | `US` | `S2-705547832,S3-585937637` | **Multi-Source Match (S2 & S3, n=2)** |
| `S1-156285671` | << Team Ecole | 175 Boulevard du Président Franklin | `France` | `S2-364557824,S3-672973403,S3-738587982` | **Multi-Source Match (S2 & S3, n=3)** |
| `S1-921369899` | ZNB Club SARL | Nouvelle-Aquitaine, La Teste-de-Buc | `France` | `S3-57594330,S2-107533354,S2-808920974...` | **Multi-Source Match (S2 & S3, n=5)** |
| `S1-909865979` | Cure Seafood | 1325 Brooklyn Walk, Issaquah, WA | `US` | `S2-528901108` | **Source 2 Match (n=1)** |
| `S1-742053041` | Roongta Sangh | Bhubaneswar, Sub Plot No.-L6/29, Ma | `India` | `S2-317425677,S2-73207920` | **Source 2 Match (n=2)** |
| `S1-479574617` | Anupama Estates Pvt Ltd | 26, Ram Nagar, Behind M.C Quarters  | `India` | `S2-262582939,S2-985539429` | **Source 2 Match (n=2)** |
| `S1-680637447` | Siliguri Media Pvt Ltd | S No. 42/2/3, H No. B/5, Santkrupa  | `India` | `S3-261988229` | **Source 3 Match (n=1)** |
| `S1-365170694` | Consulting Sai Nanak Private L | A-115, Freedom Fighter, Enclave Neb | `India` | `S2-487943573` | **Source 2 Match (n=1)** |
| `S1-778640743` | Blinny Gordon Apex Rate | 100 Hartsdale Avenue, Unit Apartmen | `US` | `S3-698916948,S2-87750197` | **Multi-Source Match (S2 & S3, n=2)** |
| `S1-684974736` | Garcia Mercury Corp | 600 SE 2nd St, Pelican Rapids, MN | `US` | `S3-894822385` | **Source 3 Match (n=1)** |
| `S1-209281220` | ZH Plus Private Limited | 37B, Pushtikar Chs Ltd, Shiv Sadan, | `India` | `S2-873605695` | **Source 2 Match (n=1)** |
| `S1-550154910` | Tech Management Private Limite | H No. 16-2-751/A/70, Kara, Chanchal | `India` | `S2-43467331` | **Source 2 Match (n=1)** |
| `S1-704604312` | Market Sciences (India) Ltd | A-15 Phase-10, Shiv Vihar, Karawal  | `India` | `S3-7278253` | **Source 3 Match (n=1)** |
| `S1-628750886` | Grain & Fils | Lille, 329 Avenue de Dunkerque, Hau | `France` | `S2-406813553,S2-696378102,S3-73499285...` | **Multi-Source Match (S2 & S3, n=4)** |
| `S1-714038427` | Delta Partners Inc | 2202 Brandywood Drive, Murfreesboro | `US` | `S3-343322433,S3-556092751,S2-9389075` | **Multi-Source Match (S2 & S3, n=3)** |
| `S1-426194890` | Midwest Alliance LLC | 1045 Lazy Acres Road, Protem, MO | `US` | `S2-12575838` | **Source 2 Match (n=1)** |
| `S1-123475953` | Sankalp India | New Delhi, South Delhi, Ii Nd Floor | `India` | `S3-669730032,S3-577626649,S2-80336498...` | **Multi-Source Match (S2 & S3, n=4)** |
| `S1-502376041` | Chayan Trading Private Limited | Flat No 404, Block-1, Rbr Complex,  | `India` | `S2-408714071,S2-551281247,S2-54256234` | **Source 2 Match (n=3)** |
| `S1-594021398` | Abohar Logistics Pvt Ltd | C/O Manoj Kumar St No. 1, Model Tow | `India` | `S3-133279691,S3-686916460` | **Source 3 Match (n=2)** |

## 2. Sample Singletons (Zero True Matches Detected)
Singletons are strictly formatted with an empty string `""` in accordance with the official validator:

| Source 1 ID | Business Name | Address | Country | Matched Entity IDs | Match Category |
|---|---|---|---|---|---|
| `S1-689823050` | Red Perfect Trading | Mirzapur, Ews 12, Uttar Pradesh, Mi | `India` | `""` (Empty) | *Singleton (No Match)* |
| `S1-481669221` | Nandlal Kisan LLP | D-61 Ifs Apartmentmayur Vihar I, Ne | `India` | `""` (Empty) | *Singleton (No Match)* |
| `S1-280204013` | Om Constructions Pvt Ltd | Karauli, Rajasthan, Karauli, Pani K | `India` | `""` (Empty) | *Singleton (No Match)* |
| `S1-913506265` | Thermal & Fils SASU | 20 Rue Parmentier, Dunkerque, Hauts | `France` | `""` (Empty) | *Singleton (No Match)* |
| `S1-870190130` | Naman Trust | 60 Kashipuri Kabirkhedi, Indore, Ma | `India` | `""` (Empty) | *Singleton (No Match)* |
| `S1-378191895` | East Marketing Private Limited | Howrah, 2, West Bengal, India Excha | `India` | `""` (Empty) | *Singleton (No Match)* |
| `S1-750279127` | Sra Export | A-73, Lajpat Nagar-I, New Delhi, So | `India` | `""` (Empty) | *Singleton (No Match)* |
| `S1-897975128` | Prime Aditya Agro | 1-76, Brahmanapalli Village, Andole | `India` | `""` (Empty) | *Singleton (No Match)* |
| `S1-122385784` | Castaneda Capital LLC | 105 Paula Boulevard, Brookhaven, NY | `US` | `""` (Empty) | *Singleton (No Match)* |
| `S1-540642140` | Tri-State Network | 3105 Stonegate Drive, Paragould, AR | `US` | `""` (Empty) | *Singleton (No Match)* |
