# EuroSAT Glavni Izveštaj o Projektu (Izvorni Dokument za Tezu)

## 1. Uloga Dokumenta i Pravila Korišćenja

Ovaj fajl nije finalni naučni rad. On je kompletan izvorni dokument za pisanje IEEE rada bez ponovnog otvaranja celog koda projekta.

Primarna namena:
- Drži sve činjenice o projektu, detalje protokola, napomene o implementaciji i verifikovane metrike na jednom mestu.
- Razdvaja verifikovane činjenice od predloženih dopuna.
- Obezbeđuje dovoljno konteksta za pisanje uvoda, pregleda literature, metodologije, rezultata i diskusije direktno iz ovog fajla.

Opseg ove verzije:
- Uključene familije modela: baseline CNN, EfficientNetB0 (Miloš opseg) i ResNet50 (Bojan opseg).
- Uključena vrsta dokaza: metapodaci izvršavanja, protokol selekcije, ablation poređenja, poređenje modela, inženjerske napomene i plan pisanja.

Pravilo za pisanje teze:
- Svaka numerička tvrdnja u finalnom radu mora biti zasnovana na ovom izveštaju.
- Ako tvrdnja nije u ovom izveštaju, prvo je dodati ovde.

## 2. Sažetak Projekta za Uvod

### 2.1 Definicija Problema

Projekat rešava 10-klasnu klasifikaciju namene zemljišta sa satelitskih snimaka korišćenjem EuroSAT skupa podataka.

Ulazno-izlazna formulacija:
- Ulaz: RGB satelitski isečak (64x64).
- Izlaz: jedna oznaka iz 10 klasa namene zemljišta.

### 2.2 Zašto je Problem Važan

Klasifikacija namene zemljišta je važna za:
- urbano planiranje i praćenje infrastrukture,
- poljoprivrednu analitiku,
- praćenje šuma i vodenih površina,
- donošenje odluka vezanih za životnu sredinu i klimu.

Automatska klasifikacija poboljšava brzinu i konzistentnost u odnosu na ručnu analizu i omogućava skalabilne sisteme monitoringa.

### 2.3 Cilj Projekta

Procena da li transfer learning sa modernim unapred treniranim CNN backbone modelima poboljšava performanse u odnosu na prilagođeni baseline CNN uz strogo i fer definisan protokol.

Istraživačka osa poređenja:
- Baseline CNN (trening od nule) naspram EfficientNetB0 i ResNet50 (dvostepeni transfer learning).

### 2.4 Odgovornosti Tima

- Miloš Milanović: EfficientNetB0 i zajednička infrastruktura.
- Bojan Živanić: ResNet50.
- Zajedničke odgovornosti: baseline, dizajn protokola, konvencije izveštavanja, podešavanje reproduktivnosti.

## 3. Istraživačka Pitanja i Hipoteze

### 3.1 Istraživačka Pitanja

RQ1: Da li transfer-learning modeli nadmašuju baseline CNN na EuroSAT skupu pod identičnim pravilima podele i selekcije?

RQ2: Koliko dvostepeni fine-tuning (zamrznut pa odmrznut) doprinosi u odnosu na samo zamrznuti transfer setup?

RQ3: Koliko su familije modela osetljive na augmentaciju i varijante stope učenja?

RQ4: Koji model daje najbolji kompromis između finalnih test performansi i brzine konvergencije?

### 3.2 Radne Hipoteze

H1: EfficientNetB0 i ResNet50 će nadmašiti baseline CNN po holdout test accuracy i macro F1.

H2: Fine-tuning u Fazi 2 će nadmašiti setup sa zamrznutim backbone-om iz Faze 1 kod oba transfer modela.

H3: Uklanjanje augmentacije će degradirati macro F1 kod svih familija modela.

## 4. Skup Podataka i Skup Klasa

Skup podataka:
- EuroSAT RGB verzija (10 klasa, ~27.000 slika).

Klase:
- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

Pretpostavke o skupu podataka u ovom projektu:
- Oznake se izvode iz folder strukture i eksportovanih split manifesta.
- Nije bila potrebna dodatna ručna anotacija.

Primarne lokalne lokacije podataka:
- `data/EuroSAT/`
- `data/EuroSAT/train.csv`
- `data/EuroSAT/validation.csv`
- `data/EuroSAT/test.csv`

## 5. Eksperimentalni Protokol (Kanonski)

### 5.1 Politika Podele i Reproduktivnosti

- Strategija: stratifikovana podela po klasama.
- Odnosi: 70% train / 15% validation / 15% test.
- Fiksirani seed: 42.
- Split artefakti se čuvaju i ponovo koriste kroz sve familije modela.

Obavezni split artefakti:
- `artifacts/splits/split_manifest.json`
- `artifacts/splits/train_split.json`
- `artifacts/splits/validation_split.json`
- `artifacts/splits/test_split.json`
- `artifacts/splits/split_summary.json`

### 5.2 Pravila Selekcije i Holdout Evaluacije

Anti-leakage pravilo:
- Selekcija modela radi se isključivo po validacionoj metrici (`val_f1_best`).
- Test skup koristi se samo jednom, za finalno holdout izveštavanje selektovanih run-ova.

Skupovi kandidata za selekciju:
- Baseline CNN: sve baseline varijante.
- EfficientNetB0: samo varijante Faze 2.
- ResNet50: samo varijante Faze 2.

### 5.3 Obavezne Metrike u Izveštaju

- Test accuracy.
- Test macro F1.
- Precision po klasama.
- Recall po klasama.
- Matrica konfuzije.

## 6. Sažetak Metodologije i Implementacije

### 6.1 Arhitektura i Organizacija Koda

Repozitorijum prati separaciju u stilu Clean Architecture:
- Domain sloj: entiteti i metric kontrakti.
- Application sloj: use case logika i orkestracija.
- Infrastructure sloj: učitavanje konfiguracije, dataset pristup, povezivanje modela, logovanje, checkpointing.
- Entrypoints: CLI tok izvršavanja.

Ključne lokacije:
- `src/eurosat_classifier/domain/`
- `src/eurosat_classifier/application/`
- `src/eurosat_classifier/infrastructure/`
- `src/eurosat_classifier/entrypoints/`

### 6.2 Familije Modela i Strategija Treninga

Baseline CNN:
- Jednostepeni trening od nule.
- Služi kao referentni komparator.

EfficientNetB0 i ResNet50:
- Faza 1: transfer setup sa zamrznutim backbone-om.
- Faza 2: odmrznuti fine-tuning nastavljen iz checkpoint-a Faze 1.
- Ista split politika i schema izveštavanja kao kod baseline modela.

### 6.3 Stil Konfiguracije i Izvršavanja

Konfiguracija kao osnova workflow-a:
- JSON konfiguracije definišu model, trening i augmentacione postavke.
- CLI podržava override parametara za kontrolisane eksperimente.

Aktivni config fajlovi:
- `configs/baseline_cnn.json`
- `configs/efficientnet_b0.stage1.json`
- `configs/efficientnet_b0.stage2.json`
- `configs/resnet50.stage1.json`
- `configs/resnet50.stage2.json`
- `configs/experiment.defaults.json`

### 6.4 Operativni Pipeline (Visoki Nivo)

1. Pripremiti ili ponovo iskoristiti determinističke split artefakte.
2. Trenirati kandidatske run-ove po familiji modela.
3. Izabrati pobednika po validation macro F1.
4. Eksportovati holdout metrike i artefakte za selektovani run.
5. Agregirati tabelu trostranog poređenja.

### 6.5 Inženjerska Ograničenja i Napomene

Uočeno u Kaggle snapshot-ovima:
- `pip install -e .` može prijaviti upozorenja resolver-a zbog preinstaliranih paketa.
- PyTorch može emitovati CuBLAS upozorenje o determinističnosti ako `CUBLAS_WORKSPACE_CONFIG` nije postavljen.

Uticaj u trenutnim run-ovima:
- Nije bilo fatalnih runtime izuzetaka.
- Zaključci o rangiranju ostali su stabilni u ovim snapshot-ovima.

## 7. Napomene za Pregled Literature (Ulaz za Rad)

Ova sekcija je zamišljena kao gotov izvor za IEEE poglavlje pregleda literature.

### 7.1 Minimalne Kategorije Literature

Uključiti bar jedan reprezentativni rad po kategoriji:
- Benchmark rad za EuroSAT dataset.
- Radove o klasifikaciji namene zemljišta u daljinskoj detekciji korišćenjem CNN modela.
- Studije transfer learning pristupa u daljinskoj detekciji.
- Osnovne radove za backbone arhitekture (ResNet i EfficientNet originalni radovi).

### 7.2 Predložene Referentne Stavke

- Helber et al., EuroSAT benchmark dataset (osnovna citacija skupa podataka).
- He et al., ResNet arhitektura.
- Tan i Le, EfficientNet arhitektura.

### 7.3 Kako Pozicionirati Ovaj Projekat u Odnosu na Prethodne Radove

Preporučeni narativ poređenja:
- Projekat stavlja fokus na strogu simetriju protokola (isti split manifest, isto pravilo selekcije).
- Upoređuje baseline i transfer pristupe pod istim evaluacionim ugovorom.
- Ističe reproduktivnost i anti-leakage selekciju modela kao metodološki doprinos.

### 7.4 Kontrolna Lista za Pisanje Pregleda Literature

Pre finalne predaje rada, obezbediti da svaki citirani rad ima:
- potpune bibliografske metapodatke,
- jasno naveden zadatak i dataset,
- eksplicitnu relevantnost za ovaj projekat,
- sažeto poređenje sa vašim setup-om.

## 8. Izvori Rezultata i Metapodaci Izvršavanja

Primarni snapshot-ovi rezultata korišćeni u ovom izveštaju:
- `results/eurosat-baseline.ipynb` (Baseline CNN)
- `results/eurosat-efficientnet.ipynb` (EfficientNetB0)
- `results/eurosat-resnet.ipynb` (ResNet50 run-ovi i export-i)

Papermill start timestamp-i:
- Baseline notebook: 2026-04-10T17:51:31
- EfficientNet notebook: 2026-04-10T17:51:51
- ResNet notebook: 2026-04-10T18:45:12

Približno vreme izvršavanja iz notebook metapodataka:
- Baseline: ~97.93 minuta
- EfficientNet: ~82.38 minuta
- ResNet50: ~58.67 minuta

Status izvršavanja:
- Sva tri notebook pipeline-a završena su bez fatalnih runtime izuzetaka.

## 9. Baseline CNN Rezultati (Zajednički Baseline)

### 9.1 Baseline Ablation Tabela

| Run ID | Augmentacija | Learning Rate | Broj epoha | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Selektovan za finale |
|---|---|---:|---:|---:|---:|---:|---|
| baseline_reference_none | none | 0.00100 | 50 | 0.915363 | 0.912346 | 0.908949 | no |
| baseline_flips | flips | 0.00100 | 50 | 0.951813 | 0.947901 | 0.946238 | no |
| baseline_flips_low_lr | flips | 0.00050 | 50 | 0.959781 | 0.961975 | 0.960863 | yes |

### 9.2 Ishod Selekcije Baseline Modela

- Skup kandidata: svi baseline run-ovi.
- Metod selekcije: `val_f1_best`.
- Izabrani run: `baseline_flips_low_lr`.

Ključni artefakti:
- `/kaggle/working/artifacts/reports/baseline/baseline_experiment_summary.csv`
- `/kaggle/working/artifacts/reports/baseline/baseline_holdout_report.csv`
- `/kaggle/working/artifacts/reports/baseline/baseline_model_selection.json`
- `/kaggle/working/artifacts/reports/baseline/baseline_flips_low_lr.json`
- `/kaggle/working/checkpoints/baseline_cnn/baseline_flips_low_lr/best_checkpoint.pt`

### 9.3 Tumačenje Baseline Rezultata

- Augmentacija (`flips`) daje veliko poboljšanje u odnosu na referencu bez augmentacije.
- Niža stopa učenja daje dodatni dobitak preko augmentovanog podešavanja.
- Macro F1 dobitak (`reference_none` -> `flips`): +0.037289.
- Macro F1 dobitak (`flips` -> `flips_low_lr`): +0.014625.

## 10. EfficientNetB0 Rezultati (Miloš Opseg)

### 10.1 EfficientNetB0 Ablation Tabela

| Run ID | Faza | Augmentacija | Learning Rate | Broj epoha | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Selektovan za finale |
|---|---|---|---:|---:|---:|---:|---:|---|
| efficientnet_stage1_reference | stage1 | flips | 0.00100 | 9 | 0.920765 | 0.926173 | 0.923145 | no |
| efficientnet_stage2_reference | stage2 | flips | 0.00010 | 24 | 0.978875 | 0.974321 | 0.973621 | yes |
| efficientnet_stage2_low_lr | stage2 | flips | 0.00005 | 24 | 0.975499 | 0.971358 | 0.970286 | no |
| efficientnet_stage2_no_aug | stage2 | none | 0.00010 | 13 | 0.965855 | 0.966914 | 0.965938 | no |

### 10.2 Ishod Selekcije EfficientNetB0 Modela

- Skup kandidata: samo run-ovi Faze 2.
- Metod selekcije: `val_f1_best`.
- Izabrani run: `efficientnet_stage2_reference`.

Ključni artefakti:
- `/kaggle/working/artifacts/reports/efficientnet/efficientnet_experiment_summary.csv`
- `/kaggle/working/artifacts/reports/efficientnet/efficientnet_holdout_report.csv`
- `/kaggle/working/artifacts/reports/efficientnet/efficientnet_model_selection.json`
- `/kaggle/working/artifacts/reports/efficientnet/efficientnet_stage2_reference.json`
- `/kaggle/working/checkpoints/efficientnet_b0/efficientnet_stage2_reference/best_checkpoint.pt`

### 10.3 Tumačenje EfficientNetB0 Rezultata

- Fine-tuning u Fazi 2 jasno poboljšava rezultate u odnosu na Fazu 1.
- Macro F1 dobitak (`stage1_reference` -> `stage2_reference`): +0.050476.
- U ablation poređenju Faze 2, referentno podešavanje ostaje najbolje po validation macro F1.
- `stage2_low_lr` je niži od reference za -0.003335 macro F1.
- `stage2_no_aug` je niži od reference za -0.007683 macro F1.

## 11. ResNet50 Rezultati (Bojan Opseg)

### 11.1 Metapodaci ResNet50 Run-a

- Izvor snapshot-a: `results/eurosat-resnet.ipynb`.
- Timestamp izvršavanja: 2026-04-10T18:45:12.
- Split seed: 42.
- Lokacija split manifesta u output-u run-a: `results/resnet/kaggle/working/artifacts/splits/`.
- Stage protokol: dvostepeni fine-tuning (zamrznuto -> odmrznuto).

### 11.2 ResNet50 Ablation Tabela

| Run ID | Faza | Augmentacija | Learning Rate | Broj epoha | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Selektovan za finale |
|---|---|---|---:|---:|---:|---:|---:|---|
| resnet50_stage1_reference | stage1 | flips | 0.00100 | 5 | 0.929148 | 0.927901 | 0.925859 | no |
| resnet50_stage2_reference | stage2 | flips | 0.00010 | 9 | 0.978449 | 0.978765 | 0.978468 | yes |
| resnet50_stage2_low_lr | stage2 | flips | 0.00005 | 11 | 0.978224 | 0.979506 | 0.979078 | no |
| resnet50_stage2_no_aug | stage2 | none | 0.00010 | 5 | 0.967177 | 0.964691 | 0.963568 | no |

### 11.3 Ishod Selekcije ResNet50 Modela

- Skup kandidata: samo run-ovi Faze 2.
- Metod selekcije: `val_f1_best`.
- Izabrani run: `resnet50_stage2_reference`.

Ključni artefakti:
- `/kaggle/working/artifacts/reports/resnet50/resnet50_experiment_summary.csv`
- `/kaggle/working/artifacts/reports/resnet50/resnet50_holdout_report.csv`
- `/kaggle/working/artifacts/reports/resnet50/resnet50_model_selection.json`
- `/kaggle/working/artifacts/reports/resnet50/resnet50_stage2_reference.json`
- `/kaggle/working/checkpoints/resnet50/resnet50_stage2_reference/best_checkpoint.pt`

### 11.4 Tumačenje ResNet50 Rezultata

- Fine-tuning u Fazi 2 daje jako poboljšanje u odnosu na Fazu 1.
- Macro F1 dobitak (`stage1_reference` -> `stage2_reference`): +0.052609.
- `stage2_low_lr` ima blago veći test macro F1 (+0.000610), ali niži validation macro F1 (-0.000225) od reference.
- `stage2_no_aug` pada za -0.014900 macro F1 u odnosu na `stage2_reference`.
- Brzina konvergencije je najveća među poređenim familijama modela.

## 12. Trostrano Poređenje Modela

### 12.1 Sažetak Performansi Selektovanih Run-ova

| Familija modela | Izabrani Run ID | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Faza 2 / Ukupno epoha |
|---|---|---:|---:|---:|---:|
| baseline_cnn | baseline_flips_low_lr | 0.959781 | 0.961975 | 0.960863 | 50 |
| efficientnet_b0 | efficientnet_stage2_reference | 0.978875 | 0.974321 | 0.973621 | 24 (33 uključujući Fazu 1) |
| resnet50 | resnet50_stage2_reference | 0.978449 | 0.978765 | 0.978468 | 9 (14 uključujući Fazu 1) |

### 12.2 Pairwise Razlike (Selektovani Run-ovi)

EfficientNetB0 vs Baseline:
- Accuracy: +0.012346.
- Macro F1: +0.012758.

ResNet50 vs Baseline:
- Accuracy: +0.016790.
- Macro F1: +0.017605.

ResNet50 vs EfficientNetB0:
- Accuracy: +0.004444.
- Macro F1: +0.004847.

### 12.3 Analiza Stabilnosti (Validation vs Test)

Razlika validation-test macro F1 za selektovane run-ove:
- Baseline: +0.001082.
- EfficientNetB0: -0.005254.
- ResNet50: +0.000019.

Tumačenje:
- ResNet50 pokazuje najjaču stabilnost između najboljeg validation rezultata i test holdout-a.
- EfficientNetB0 i dalje pokazuje vrlo jake performanse, ali sa većim očekivanim padom generalizacije.

### 12.4 Obrazac Težine po Klasama

Konzistentan obrazac kroz familije modela:
- Generalno jače klase: Forest, Industrial, Residential.
- Ponavljano teže klase: PermanentCrop, River, Highway.

Radno objašnjenje za diskusiju:
- Sličnost spektralnih/teksturnih karakteristika i granična dvosmislenost između nekih poljoprivrednih i koridornih klasa verovatno podižu konfuziju.

## 13. Analiza Grešaka i Input za Diskusiju

Koristiti sledeće tačke pri pisanju diskusije:

1. Dobici transfer learning pristupa su značajni i konzistentni kod oba napredna backbone modela.
2. Dvostepeni fine-tuning je neophodan; setup sa samo zamrznutim backbone-om je jasno slabiji.
3. Augmentacija ostaje kritična čak i za transfer modele.
4. ResNet50 je postigao najbolje holdout metrike i najbržu konvergenciju u ovom skupu eksperimenata.
5. Validation-only selekcija sprečila je test leakage tokom izbora modela.

Granice diskusije:
- Izbegavati tvrdnje o univerzalnoj superiornosti van EuroSAT skupa i trenutnog seed-a, osim ako se doda multi-seed evidencija.
- Sve tvrdnje držati striktno vezane za merene veličine iz ovog izveštaja.

## 14. Pretnje Validnosti i Reproduktivnosti

Rizici interne validnosti:
- Primarno poređenje sa jednim seed-om (42) može potceniti varijansu među run-ovima.
- Razlike u notebook okruženju mogu uticati na tačnu determinističnost.

Rizici eksterne validnosti:
- Rezultati su specifični za dataset (EuroSAT RGB, 64x64).
- Prenosivost na druge skupove podataka iz daljinske detekcije još nije merena.

Mitigacije koje već postoje:
- Fiksirana stratifikovana podela i sačuvani manifesti.
- Uniformna politika selekcije i holdout izveštavanja kroz sve familije modela.
- Eksplicitni artefakti i checkpoint-i za svaki run.

## 15. IEEE Plan Rada (Mapiranje Profesorovih Zahteva)

Profesorom zahtevana struktura i šta preuzeti iz ovog izveštaja:

### 15.1 Uvod i Motivacija

Izvorne sekcije u ovom izveštaju:
- Sekcija 2 (problem, značaj, cilj).
- Sekcija 3 (istraživačka pitanja i hipoteze).

### 15.2 Pregled Relevantne Literature

Izvorne sekcije u ovom izveštaju:
- Sekcija 7 (kategorije, referentni radovi, pozicioniranje).

### 15.3 Metodologija i Implementacija

Izvorne sekcije u ovom izveštaju:
- Sekcija 4 (dataset).
- Sekcija 5 (protokol).
- Sekcija 6 (arhitektura i pipeline).

### 15.4 Rezultati i Diskusija

Izvorne sekcije u ovom izveštaju:
- Sekcije 8-13 (metapodaci, ablation, poređenja, tumačenje, analiza grešaka).

### 15.5 Zaključak i Budući Rad

Izvorne sekcije u ovom izveštaju:
- Sekcija 13 (glavni zaključci).
- Sekcija 16 (checklist budućeg rada).

## 16. Stanford Smernice za Pisanje (Praktične Beleške)

Na osnovu teksta Jennifer Widom, "Tips for Writing Technical Papers":

1. Uvod treba eksplicitno da odgovori na pet pitanja:
- Šta je problem?
- Zašto je važan?
- Zašto je težak?
- Zašto postojeći pristupi nisu dovoljni?
- Koji su ključni doprinosi i ograničenja?

2. Dodati sažetu listu "Summary of Contributions" na kraju uvoda.

3. Zadržati linearnu priču kroz telo rada:
- preliminaries -> metod -> eksperimenti -> interpretacija.

4. Eksperimenti treba da pokažu:
- apsolutne performanse,
- relativne performanse u odnosu na baseline,
- osetljivost na glavne parametre.

5. Zaključak treba da sumira nalaze bez doslovnog ponavljanja teksta iz abstrakta.

6. Budući rad treba da bude konkretan i dat kroz bullet listu.

7. Pravila kvaliteta pisanja:
- definisati terminologiju pre upotrebe,
- izbegavati neodređene tvrdnje,
- osigurati konzistentnost i kompletnost citata.

## 17. Budući Rad i Završna Kontrolna Lista

Preporučene dopune dokaza pre zaključavanja teze:
- Pokrenuti multi-seed evaluaciju (na primer 42, 43, 44) za selektovane run-ove.
- Prijaviti srednju vrednost i standardnu devijaciju za accuracy i macro F1.
- Dodati figure matrice konfuzije za selektovane run-ove u paper-ready formatu.
- Zamrznuti detalje okruženja i verzije zavisnosti za appendix.
- Proširiti pregled literature punim bibliografskim metapodacima.

Završna pre-submission kontrolna lista:
- Potvrditi da sve tabele koriste isti protokol i seed politiku.
- Potvrditi da su svi izabrani run-ovi selektovani na validaciji, a prijavljeni na testu.
- Obezbediti konzistentno imenovanje result snapshot fajlova i putanja artefakata.
- Obezbediti da svaka numerička tvrdnja u radu postoji u ovom izveštaju.

## 18. Appendix: Brzi Indeks Artefakata

Core fajlovi za izveštavanje:
- `docs/evaluation_protocol.md`
- `docs/experiments_log.md`
- `results/eurosat-baseline.ipynb`
- `results/eurosat-efficientnet.ipynb`
- `results/eurosat-resnet.ipynb`

Split i reproduktivni artefakti:
- `artifacts/splits/split_manifest.json`
- `artifacts/splits/train_split.json`
- `artifacts/splits/validation_split.json`
- `artifacts/splits/test_split.json`

Notebook šabloni za execution flow:
- `notebooks/eurosat_baseline_kaggle.ipynb`
- `notebooks/eurosat_efficientnet_kaggle.ipynb`
- `notebooks/eurosat_resnet50_kaggle.ipynb`
