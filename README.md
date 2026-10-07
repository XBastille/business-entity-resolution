# Business Entity Resolution

For every Source 1 business, find the Source 2 and Source 3 records that are the same business, or none.
This is the pipeline behind our final submission. On held-out businesses shaped like test it scores about 0.988
macro F0.5 in both training countries (US 0.98866, India 0.98810 on the businesses that fit the final combiner).
Earlier versions on the public board: 0.958 (first pipeline), 0.971 (test-like training).

## How it works

1. **Clean** names and addresses (`data/text.py`): lowercase, accents off, punctuation to spaces, house numbers
   without leading zeros ("0110" and "110" are the same number). Indian-script names are rewritten word by
   word with a dictionary learned from training pairs (`data/translit.py`).
2. **Shortlist** (`blocking/shortlist.py`): character-trigram TF-IDF on name and address, per country. Each
   Source 2/3 record keeps its 3 most similar Source 1 businesses (sparse_dot_topn, no full score table).
3. **Make training look like test** (`blocking/testlike.py`). Test has about 5.8 records per business against 4.7 in
   train, and the extra records belong to nobody. We delete Source 1 businesses from train until the ratio
   matches test in each country, so their copies become decoys the model has to turn down, as on test.
4. **Compare** (`features/base.py`, `families.py`, `sharp.py`): 78 numbers per pair, all computed without labels:
   shortlist scores, name and address similarities, house numbers (equal, off by a few, a digit changed),
   how the other records pointing at the same business look, word rarity and which words differ, legal forms,
   websites and scripts.
5. **Trim** (`decision/rules.py`): a small first model drops pairs below 1%, leaving about 6 candidates per business
   while keeping 99.7 to 99.9% of true pairs. This set is `candidate_pairs.tsv`.
6. **Score** (`core.py`): LightGBM, 255 leaves, early stopping on held-out businesses.
7. **Second opinion** (`models/pair_model.py`): xlm-roberta-large (MIT, 561M parameters), fine-tuned on 4 million
   "name | address" text pairs, re-judges the pairs the trees are between 5% and 95% sure about. A small
   logistic combiner mixes the two scores.
8. **Decide** (`decision/rules.py`): each record goes to at most one business; a business answers only when its best
   record is convincing, and keeps the records close to that best. On test the cut is set so each country
   accepts as many records per business as held-out did. Countries not in train (France) get the average
   settings.
9. **Write** both output files in the required format (`data/io.py`).

File names in the steps are under `src/entity_resolution/`.

## Layout

```
data/        the organisers' train/ and test/ folders go here (not in git)
outputs/     matching_results.tsv and candidate_pairs.tsv (not in git)
work/        stage caches, the LightGBM model and the pair model (not in git)
src/
├── run.py                   runs everything, raw TSVs to both output files
└── entity_resolution/
    ├── core.py              seed, labels, LightGBM training, macro F0.5
    ├── data/                reading and writing TSVs, cleaning, transliteration
    ├── blocking/            TF-IDF shortlist, test-like deletion
    ├── features/            the 78 comparison columns
    ├── models/              the xlm-roberta pair model
    └── decision/            first-model trim and the answer rule
```

## Run

```
pip install -r requirements.txt
python src/run.py --data-dir data --out-dir outputs --work-dir work --threads 32
```

The full dataset needs about 160 GB of RAM and, for the pair model, a GPU with 48 GB or more. On 32 cores and
an RTX PRO 6000 the whole run takes 4 to 5 hours, about half of it training the pair model. `--work-dir` keeps
each finished stage, so a stopped run picks up where it left off.
