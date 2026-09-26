# timepass · Amazon ML Challenge 2026, business entity resolution

For every Source 1 business, find the Source 2 and Source 3 records that are the same business, or none.
This is the pipeline behind our first real submission: public score 0.958.

## How it works

1. **Clean** names and addresses (`text.py`): lowercase, accents off, punctuation to spaces, house numbers
   without leading zeros ("0110" and "110" are the same number). Indian-script names are rewritten word by
   word with a dictionary learned from training pairs (`translit.py`).
2. **Shortlist** (`shortlist.py`): character-trigram TF-IDF on name and address, per country. Each
   Source 2/3 record keeps its 3 most similar Source 1 businesses (sparse_dot_topn, no full score table).
3. **Compare** (`features.py`): 24 numbers per pair: shortlist scores and ranks, several name and address
   similarities (RapidFuzz), house-number agreement, and a few flags.
4. **Score** (`pipeline.py`): LightGBM on US and India training pairs. 20% of training businesses are held
   out to pick the acceptance cut per country; countries not in train (France) get the average cut.
5. **Decide**: each record goes to at most one business, its best-scoring one, and only above the cut.
6. **Write** both output files in the required format (`files.py`).

## Run

```
pip install -r requirements.txt
cd src
python run.py --data-dir <folder with train/ and test/> --out-dir ../output --threads 16
```

The full dataset needs a large machine (about 64 GB of RAM); the search step takes about 2 hours of
16 cores for test. Held-out scores at full size: US 0.963, India 0.958.

## Next

New comparison features, training on test-like data and a counted acceptance cut are on the way.
