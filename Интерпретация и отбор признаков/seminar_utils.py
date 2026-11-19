import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from catboost.utils import eval_metric
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42
THREADS = 4
SOURCE_REVISION = "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"
SOURCE_URL = f"https://raw.githubusercontent.com/wayfair/WANDS/{SOURCE_REVISION}"
METRIC = "NDCG:top=5;type=Exp;denominator=LogPosition"
FEATURES = [
    "title_coverage",
    "description_coverage",
    "attributes_coverage",
    "title_tfidf",
    "exact_phrase",
    "title_length",
    "average_rating",
    "has_rating",
    "log_rating_count",
    "log_review_count",
    "query_length",
    "random_noise",
]
NAMES = {
    "title_coverage": "Query-term coverage in title",
    "description_coverage": "Query-term coverage in description",
    "attributes_coverage": "Query-term coverage in attributes",
    "title_tfidf": "Query-title similarity (TF-IDF)",
    "exact_phrase": "Exact query phrase in title",
    "title_length": "Title length (words)",
    "average_rating": "Average rating (0 = missing)",
    "has_rating": "Rating available",
    "log_rating_count": "log(1 + rating count)",
    "log_review_count": "log(1 + review count)",
    "query_length": "Query length (words)",
    "random_noise": "Random control feature",
}
LABELS = {"Irrelevant": 0, "Partial": 1, "Exact": 2}


def tokens(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def load_wands(cache_dir=None):
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent / "data/wands"

    cache = Path(cache_dir)
    table_path = cache / "model_table.csv.gz"
    manifest_path = cache / "manifest.json"

    if table_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("format_version") != 1
            or manifest.get("source_revision") != SOURCE_REVISION
        ):
            raise ValueError("Unsupported WANDS cache version; choose a new cache_dir")

        digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
        if digest != manifest["table_sha256"]:
            raise ValueError("Checksum mismatch for model_table.csv.gz")

        return pd.read_csv(table_path), manifest

    cache.mkdir(parents=True, exist_ok=True)
    raw_hashes = {}

    def download(name):
        path = cache / name
        if not path.exists():
            directory = "dataset/" if name.endswith(".csv") else ""
            url = f"{SOURCE_URL}/{directory}{name}"
            print(f"Downloading {name} from WANDS ({SOURCE_REVISION[:8]})", flush=True)
            temporary = path.with_name(path.name + ".part")

            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=120) as response:
                        with temporary.open("wb") as target:
                            while chunk := response.read(1024 * 1024):
                                target.write(chunk)
                    break
                except urllib.error.HTTPError:
                    raise
                except (TimeoutError, urllib.error.URLError, OSError):
                    if attempt == 2:
                        raise
                    print(f"Retrying download of {name}: {attempt + 2}/3", flush=True)
                    time.sleep(2)

            temporary.replace(path)
            path.chmod(0o644)

        raw_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    queries = pd.read_csv(download("query.csv"), sep="\t")
    labels = pd.read_csv(download("label.csv"), sep="\t")
    products = pd.read_csv(download("product.csv"), sep="\t")
    download("LICENSE")

    assert queries.query_id.is_unique and products.product_id.is_unique
    assert set(labels.label) == set(LABELS)

    original_label_count = len(labels)
    keys = ["query_id", "product_id"]
    conflicts = labels.groupby(keys).label.transform("nunique").gt(1)
    conflicting_pairs = len(labels.loc[conflicts, keys].drop_duplicates())
    conflicting_rows = int(conflicts.sum())
    labels = labels.loc[~conflicts].drop_duplicates(keys).copy()
    identical_duplicates = original_label_count - conflicting_rows - len(labels)
    assert not labels.duplicated(keys).any()

    stats = labels.groupby("query_id").label.agg(["size", "nunique"])
    eligible = stats.index[(stats["size"] >= 10) & (stats["nunique"] >= 2)]
    assert 3 in eligible and len(eligible) >= 240

    other_ids = np.random.default_rng(SEED).permutation(eligible[eligible != 3])[:239]
    split_ids = {
        "train": other_ids[:144],
        "validation": np.r_[3, other_ids[144:191]],
        "test": other_ids[191:],
    }
    split_lookup = {
        int(query_id): split
        for split, query_ids in split_ids.items()
        for query_id in query_ids
    }
    frame = labels[labels.query_id.isin(split_lookup)].merge(
        queries, on="query_id", validate="many_to_one"
    )
    frame = frame.merge(products, on="product_id", validate="many_to_one")
    frame["split"] = frame.query_id.map(split_lookup)
    frame["relevance"] = frame.label.map(LABELS)
    frame = frame.sort_values(["query_id", "product_id"]).reset_index(drop=True)

    query_tokens = {
        int(row.query_id): tokens(row.query)
        for row in queries.itertuples()
    }
    product_tokens = {}
    for column in ["product_name", "product_description", "product_features"]:
        product_tokens[column] = {
            int(product_id): tokens(text)
            for product_id, text in zip(products.product_id, products[column].fillna(""))
        }

    query_terms = {
        query_id: set(words)
        for query_id, words in query_tokens.items()
    }
    coverage_columns = {
        "product_name": "title_coverage",
        "product_description": "description_coverage",
        "product_features": "attributes_coverage",
    }
    for column, feature in coverage_columns.items():
        product_terms = {
            product_id: set(words)
            for product_id, words in product_tokens[column].items()
        }
        frame[feature] = [
            len(query_terms[query_id] & product_terms[product_id])
            / max(len(query_terms[query_id]), 1)
            for query_id, product_id in zip(frame.query_id, frame.product_id)
        ]

    clean_titles = products.product_name.fillna("").astype(str)
    train_products = frame.loc[frame.split.eq("train"), "product_id"].unique()
    train_titles = products.loc[
        products.product_id.isin(train_products), "product_name"
    ].fillna("")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
    )
    vectorizer.fit(train_titles)

    title_vectors = vectorizer.transform(clean_titles)
    query_vectors = vectorizer.transform(queries["query"])
    product_indices = pd.Series(np.arange(len(products)), index=products.product_id)
    query_indices = pd.Series(np.arange(len(queries)), index=queries.query_id)
    row_titles = title_vectors[frame.product_id.map(product_indices)]
    row_queries = query_vectors[frame.query_id.map(query_indices)]
    similarities = row_titles.multiply(row_queries).sum(axis=1)
    frame["title_tfidf"] = np.asarray(similarities).ravel()

    normalized_queries = {
        query_id: " " + " ".join(words) + " "
        for query_id, words in query_tokens.items()
    }
    normalized_titles = {
        product_id: " " + " ".join(words) + " "
        for product_id, words in product_tokens["product_name"].items()
    }
    frame["exact_phrase"] = [
        int(normalized_queries[query_id] in normalized_titles[product_id])
        for query_id, product_id in zip(frame.query_id, frame.product_id)
    ]
    title_lengths = {
        product_id: len(words)
        for product_id, words in product_tokens["product_name"].items()
    }
    query_lengths = {
        query_id: len(words)
        for query_id, words in query_tokens.items()
    }
    frame["title_length"] = frame.product_id.map(title_lengths)
    frame["query_length"] = frame.query_id.map(query_lengths)
    frame["has_rating"] = frame.average_rating.notna().astype(int)
    frame["average_rating"] = frame.average_rating.fillna(0)

    for column in ["rating_count", "review_count"]:
        frame[f"log_{column}"] = np.log1p(frame[column].fillna(0).clip(lower=0))

    frame["random_noise"] = np.random.default_rng(SEED + 1).uniform(size=len(frame))
    metadata_columns = [
        "query_id",
        "product_id",
        "query",
        "product_name",
        "product_class",
        "label",
        "relevance",
        "split",
    ]
    frame = frame[metadata_columns + FEATURES]
    assert np.isfinite(frame[FEATURES].to_numpy()).all()
    frame.to_csv(table_path, index=False, compression={"method": "gzip", "mtime": 0})

    manifest = {
        "format_version": 1,
        "source": "https://github.com/wayfair/WANDS",
        "source_revision": SOURCE_REVISION,
        "seed": SEED,
        "features": FEATURES,
        "original_queries": len(queries),
        "original_products": len(products),
        "original_labels": original_label_count,
        "clean_labels": len(labels),
        "conflicting_pairs_removed": conflicting_pairs,
        "conflicting_rows_removed": conflicting_rows,
        "identical_duplicate_rows_removed": identical_duplicates,
        "eligible_queries": len(eligible),
        "selected_queries": frame.query_id.nunique(),
        "selected_rows": len(frame),
        "demo_query_id": 3,
        "split_queries": {
            split: len(query_ids)
            for split, query_ids in split_ids.items()
        },
        "split_rows": frame.split.value_counts().to_dict(),
        "label_mapping": LABELS,
        "raw_sha256": raw_hashes,
        "table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
        "selection": (
            "240 whole queries; >=10 labelled candidates and >=2 labels; "
            "query 3 reserved for validation"
        ),
        "tfidf": (
            "train product titles only; word unigrams+bigrams; min_df=2; "
            "max_features=20000; sublinear_tf=True"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for path in [cache, table_path, manifest_path]:
        path.chmod(0o755 if path.is_dir() else 0o644)

    return frame, manifest


def make_pool(frame, features=FEATURES):
    return Pool(frame[features], label=frame.relevance, group_id=frame.query_id)


def fit_ranker(train, valid, features=FEATURES):
    model = CatBoostRanker(
        loss_function="PairLogit:max_pairs=1000",
        eval_metric=METRIC,
        iterations=240,
        depth=5,
        learning_rate=0.08,
        l2_leaf_reg=5,
        random_seed=SEED,
        thread_count=THREADS,
        task_type="CPU",
        allow_writing_files=False,
    )
    model.fit(
        make_pool(train, features),
        eval_set=make_pool(valid, features),
        early_stopping_rounds=35,
        use_best_model=True,
        verbose=False,
    )
    return model


def predict(model, frame, features=FEATURES):
    return model.predict(frame[features], thread_count=THREADS)


def ndcg(frame, scores):
    scores = np.asarray(scores)
    assert scores.shape == (len(frame),) and np.isfinite(scores).all()
    return eval_metric(
        frame.relevance.to_numpy(),
        scores,
        METRIC,
        group_id=frame.query_id.to_numpy(),
        thread_count=THREADS,
    )[0]


def sample_queries(frame, per_query=12):
    rng = np.random.default_rng(SEED)
    indices = [
        rng.choice(group.index, min(per_query, len(group)), replace=False)
        for _, group in frame.groupby("query_id", sort=True)
    ]
    sampled = frame.loc[np.concatenate(indices)]
    return sampled.sort_values(["query_id", "product_id"]).reset_index(drop=True)


def cards(frame, scores=None, limit=6):
    view = frame.copy()
    if scores is not None:
        view["score"] = scores
        view = view.sort_values("score", ascending=False, kind="stable")

    columns = [
        "product_id",
        "product_name",
        "label",
        "title_coverage",
        "title_tfidf",
        "average_rating",
    ]
    if scores is not None:
        columns.append("score")

    names = {
        "product_name": "Product",
        "label": "Relevance",
        "title_coverage": "Query-term coverage",
        "title_tfidf": "Title similarity",
        "average_rating": "Rating",
        "score": "Score",
    }
    return view[columns].head(limit).rename(columns=names).round(3)


def lime_coordinates(explainer, values, original):
    values = np.asarray(values, dtype=float)
    coordinates = (values - explainer.scaler.mean_) / explainer.scaler.scale_
    for index in explainer.categorical_features:
        coordinates[:, index] = (values[:, index] == original[index]).astype(float)
    return coordinates


def lime_prediction(explanation, explainer, values, original):
    coefficients = np.zeros(len(FEATURES))
    for index, weight in explanation.local_exp[1]:
        coefficients[index] = weight

    coordinates = lime_coordinates(explainer, values, original)
    return explanation.intercept[1] + coordinates @ coefficients


if __name__ == "__main__":
    frame, manifest = load_wands()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    train_ids = set(frame.loc[frame.split.eq("train"), "query_id"])
    valid_ids = set(frame.loc[frame.split.eq("validation"), "query_id"])
    test_ids = set(frame.loc[frame.split.eq("test"), "query_id"])
    assert train_ids.isdisjoint(valid_ids | test_ids) and valid_ids.isdisjoint(test_ids)
    assert frame.groupby("query_id").split.nunique().eq(1).all()

    check = pd.DataFrame({"query_id": [0, 0, 1, 1], "relevance": [0, 2, 1, 0]})
    assert np.isclose(ndcg(check, [0.0, 2.0, 1.0, 0.0]), 1.0)