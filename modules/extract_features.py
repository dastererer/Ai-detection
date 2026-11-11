import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from statistics import mean, stdev

from pystylometry.lexical import (
    compute_mtld,
    compute_yule,
    compute_hapax_ratios,
)
from pystylometry.readability.complex_words import is_complex_word
from pystylometry.ngrams import compute_character_bigram_entropy

from modules.tools import  cached_syllables, token_depth
from collections import Counter



nlp = spacy.load("en_core_web_sm", disable=["ner"])




def extract_linguistic_features(
    texts,
    features=None,
    batch_size=128,
):
    if features is None:
        raise ValueError("features must be specified")

    features = list(dict.fromkeys([*features]))
    unknown = set(features) - set(features)
    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")

    texts = list(texts)
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("texts must be a list of strings")

    if not texts or not features:
        return pd.DataFrame(index=range(len(texts)), columns=features, dtype=float)

    dependent = {"csubj", "ccomp", "xcomp", "advcl", "acl", "relcl"}
    rows = []

    docs = nlp.pipe(texts, batch_size=batch_size)

    for doc in tqdm(docs, total=len(texts), desc="extracting features: ", smoothing=0.05):
        words = [token for token in doc if token.is_alpha]
        sentences = [
            sent for sent in doc.sents
            if any(token.is_alpha for token in sent)
        ]

        if not words or not sentences:
            rows.append({name: np.nan for name in features})
            continue

        text = doc.text
        n_words = len(words)
        n_sentences = len(sentences)
        values = {}

        if "mtld" in features:
            values["mtld"] = compute_mtld(text).mtld_average

        if "yule_k" in features:
            values["yule_k"] = compute_yule(text).yule_k

        if "hapax_ratio" in features:
            values["hapax_ratio"] = compute_hapax_ratios(text).hapax_ratio

        if "character_bigram_entropy" in features:
            values["character_bigram_entropy"] = (
                compute_character_bigram_entropy(text).entropy
            )

        if "flesch" in features:
            syllables = sum(cached_syllables(t.lower_) for t in words)
            values["flesch"] = (
                206.835
                - 1.015 * n_words / n_sentences
                - 84.6 * syllables / n_words
            )

        if "gunning_fog" in features:
            complex_words = sum(
                is_complex_word(
                    word=t.text,
                    syllable_count=cached_syllables(t.lower_),
                    use_spacy=True,
                    pos=t.pos_,
                    lemma=t.lemma_,
                )
                for t in words
            )
            values["gunning_fog"] = 0.4 * (
                n_words / n_sentences
                + 100 * complex_words / n_words
            )

        pos = Counter(t.pos_ for t in words)

        values.update({
            "noun_ratio": (pos["NOUN"] + pos["PROPN"]) / n_words,
            "verb_ratio": pos["VERB"] / n_words,
            "adjective_ratio": pos["ADJ"] / n_words,
            "adverb_ratio": pos["ADV"] / n_words,
            "function_word_ratio": (
                pos["DET"] + pos["ADP"] + pos["CCONJ"] + pos["SCONJ"]
            ) / n_words,
        })

        lengths = [
            sum(t.is_alpha for t in sent)
            for sent in sentences
        ]

        values["sentence_length_mean"] = mean(lengths)
        values["sentence_length_std"] = (
            stdev(lengths) if len(lengths) > 1 else 0.0
        )

        edges = [
            t for t in doc
            if not t.is_space and t.head != t
        ]
        distances = [abs(t.i - t.head.i) for t in edges]

        values["dependency_distance"] = (
            mean(distances) if distances else 0.0
        )
        values["left_branching_ratio"] = (
            sum(t.i < t.head.i for t in edges) / len(edges)
            if edges else np.nan
        )

        depths = []
        dependent_count = 0
        coordinate_count = 0
        passive_count = 0

        for sent in sentences:
            depths.append(max(token_depth(t) for t in sent))

            dependent_count += sum(t.dep_ in dependent for t in sent)
            coordinate_count += sum(
                t.dep_ == "conj" and t.pos_ == "VERB"
                for t in sent
            )
            passive_count += any("pass" in t.dep_ for t in sent)

        total_clauses = n_sentences + dependent_count + coordinate_count

        values.update({
            "parse_tree_depth_mean": mean(depths),
            "dependent_clause_ratio": dependent_count / total_clauses,
            "coordination_index": coordinate_count / total_clauses,
            "passive_voice_ratio": passive_count / n_sentences,
            "clausal_density": total_clauses / n_sentences,
        })

        rows.append({name: values[name] for name in features})

    return (
        pd.DataFrame(rows, columns=features, dtype=float)
        .replace([np.inf, -np.inf], np.nan)
    )