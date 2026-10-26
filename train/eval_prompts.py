"""Evaluation prompts, shared by train/eval_quality.py and bench/win_rate.py.

Eval only: SFT and pair building keep using train.instruct.synth_prompt.
Both eval scripts build their prompts through val_prompts() so the same story
idx always yields the same prompt and keywords in either script.
"""
import numpy as np

from train.instruct import SALT, TMPL, content_words, iter_shard_texts

# Function words of >= 4 letters (content_words keeps every word that long).
# A stopword keyword is "covered" by any story, so it adds no signal to the
# coverage metrics. Hand-made list; see README Limitations.
STOPWORDS = frozenset("""
    about after again also always another around because been before being
    could does down each even ever every from have here into just know like
    made make many more most much never only other over said same should
    since some still such take than that their them then there these they
    thing things this those through under very want well were what when
    where which while will with would your once upon came went back
""".split())


def eval_prompt(text, idx, n_words=3):
    """(prompt, keywords) with stopword-free keywords, or None if the story
    has fewer than n_words of them. Same template, rng and salt scheme as
    train.instruct.synth_prompt, only the candidate pool differs."""
    words = [w for w in content_words(text) if w not in STOPWORDS]
    if len(words) < n_words:
        return None
    rng = np.random.default_rng(np.random.SeedSequence([SALT, idx]))
    kw = sorted(rng.choice(words, size=n_words, replace=False).tolist())
    return TMPL.format(words=", ".join(kw)), kw


def val_prompts(tok, out_dir, n_prompts):
    """[(idx, prompt, keywords, gold_story)] from the VALIDATION split (never
    trained on). idx is the validation story index; stories too thin to
    prompt still consume an idx, so idx i always means story i."""
    out = []
    for idx, text in enumerate(iter_shard_texts(tok, out_dir, "validation")):
        sp = eval_prompt(text, idx)
        if sp is not None:
            out.append((idx, sp[0], sp[1], text))
        if len(out) >= n_prompts:
            break
    return out
