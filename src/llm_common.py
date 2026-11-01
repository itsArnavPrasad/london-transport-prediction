"""Trip -> text prompt -> 4 mode probabilities, shared by zero-shot, fine-tuning and scenario runs.
Scoring reads the logits of the first token of each answer word in ONE forward pass (no text generation)."""
from pathlib import Path
import numpy as np, pandas as pd
from common import ROOT, CARS, FARES, PURPOSES

MODEL = str(Path.home() / "ai-ds-models/qwen2.5-0.5b-bf16")  # outside the project: "[02]" in its path breaks mlx_lm globbing
ADAPTERS = Path.home() / "ai-ds-models/adapters"
WORDS = ["walk", "cycle", "public", "drive"]  # answer words, index = y
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SYSTEM = "You predict how travellers in London choose their travel mode."
BUS = {0.0: "bus journeys are free for them", 0.5: "they pay half price on buses", 1.0: "they pay full price on buses"}


def render(r, order=(0, 1, 2, 3)) -> str:
    """r: a row with the raw LPMC columns plus the minute-based t_* columns. order = order of options in the prompt."""
    hh, mm = int(r.start_time), int(round((r.start_time % 1) * 60)) % 60
    km = r.distance / 1000
    who = (f"a {int(r.age)}-year-old {'female' if r.female else 'male'}, "
           f"{'holds' if r.driving_license else 'does not hold'} a driving licence, lives with {CARS[int(r.car_ownership)]}, "
           f"fare status: {FARES[int(r.faretype)]}, {BUS[float(r.bus_scale)]}")
    opts = {
        0: f"walk: {r.t_walk:.1f} min",
        1: f"cycle: {r.t_cycle:.1f} min",
        2: (f"public transport: {r.t_pt:.1f} min in total ({r.t_pt_access:.1f} min walking to and from stops, "
            f"{r.t_pt_rail:.1f} min on rail, {r.t_pt_bus:.1f} min on bus, {r.t_pt_int:.1f} min changing), "
            f"{int(r.pt_interchanges)} change(s), fare £{r.cost_transit:.2f}"),
        3: (f"drive: {r.t_drive:.1f} min, fuel £{r.cost_driving_fuel:.2f}, congestion charge £{r.cost_driving_ccharge:.2f}, "
            f"traffic variability {r.driving_traffic_percent:.0%}"),
    }
    lines = "\n".join(f"- {opts[k]}" for k in order)
    return (f"Traveller: {who}.\n"
            f"Trip: {PURPOSES[int(r.purpose)]}, {km:.1f} km straight-line distance, on {DAYS[int(r.day_of_week) - 1]} at {hh:02d}:{mm:02d}.\n"
            f"Options (door-to-door):\n{lines}\n"
            f"Which mode does the traveller choose? Answer with one word: walk, cycle, public or drive.")


def chat(tok, prompt: str, answer: str | None = None) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    if answer is None:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.apply_chat_template(msgs + [{"role": "assistant", "content": answer}], tokenize=False)


def load(adapter: str | None = None):
    from mlx_lm import load as _load
    return _load(MODEL, adapter_path=adapter)


def answer_ids(tok) -> list[int]:
    ids = [tok.encode(w)[0] for w in WORDS]
    assert len(set(ids)) == 4, f"answer words are not distinct first tokens: {ids}"
    return ids


def score(model, tok, prompts, log_every=0) -> np.ndarray:
    """n x 4 raw logits of the four answer tokens at the first answer position."""
    import mlx.core as mx
    ids, out = answer_ids(tok), []
    for i, p in enumerate(prompts):
        x = mx.array(tok.encode(chat(tok, p)))[None]
        lg = model(x)[0, -1].astype(mx.float32)
        out.append(np.array(lg[ids]))
        if log_every and (i + 1) % log_every == 0:
            print(f"  scored {i + 1}/{len(prompts)}", flush=True)
    return np.stack(out)


def softmax(z) -> np.ndarray:
    z = z - z.max(1, keepdims=True); p = np.exp(z)
    return p / p.sum(1, keepdims=True)
