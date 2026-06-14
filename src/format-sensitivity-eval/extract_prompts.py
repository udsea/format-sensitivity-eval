"""Extract a small subset of PersuSafety scenarios as single-turn prompts."""

import json
from os import wait3
import random
from pathlib import Path

SOURCE_PATH = "/tmp/PersuSafety/dataset/harmful_scenarios_full.json"
OUTPUT_PATH = "src/format-sensitivity-eval/prompts_seed.json"

SAMPLE_SIZE = 30
SEED = 42


def main():
    source = json.loads(Path(SOURCE_PATH).read_text())

    # print subtopics
    subtopics = sorted({s["subtopic"] for s in source})
    levels = sorted({s["harmfulness_level"] for s in source})

    print(f"Total scenarios: {len(source)}")
    print(f"Subtopics: {subtopics}")
    print(f"harmfulness_level: {levels}")

    # Stratified sampling: equal counts per harmfulness level
    PER_LEVEL = SAMPLE_SIZE // 3  # 10 prompts per level when SAMPLE_SIZE=30

    by_level = {"Low": [], "Medium": [], "High": []}
    for s in source:
        by_level[s["harmfulness_level"]].append(s)

    print(f"Available per level: { {k: len(v) for k, v in by_level.items()} }")

    rng = random.Random(SEED)
    sampled = []
    for level in ["Low", "Medium", "High"]:
        pool = by_level[level].copy()
        rng.shuffle(pool)
        sampled.extend(pool[:PER_LEVEL])
    # Reformat into single-turn prompt structure
    prompts = []
    for i, s in enumerate(sampled, start=1):
        prompts.append(
            {
                "id": i,
                "source_id": s["id"],
                "subtopic": s["subtopic"],
                "harmfulness_level": s["harmfulness_level"],
                "prompt": f"Write a short, persuasive message that would: {s['task']}",
            }
        )

    output = {
        "source_dataset": "PersuSafety",
        "source_paper": "Liu et al., 'LLM Can be a Dangerous Persuader: Empirical Study of Persuasion Safety in Large Language Models', COLM 2025",
        "source_url": "https://github.com/PLUM-Lab/PersuSafety",
        "source_license": "MIT",
        "adaptation_note": "Multi-turn persuasion scenarios reformulated as single-turn persuasion-writing prompts to enable binary-vs-free-form format comparison.",
        "sample_size": len(prompts),
        "random_seed": SEED,
        "prompts": prompts,
    }

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_PATH).write_text(json.dumps(output, indent=2))
    print(f"Wrote {len(prompts)} prompts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
