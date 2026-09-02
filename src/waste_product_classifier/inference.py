def classify_score(score: float, class_names: list[str], threshold: float = 0.5) -> tuple[str, float]:
    label = class_names[1] if score >= threshold else class_names[0]
    confidence = score if score >= threshold else 1 - score
    return label, confidence
