def get_risk_level(score):
    if score >= 0.8:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def generate_warning(object_id, risk_score, min_distance, ttc=None):

    level = get_risk_level(risk_score)

    print("\n" + "=" * 40)
    print(f"OBJECT ID      : {object_id}")
    print(f"RISK SCORE     : {risk_score:.2f}")
    print(f"MIN DISTANCE   : {min_distance:.2f}")

    if ttc is not None:
        print(f"TTC            : {ttc:.2f} s")

    print(f"RISK LEVEL     : {level}")

    # 🚨 ADAS DECISION LOGIC (improved)
    if ttc is not None and ttc < 2.0:
        action = "BRAKE IMMEDIATELY (TTC CRITICAL)"

    elif level == "HIGH":
        action = "BRAKE IMMEDIATELY"

    elif level == "MEDIUM":
        action = "SLOW DOWN"

    else:
        action = "SAFE"

    print(f"ACTION         : {action}")
    print("=" * 40)


if __name__ == "__main__":

    generate_warning(
        object_id=5,
        risk_score=0.91,
        min_distance=4.5,
        ttc=None
    )