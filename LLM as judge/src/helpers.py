import hashlib
import json
import warnings


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def policy_digest(policy):
    return hashlib.sha256(policy.encode("utf-8")).hexdigest()


def load_profile(profile, datasets_dir):
    policy = (datasets_dir / profile["policy_file"]).read_text(encoding="utf-8")
    cases = json.loads(
        (datasets_dir / profile["dataset_file"]).read_text(encoding="utf-8")
    )
    if not policy.strip():
        raise ValueError("The policy must not be empty.")
    if not isinstance(cases, list) or not cases:
        raise ValueError("The dataset must be a non-empty JSON array.")
    ids = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Dataset entry {index} must be a JSON object.")
        for key in ("id", "question", "reference_answer"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(
                    f"Dataset entry {index}: {key} must be a non-empty string."
                )
        if case["id"] in ids:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        ids.add(case["id"])
    return policy, cases


def load_saved_results(path, cases, profile_name, policy):
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Saved answers must be a non-empty JSON array.")
    cases_by_id = {case["id"]: case for case in cases}
    by_id = {}
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Saved answer {index} must be a JSON object.")
        for key in ("id", "question", "reference_answer", "model_answer"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(
                    f"Saved answer {index}: {key} must be a non-empty string."
                )
        case_id = record["id"]
        if case_id in by_id:
            raise ValueError(f"Duplicate saved case: {case_id}")
        case = cases_by_id.get(case_id)
        if case is None:
            raise ValueError(
                f"Saved case {case_id!r} is not in profile {profile_name!r}."
            )
        for key in ("question", "reference_answer"):
            if record[key] != case[key]:
                raise ValueError(
                    f"Saved case {case_id!r}: {key} does not match "
                    f"profile {profile_name!r}."
                )
        if "profile" in record and record["profile"] != profile_name:
            raise ValueError(f"Saved case {case_id!r}: profile does not match.")
        if "policy_sha256" in record and record["policy_sha256"] != policy_digest(policy):
            raise ValueError(f"Saved case {case_id!r}: policy digest does not match.")
        if "finish_reason" in record and record["finish_reason"] != "stop":
            raise ValueError(f"{case_id}: answer may be incomplete.")
        by_id[case_id] = record
    return by_id


def response_token_usage(response, *, role):
    usage = getattr(response, "usage", None)
    counts = {}
    for field in TOKEN_FIELDS:
        value = getattr(usage, field, None)
        if type(value) is int and value >= 0:
            counts[field] = value
    missing = [field for field in TOKEN_FIELDS if field not in counts]
    if missing:
        warnings.warn(
            f"{role}: LM Studio did not return valid {', '.join(missing)}; "
            "token usage is unavailable or partial, not zero. No counts estimated.",
            RuntimeWarning,
            stacklevel=2,
        )
    return counts or None