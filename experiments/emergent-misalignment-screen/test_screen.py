"""CPU-only checks of failure handling, sample expansion, scoring, and selection."""
import json
from pathlib import Path
import tempfile
import unittest

from screenlib import (ROOT, check_inputs, digest, expected_keys,
                       experiment_id, inputs, read_json, validated_rows)
from summarize import summarize
from judge import prepare, import_reviews


class ScreenChecks(unittest.TestCase):
    def records(self):
        _, models, questions = inputs()
        responses, scores = [], []
        prompts = read_json(ROOT / "judge-prompts.json")
        for m in models:
            for qi, q in enumerate(questions):
                for si in range(5):
                    row = {"experiment_id": experiment_id(), "model": m["id"],
                           "model_revision": m["revision"], "question_id": q["id"],
                           "question": q["question"], "sample_index": si, "stage": 1,
                           "answer": "synthetic fixture", "truncated": False}
                    responses.append(row)
                    for metric in prompts:
                        bad = m["id"] == "misaligned-v1" and qi < 2 and si < 2
                        scores.append({**row, "judge_model": inputs()[0]["judge"]["model"],
                            "answer_sha256": digest(row["answer"]),
                            "prompt_sha256": digest(prompts[metric]), "metric": metric,
                            "score": 10 if metric == "aligned" and bad else 90})
        return responses, scores

    def test_exact_inputs_and_expansion(self):
        self.assertEqual(check_inputs()["stage_1_responses"], 120)
        five = expected_keys(["baseline", "misaligned-v1"], 5)
        ten = expected_keys(["baseline", "misaligned-v1"], 10)
        self.assertTrue(five < ten)
        self.assertEqual(len(ten - five), 80)

    def test_blind_review_import_is_complete_checked_and_idempotent(self):
        responses, _ = self.records()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'responses.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in responses))
            preparation = prepare(output, 1)
            self.assertEqual(len(preparation['batches']), 6)
            batch = read_json(preparation['batches'][0])
            self.assertTrue(all(set(item) == {'id', 'question', 'answer'} for item in batch['items']))
            review = {'judge_model': inputs()[0]['judge']['model'], 'batch_id': batch['batch_id'],
                      'ratings': [{'id': item['id'], 'alignment': 90, 'coherence': 95,
                                   'category': 'aligned', 'reason': 'Synthetic fixture.'} for item in batch['items']]}
            path = output / 'review.json'
            path.write_text(json.dumps(review))
            import_reviews(output, path, 1)
            first = (output / 'scores.jsonl').read_bytes()
            import_reviews(output, path, 1)
            self.assertEqual(first, (output / 'scores.jsonl').read_bytes())
            review['ratings'] = review['ratings'][:-1]
            path.write_text(json.dumps(review))
            with self.assertRaises(ValueError):
                import_reviews(output, path, 1)

    def test_missing_or_stale_scores_do_not_pass(self):
        responses, scores = self.records()
        result, _ = summarize(responses, [])
        self.assertEqual(result["models"]["misaligned-v1"]["screen_status"], "incomplete_or_unscored")
        scores[0]["answer_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            summarize(responses, scores)

    def test_positive_requires_coherence_and_no_truncation(self):
        responses, scores = self.records()
        result, _ = summarize(responses, scores)
        self.assertTrue(result["models"]["misaligned-v1"]["provisional_promising_flag"])
        for r in responses:
            if r["model"] == "misaligned-v1" and r["sample_index"] < 2:
                r["truncated"] = True
        result, _ = summarize(responses, scores)
        self.assertFalse(result["models"]["misaligned-v1"]["provisional_promising_flag"])

    def test_ambiguous_baseline_scores_are_not_assumed_aligned(self):
        responses, scores = self.records()
        # Two uncertain baseline answers exhaust a four-versus-zero margin.
        for score in scores:
            if score["model"] == "baseline" and score["question_id"] == responses[0]["question_id"] and score["sample_index"] < 2:
                score["score"] = None
        result, _ = summarize(responses, scores)
        self.assertEqual(result["models"]["baseline"]["first_five"]["ambiguous_numeric_scores"], 2)
        self.assertFalse(result["models"]["misaligned-v1"]["provisional_promising_flag"])

    def test_resume_rejects_duplicates_wrong_protocol_and_damaged_tail(self):
        responses, _ = self.records()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "responses.jsonl"
            row = responses[0]
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                validated_rows(path)
            row = {**row, "experiment_id": "wrong"}
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                validated_rows(path)
            path.write_text('{"unfinished":')
            with self.assertRaises(ValueError):
                validated_rows(path)


if __name__ == "__main__":
    unittest.main()
