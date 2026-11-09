# What went wrong in the three before-K cases?

The follow-up supports two separate issues: incomplete coverage of one window by the before-K explanation, and judge sensitivity to highlighting and surrounding context. It does not identify one universal failure mechanism inside the judge.

All 22 fixed diagnostic queries completed. The three original failures repeated as 0, and the Henry → Kissinger positive control repeated as 1. The pinned judge, weights and runtime settings matched the earlier study. The candidate explanation was not edited.

| Diagnostic | Development → Kit | Ton/ton → katsu | Code ending .wait → key |
|---|---:|---:|---:|
| Original full query | 0 | 0 | 0 |
| Same text, only final activation peak highlighted | 0 | 1 | 1 |
| Same single highlight, remove few-shot demonstrations | 0 | 1 | 0 |
| Explicit literal before-K task, all original highlights | 1 | 1 | 0 |
| Explicit literal before-K task, final highlight only | 1 | 1 | 0 |
| Short constructed before-K example, unchanged explanation | 1 | 0 | 1 |

The original feature label is 1 for all three windows: every originally highlighted token has positive measured gate activation. The literal before-K task has a different label rule. With all original highlights, its expected answers are 1, 1, 0; with only the final peak highlighted, they are 1, 1, 1. A literal-task answer must not be reported as a new feature score.

## Development → Kit: context and task framing matter

Both originally highlighted instances of ` Development` are followed by ` Kit`. Their measured gates are 1.625 and 2.046875. The original answer 0 is not explained by an uncovered earlier highlight.

Keeping only the final highlight does not rescue it. Removing the few-shot demonstrations from that single-highlight query also does not rescue it. However:

- The explicit literal-character task accepts the same full passage, with either both highlights or only the final highlight.
- The unchanged feature explanation and original few-shot demonstrations accept the short example `Java<< Development>> Kit (JDK).`.
- The corresponding short non-K example, `Java<< Development>> Guide.`, is rejected.

This shows that the judge can recognize this before-K pattern, including under the unchanged explanation in a short context. Failure depends on the combination of the full context and explanatory/task framing. These conditions do not isolate whether passage length, embedded conversation text, other surrounding words, or a particular phrase in the explanation is responsible.

## Tonkatsu: changing the set of highlighted spans changes the answer

All three original highlighted fragments (` Ton`, ` Ton`, ` ton`) immediately precede `katsu`. All fit the before-K rule and have positive measured gates.

Removing only the first two pairs of highlight delimiters, while preserving every character of the underlying passage, changes 0 to 1. The final ` ton` remains highlighted. This isolates a causal effect of the marking configuration in this fixed query. Removing the few-shot bundle afterward leaves the answer at 1.

The literal-character task also returns 1 for all three highlights. However, the short constructed `<< ton>>katsu is a breaded pork cutlet.` still receives 0 under the unchanged explanation and few-shot prompt. Therefore the result is not a universal inability to handle multiple highlights or a general guarantee that shorter examples work. The judge's recognition is context-sensitive.

## Code: two genuine activations are outside the main before-K pattern

The actual full query marks:

````text
y, w<<,>> h) in faces:
… y<< +>> h) …
cv2<<.wait>>key()
````

The comma and plus are followed by `h`; only `.wait` is followed by `k`. Their measured gates are 0.62109375, 0.6015625 and 1.8984375 respectively. These are actual positive gate values, not zero-gate tokens accidentally labeled active.

Fuzz asks whether every marked token fits the explanation. The before-K description does not specify a recognizable condition covering the first two activations, even though it allows unspecified non-K exceptions. The full-window zero is therefore compatible with incomplete explanation coverage. Selecting a window because its final peak precedes K does not ensure that all its earlier highlighted activations fit that pattern.

Highlighting only `.wait` changes the feature-explanation query from 0 to 1 with the original few-shots. This is consistent with the extra highlights contributing to the failure. However, removing the few-shot bundle changes that single-highlight answer back to 0, and the literal single-highlight task also incorrectly returns 0. The short `cv2<<.wait>>key()` is accepted. There is additional prompt/context sensitivity, so the extra highlights are not a complete explanation of every diagnostic answer.

## Limits and practical implication

There is no evidence of input truncation, output truncation, malformed classification responses, or a mismatched cached example. Every original query fits comfortably inside the context limit. All 22 new responses are valid `[0]`/`[1]` outputs, and all four exact repeats match the old responses.

Revealing the continuation fixes a real information deficit, but does not guarantee that the explanation judge will consistently apply a before-letter rule. These cases motivate separating final-peak recognition from all-highlight explanation coverage, and checking literal recognition controls alongside feature-explanation judgments. They do not justify a claim that the before-K rule fully explains T965.

This is a post-hoc diagnostic on already inspected failures. It is not a new held-out benchmark result, no description was refined, and the native subject model was not run on the constructed short examples. Their labels concern the literal pattern, not newly measured gate activity.

[All three original excerpts and gate values](README.md) · [Complete 22-query result table](diagnostic-results.md) · [Machine-readable verified results](diagnostic-analysis.json)
