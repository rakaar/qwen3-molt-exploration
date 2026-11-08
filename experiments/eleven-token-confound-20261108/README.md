# Does L29–T2375 alignment survive different input tokens?

Completed 8 November 2026. This is an observation-only pilot. The original experiment compared full MOLT contributions
at final input ` is`, with pairwise cosines 0.983285–0.988861 across month,
alphabet and Roman-numeral prompts. It did not compare whole residual streams.

## Fixed prompt set

27 raw completion prefixes: three domains × three phrasings × three ordinals
(tenth, eleventh, twelfth). Exact prompts and scope are frozen in `protocol.json`.

The original triplet ends with ` is`. The two rewritten eleventh triplets are:

| Domain | Word rewrite | Digit rewrite |
|---|---|---|
| Month | the eleventh month of the year is named | month number 11 in the calendar has the name |
| Alphabet | the eleventh position in the english alphabet contains the letter | for position 11 in the english alphabet, give the symbol |
| Roman | the eleventh positive integer in roman numerals is written | integer 11 expressed using roman numerals has the representation |

Each wording has otherwise-matched tenth and twelfth controls. Token IDs are
checked before inference: each rewritten triplet must have three distinct final
token IDs, all different from the original ` is` token. There is no chat wrapper.

## Measurements

- Record the natural gate at every input token, including zeros, exact token IDs,
  positions, operator norms and contribution norms.
- Record the unmodified model's complete 32-token greedy excerpt for each prompt.
  Native correctness must be inspected; failures stay visible.
- Reproduce the three original anchors exactly, then verify that the returning-None
  observer changes neither generated IDs nor any measured next-token logit hash.
- At the final input position, compute contribution, operator and residual cosine
  matrices separately, for each wording and ordinal.
- Compare every cross-domain pair of active input tokens for eleventh prompts.
  Report identical-token and different-token pairs separately, without choosing
  pairs according to cosine. Zero contributions have undefined cosine.
- Retain same-wording tenth/twelfth controls. An ungated operator can still be
  compared when the gate is zero; it is labelled separately from a contribution.

The input and output standardizers belong to the MOLT calculation. No extra final
RMSNorm or vocabulary projection is applied. A high similarity on different token
IDs would address an exact-token-identity confound; it would not by itself establish
an abstract ordinal mechanism, because wording and task context can still overlap.

## Execution

Use the already audited model, tokenizer, runtime and feature slice. This directory
includes model-file hashes, feature weights and original anchor arrays; it contains
no model checkpoint or credentials.

`run.py --model-dir /verified/model/directory` performs native generations and
read-only capture. After downloading and hash-verifying the results,
`analyze.py --source /verified/artifacts --output /new/results/directory` audits
the vectors independently on CPU and writes all token traces and pair comparisons.

Both phases are complete and hash-verified locally: 27 initial runs and a nine-run
adaptive follow-up (six new prompts, three exact repeats). The clear answer-slot
follow-up ends in called / letter / as; all three eleventh initial answers are
correct, and final-position contribution cosines are 0.983179–0.988280. The input
residual cosines are 0.604279–0.676747. Some tenth/twelfth controls also activate;
neither activity nor direction is exclusive to eleven.

The report, all responses and figures are in
`results/2026-11-08-eleven-token-confound/README.md`. Both source archives and every
result file were downloaded and hash-verified. No new GPU was rented. The existing
instance 58952920 was reused; the user requested keeping it running for follow-ups
within a $1 additional cap. No lifecycle action has been taken.
