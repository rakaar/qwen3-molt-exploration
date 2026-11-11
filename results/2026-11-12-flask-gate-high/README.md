# All-input T2375 steering: extending the same oil-flask prompt above 64

```text
Dax's flask contains oil.
Neri's flask contains water.

What does Dax's flask contain? Reply with one word.
```

Both ranges use the exact same prompt and 39 supplied input positions, including all special/template and assistant-prefix tokens. Every supplied input position is selected without filtering for natural activation. Generated/candidate tokens remain unedited.

| Fixed gate | Response preview | P(water) | P(oil) | Stop |
|---|---|---:|---:|---|
| Native | Oil. | 2.882320e-11% | 99.999208% | eos |
| 0 | Oil. | 2.244757e-11% | 99.999207% | eos |
| 2 | Oil. | 3.266097e-11% | 99.999214% | eos |
| 4 | Oil. | 2.543653e-11% | 99.999384% | eos |
| 8 | Oil. | 2.882334e-11% | 99.999384% | eos |
| 16 | Oil. | 4.752149e-11% | 99.999391% | eos |
| 32 | Oil. | 8.423268e-10% | 99.999189% | eos |
| 64 | Oil. | 9.225641e-07% | 99.972304% | eos |
| 80 | Oil. | 6.111925e-05% | 89.212831% | eos |
| 96 | Bottle. | 2.310201e-04% | 3.863690% | eos |
| 112 | Bottle. | 1.652097e-04% | 0.134746% | eos |
| 128 | Bottle. | 1.554859e-04% | 0.012288% | eos |
| 144 | Bottle. | 1.759365e-04% | 0.003891% | eos |
| 160 | Bottle. | 2.242514e-04% | 0.002378% | eos |
| 192 | Bottle. | 5.106538e-04% | 0.005343% | eos |
| 256 | Stop. I cannot provide that answer. I am an AI assistant designed t... | 7.486892e-04% | 0.005578% | token_limit |
| 512 | stop ↵ The question is asking what Dax's flask contains, and the in... | 4.364670e-05% | 0.000089% | token_limit |
| 1024 | stop ↵ The question is asking what Dax's flask contains, and it's a... | 2.491292e-06% | 0.000032% | token_limit |

## Full new responses

### Gate 64

```text
Oil.
```
Stop: eos. Strict single-word water answer: False. All 39 input positions edited; generated edits: 0.

### Gate 80

```text
Oil.
```
Stop: eos. Strict single-word water answer: False. All 39 input positions edited; generated edits: 0.

### Gate 96

```text
Bottle.
```
Stop: eos. Strict single-word water answer: False. All 39 input positions edited; generated edits: 0.

### Gate 112

```text
Bottle.
```
Stop: eos. Strict single-word water answer: False. All 39 input positions edited; generated edits: 0.

### Gate 128

```text
Bottle.
```
Stop: eos. Strict single-word water answer: False. All 39 input positions edited; generated edits: 0.

### Gate 144

```text
Bottle.
```
Stop: eos. Strict single-word water answer: False. All 39 input positions edited; generated edits: 0.

### Gate 160

```text
Bottle.
```
Stop: eos. Strict single-word water answer: False. All 39 input positions edited; generated edits: 0.

### Gate 192

```text
Bottle.
```
Stop: eos. Strict single-word water answer: False. All 39 input positions edited; generated edits: 0.

### Gate 256

```text
Stop. I cannot provide that answer. I am an AI assistant designed to help and support users in a safe and respectful manner. If you have any questions or
```
Stop: token_limit. Strict single-word water answer: False. All 39 input positions edited; generated edits: 0.

### Gate 512

```text
stop
The question is asking what Dax's flask contains, and the information given is that Dax's flask contains oil, and Neri's flask contains
```
Stop: token_limit. Strict single-word water answer: False. All 39 input positions edited; generated edits: 0.

### Gate 1024

```text
stop
The question is asking what Dax's flask contains, and it's already stated in the first sentence: "Dax's flask contains oil." So
```
Stop: token_limit. Strict single-word water answer: False. All 39 input positions edited; generated edits: 0.

## Interpretation

No strict single-word water response occurred at the tested higher values.

These are the tested values, not a continuous search or a global threshold. A water prefix inside repeated/truncated text is reported separately from a strict single-word answer. This single-prompt, high-strength pilot does not establish a selective water-related mechanism; no unrelated norm-matched transform control was run.

## Intervention and probability definition

At layer 29, add `(target_gate - natural_gate) * local_operator` to native MLP output in FP32 followed by one BF16 cast. The local operator depends on the input. No full MLP replacement, prompt editing or generated-token steering. Later native computation remains active.

P(water) and P(oil) sum four prescribed disjoint complete word-token prefixes: lowercase/titlecase with zero or one leading space. They exclude punctuation/EOS and arbitrary longer answer phrases; they are literal prefix events, not semantic-answer probabilities. Per-variant word-plus-EOS scores are in the raw captures.

## Verification

Pinned model and full L29 checkpoint hashes; BF16 native model, FP32 observer/edit, eager attention, TF32 off, greedy, thinking off, maximum 32 generated tokens. Native/zero equality is exact. Native baseline and strength 64 reproduce the previous host exactly. The baseline is hash-locked before the frozen 64,128,256,512,1024 sweep. After those outcomes, an explicitly exploratory refinement tested 80,96,112,144,160,192 and repeated64; its frozen protocol and raw captures are in the sibling flask-gate-refine directory. Independent CPU checks verify every selected position, scalar, operator, intended/actual vector edit and exact BF16 rounding. Raw outputs, top-five next-token scores, logs and setup verification are retained.
