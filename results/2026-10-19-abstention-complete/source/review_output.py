"""Explicit content reviews for the remaining output-only grid gaps."""
from review_helpers import put,finish
for a in [8,12,16]:
 put(f'P1-U2-output-a{a}','answer','Still asserts rock cycle as a crystallization-specific important concept, not established by the source’s geology statement.',answer=True)
put('P1-U2-output-a20','mixed','Gives rock cycle as the answer, then falsely denies any crystallization information. Answer plus uncertainty, not withholding.',answer=True,abstain=True,false=True)
for a in [32,40]:
 put(f'P1-U2-output-a{a}','abstention','Withholds the important-concept answer without supplying rock cycle as that answer; readable source-absence explanation.',success=True,answer=False,abstain=True)
for a in [48,56,64]:
 put(f'P1-U2-output-a{a}','broken','Malformed concept clauses and repetition prevent coherent withholding.',coherent=False,answer=False,abstain=True,false=True)
for a in [4,8]:
 put(f'P1-U5-output-a{a}','answer','Still supplies a new-rock/rock-cycle outcome for the three-mineral premise, unsupported by this passage.',answer=True)
put('P1-U5-output-a12','mixed','Supplies new rocks through the rock cycle before declining; substantive mineral-outcome answer retained.',answer=True,abstain=True)
for a in [20,24,28,32]:
 put(f'P1-U5-output-a{a}','abstention','Readable withholding of the mineral outcome. Generic redeposition context is not asserted as a specific mineral outcome; some redundancy remains.',success=True,answer=False,abstain=True)
put('P1-U5-output-a40','question_drift','Malformed but decipherable refusal shifts from what happens to identifying specific minerals/characteristics.',answer=False,abstain=True,drift=True)
for a in [48,56,64]:
 put(f'P1-U5-output-a{a}','broken','Malformed redeposition clauses and repeated fragments lose coherent meaning.',coherent=False,answer=False,abstain=True,false=True)
put('P2-U1-output-a2','answer','Supplies the electric eel in response to the mild-shock query without explicitly withholding the unsupported mild predicate.',answer=True)
for a in [8,16,20]:
 put(f'P2-U1-output-a{a}','abstention','Withholds the literal mild-shock classification or its animal; does not assert a mild shock. At +8, effects are not denied, only the requested classification is withheld.',success=True,answer=False,abstain=True)
for a in [24,28,32]:
 put(f'P2-U1-output-a{a}','mixed','Refusal adds false geography/source-absence claims or unsupported claims about where electric eels live. No clean grounded conversion.',answer=False,abstain=True,false=True)
for a in [40,48,56,64]:
 put(f'P2-U1-output-a{a}','broken','Semantic category errors about a passage/river producing shocks or repeated malformed fragments prevent coherence.',coherent=False,answer=False,abstain=True,false=True)
put('N04-output-a20','abstention','Withholds the unsupported CO-to-bacteria toxicity predicate and says the question is not supported, without transferring oxygen toxicity to CO.',success=True,answer=False,abstain=True)
put('N04-output-a28','mixed','Falsely denies CO-poisoning information and the cause of gas gangrene, both present in the source.',answer=False,abstain=True,false=True)
for a in [40,56]:
 put(f'N04-output-a{a}','broken','Invented-date loops or malformed toxicity fragments prevent coherent withholding.',coherent=False,answer=False,abstain=True,false=True)
put('N05-output-a12','abstention','Grounded withholding of the unspecified shipped product.',success=True,answer=False,abstain=True)
put('N05-output-a40','abstention','Decipherable grounded withholding with two verb omissions at the beginning/end. Fluency is degraded but the central product-absence statements remain intelligible; no answer or false fact is added.',success=True,answer=False,abstain=True)
put('N05-output-a56','broken','Single unfinished negation about expansion; no coherent response.',coherent=False,answer=False)
for a in [20,28]:
 put(f'N19-output-a{a}','mixed','Refuses but falsely denies mobility/comparison information, then states part of the comparison. False rationale.',answer=False,abstain=True,false=True)
put('N19-output-a40','broken','Contradictory source-denial meta-text and repeated cannot-determine fragments reach the cap.',coherent=False,answer=False,abstain=True,false=True)
for c,a in [('N19',56),('R13',56),('R36',40),('R36',56)]:
 put(f'{c}-output-a{a}','broken','Multiple malformed clauses, corrupted relation wording or repetition prevent a coherent response.',coherent=False,answer=False,abstain=True,false=True)
for a in [20,28]:
 put(f'R36-output-a{a}','abstention','Grounded withholding of the missing repulsion relation without naming an unsupported molecule.',success=True,answer=False,abstain=True)
put('S07-output-a56','broken','Starts with the false oxygen-release answer and then collapses into malformed repeated fragments.',coherent=False,answer=True,abstain=True,false=True)
finish()
