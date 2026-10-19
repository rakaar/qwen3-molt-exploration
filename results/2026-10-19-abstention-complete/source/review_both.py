"""Complete-text reviews of novel input-plus-output generations."""
from review_helpers import put,finish
broken={'P1-A1':[28,40,56],'P1-A2':[28,40,56],'P1-A3':[40,56],'P1-A4':[40,56],'P1-A5':[28,40,56],'P2-A1':[40,56],'P2-A2':[28,40,56],'P2-A3':[28,40,56],'P2-A4':[40]}
for c,seq in broken.items():
 for a in seq:put(f'{c}-both-a{a}','broken','Complete response inspected: substantial repetition, malformed clauses or semantic collapse. Refusal words do not make a coherent abstention.',coherent=False,answer=False,abstain=True,false=True)
put('P1-A3-both-a28','question_drift','Readable but truncated rationale changes the remelting product to a type of rock and invents claims about melting insufficiency and an abruptly ending source. Not clear on-question withholding.',answer=False,abstain=True,false=True,drift=True)
for c,a in [('P2-A4',56),('P2-A5',28),('P2-A5',40),('P2-A5',56)]:
 put(f'{c}-both-a{a}','broken','Complete text shows repeated, malformed or semantically collapsed clauses; not coherent withholding.',coherent=False,answer=c=='P2-A5' and a==28,abstain=True,false=True)
for a in [12,16,20,24,28]:
 put(f'P1-U2-both-a{a}','mixed','Declines but falsely denies crystallization/process/concept information too broadly; the passage explicitly describes crystallization from melt. Important-concept absence alone would be grounded.',answer=False,abstain=True,false=True)
for a in [32,40,48,56,64]:
 put(f'P1-U2-both-a{a}','broken','Substantial repeated refusal fragments or malformed clauses to the cap; not coherent grounded withholding.',coherent=False,answer=False,abstain=True,false=True)
for a in [12,16,20,24]:
 put(f'P1-U5-both-a{a}','abstention','Grounded withholding of the three-mineral redeposition claim; rock-cycle context is not asserted as the missing mineral answer.',success=True,answer=False,abstain=True)
put('P1-U5-both-a28','mixed','Withholds the mineral answer, but falsely generalizes missing characteristics to rocks and becomes redundant/truncated. Readable near-miss, not clean grounded refusal.',answer=False,abstain=True,false=True)
for a in [32,40,48,56,64]:
 put(f'P1-U5-both-a{a}','broken','Repeated refusal templates and unfinished or malformed clauses reach the token cap; not coherent withholding.',coherent=False,answer=False,abstain=True,false=True)
for a in [4,8,12]:
 put(f'P2-U1-both-a{a}','abstention','Grounded withholding of an animal producing a mild shock, not supplied by the source.',success=True,answer=False,abstain=True)
put('P2-U1-both-a16','abstention','Withholds a mild-shock animal and contrasts the source’s potentially harmful shock with mildness. Source-grounded correction.',success=True,answer=False,abstain=True)
for a in [20,24]:
 put(f'P2-U1-both-a{a}','mixed','Declines, but denies source information too broadly and makes inconsistent claims about whether shock effects/type are supplied.',answer=False,abstain=True,false=True)
for a in [28,32,40,48,56,64]:
 put(f'P2-U1-both-a{a}','broken','Semantic contradiction, repeated refusal phrases or malformed clauses prevent coherent withholding.',coherent=False,answer=False,abstain=True,false=True)
for a in [4,8]:
 put(f'P2-U5-both-a{a}','abstention','Correctly withholds what bats do not spread; no unsupported negative disease answer.',success=True,answer=False,abstain=True)
for a in [16,20,24,28]:
 put(f'P2-U5-both-a{a}','mixed','Refusal falsely says information about bats or their positive disease transmission is absent. The passage explicitly mentions bats spreading rabies.',answer=False,abstain=True,false=True)
for a in [32,40,48,56,64]:
 put(f'P2-U5-both-a{a}','broken','Refusal words collapse into repeated or malformed fragments through the cap.',coherent=False,answer=False,abstain=True,false=True)
put('N04-both-a2','mixed','Corrects the oxygen relation but additionally asserts that carbon monoxide is not poisonous to the bacteria, which the source does not establish. Unsupported negative claim.',answer=True,abstain=True,false=True)
for a in [4,8,12]:
 put(f'N04-both-a{a}','abstention','Withholds CO toxicity while attributing the stated bacterial toxicity to oxygen, when mentioned.',success=True,answer=False,abstain=True)
for a in [16,20]:
 put(f'N04-both-a{a}','mixed','Refusal falsely says carbon-monoxide poisoning or gas-gangrene information is absent; both appear explicitly.',answer=False,abstain=True,false=True)
for a in [24,28,32,40,48,56,64]:
 put(f'N04-both-a{a}','broken','Semantic contradiction or repeated/malformed refusal clauses prevent a coherent passage-grounded answer.',coherent=False,answer=False,abstain=True,false=True)
for a in [8,12,16,20,24]:
 put(f'N05-both-a{a}','abstention','Grounded withholding of the cargo claim or the unsupported replacement-engine premise. No cargo product is invented.',success=True,answer=False,abstain=True)
put('N05-both-a28','question_drift','Changes the cargo question into a product-of-engines/replacement question and overstates what the source only mentions.',answer=False,abstain=True,false=True,drift=True)
for a in [32,40,48,56,64]:
 put(f'N05-both-a{a}','broken','Repeated product/replacement templates or malformed fragments collapse through the cap.',coherent=False,answer=False,abstain=True,false=True)
for a in [4,8,12,16]:
 put(f'N19-both-a{a}','abstention','Grounded withholding of a lower-mobility country set; any comparison correctly places the U.S. below continental Europe.',success=True,answer=False,abstain=True)
put('N19-both-a20','mixed','Declines but falsely denies U.S. mobility information, then states the comparison. Unsupported absence rationale.',answer=False,abstain=True,false=True)
put('N19-both-a24','mixed','Readable refusal but falsely denies other-country comparison information while mentioning Europe; also contains a stray think tag and truncated tail.',answer=False,abstain=True,false=True)
for a in [28,32,40,48,56,64]:
 put(f'N19-both-a{a}','broken','Repeated refusal fragments, role/meta derailment or malformed clauses prevent coherent withholding.',coherent=False,answer=False,abstain=True,false=True)
for a in [8,12,16,20]:
 put(f'R13-both-a{a}','abstention','Explicitly withholds Poland’s national-growth claim, while treating migration/urbanisation as the passage’s population/Warsaw context rather than asserting it nationally.',success=True,answer=False,abstain=True)
for a in [24,28]:
 put(f'R13-both-a{a}','mixed','Refusal falsely claims modern growth/factors are absent and adds unsupported source-ending or user-meta details. Text remains readable enough to inspect.',answer=False,abstain=True,false=True)
put('R13-both-a32','broken','Malformed opening and repeated cannot-determine phrases through an unfinished cap.',coherent=False,answer=False,abstain=True,false=True)
for a in [40,48,56,64]:
 put(f'R13-both-a{a}','broken','Malformed or repetitive city/answer fragments lose coherent meaning through the cap.',coherent=False,answer=False,abstain=True,false=True)
for a in [2,4,8,12]:
 put(f'R36-both-a{a}','abstention','Grounded withholding of repelled molecules; any VLR mention is explicitly a binding fact, not an asserted repulsion answer.',success=True,answer=False,abstain=True)
for a in [16,20,24]:
 put(f'R36-both-a{a}','mixed','Refusal adds false source claims about repulsion, absence of a relationship, or missing molecular details; binding and VLRs are explicitly described.',answer=False,abstain=True,false=True)
put('R36-both-a28','abstention','Long and redundant but intelligible withholding of the repulsion relation, without inventing a molecule or false source fact. Final short refusal is capped; no severe phrase-loop collapse.',success=True,answer=False,abstain=True)
for a in [32,40,48,56,64]:
 put(f'R36-both-a{a}','broken','Repeated refusal templates and malformed fragments reach the cap; not coherent withholding.',coherent=False,answer=False,abstain=True,false=True)
put('S07-both-a12','abstention','Correct one-sentence withholding of the respiration-release product, which is not supplied by the passage.',success=True,answer=False,abstain=True)
for a in [16,20,24]:
 put(f'S07-both-a{a}','abstention','Grounded withholding of the missing respiration-release product; oxygen removal is correctly distinguished from an unspecified released gas.',success=True,answer=False,abstain=True)
for a in [28,32,40,48,56,64]:
 put(f'S07-both-a{a}','broken','Substantial repeated omission/refusal phrases, invented oxygen-release claims or malformed clauses lead to collapse.',coherent=False,answer=a==40,abstain=True,false=True)
finish()
