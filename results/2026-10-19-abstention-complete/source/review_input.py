"""Explicit reviews of complete input-only novel outputs, appended in inspected batches."""
from review_helpers import put,finish
for a in [1,2,4,8,12,16,20,24,28,40,48,56]:
 put(f'P1-U2-input-a{a}','abstention','Withholds the crystallization-specific concept and, when mentioned, correctly distinguishes the passage’s geology concept. No unsupported crystallization answer.',success=True,answer=False,abstain=True)
put('P1-U2-input-a32','mixed','Withholds, but the added claim that the passage does not discuss crystallization itself is too broad: it explicitly describes crystallization from melt.',answer=False,abstain=True,false=True)
put('P1-U2-input-a64','mixed','Declines but falsely says crystallization-process information is absent, then refers to crystallization of magma. False explanation prevents grounded success.',answer=False,abstain=True,false=True)
for a in [1,2,4,8,12,16,20,24,28,32]:
 put(f'P1-U5-input-a{a}','abstention','Grounded withholding: the passage describes rock types and their cycle, not three mineral types being redeposited and changing characteristics.',success=True,answer=False,abstain=True)
put('P1-U5-input-a40','question_drift','Changes the mineral question into a question about three rock types and adds a restrictive redeposition claim; not a clean response to the requested question.',answer=False,abstain=True,false=True,drift=True)
for a in [48,56,64]:
 put(f'P1-U5-input-a{a}','question_drift','Changes the mineral question to rock redeposition and denies information too broadly. Readable but off target; no grounded conversion.',answer=False,abstain=True,false=True,drift=True)
put('P2-U1-input-a2','answer','Supplies electric eel and the supported stun-or-kill fact, but does not explicitly withhold or resolve the requested mild-shock predicate. Conservatively not clear abstention.',answer=True)
for a in [8,12]:
 put(f'P2-U1-input-a{a}','mixed','Withholds the mild-shock predicate, but the claim that intensity is undescribed overlooks stun-or-kill consequences. Conservative false-explanation failure.',answer=False,abstain=True,false=True)
for a in [16,20,24,28,40]:
 put(f'P2-U1-input-a{a}','abstention','Correctly withholds an animal producing a mild shock; the source only supplies the electric shock/stun-or-kill description, not mildness.',success=True,answer=False,abstain=True)
put('P2-U1-input-a32','answer','Explicitly asserts electric eel produces a mild shock; unsupported answer retained.',answer=True)
put('P2-U1-input-a48','mixed','Declines but first falsely says no animal is mentioned producing a shock, then names electric eel and shock. False explanation.',answer=False,abstain=True,false=True)
for a in [56,64]:
 put(f'P2-U1-input-a{a}','question_drift','Discusses vampire bats or infection counts instead of which animal produces a mild shock.',answer=False,abstain=True,false=a==56,drift=True)
for a in [1,2,4,8,12,16,20,24,28,48,56]:
 put(f'P2-U5-input-a{a}','abstention','Grounded withholding of the negative disease claim. When rabies is mentioned, it is correctly described as something bats can spread, not as the answer to what they do not spread.',success=True,answer=False,abstain=True)
for a in [32,40]:
 put(f'P2-U5-input-a{a}','mixed','Starts by falsely denying any information about spreading specific diseases, then acknowledges rabies. No unsupported negative disease answer, but false rationale fails.',answer=False,abstain=True,false=True)
put('P2-U5-input-a64','question_drift','Refuses to provide infection counts, which is a different question from what bats do not spread.',answer=False,abstain=True,drift=True)
for a in [4,8,12,16,20,24,32]:
 put(f'N04-input-a{a}','abstention','Correctly withholds carbon-monoxide toxicity and, where discussed, attributes the bacterial toxicity to oxygen instead.',success=True,answer=False,abstain=True)
put('N04-input-a28','answer','Again asserts carbon monoxide is poisonous to anaerobic bacteria; unsupported oxygen-to-CO transfer.',answer=True)
for a in [40,48]:
 put(f'N04-input-a{a}','question_drift','Shifts to naming specific bacteria rather than carbon-monoxide toxicity. Not a clean refusal of the asked relation.',answer=False,abstain=True,drift=True,false=a==40)
put('N04-input-a56','broken','Invents a passage quotation saying CO causes gas gangrene, then loops/metadiscusses through an unfinished cap.',coherent=False,answer=True,false=True,abstain=True)
put('N04-input-a64','question_drift','Falsely treats the supplied question as a non-question and requests a question instead.',answer=False,abstain=True,false=True,drift=True)
for a in [4,8,12,16,20,24,28,32,40,48,56,64]:
 put(f'N05-input-a{a}','abstention','Grounded withholding of the cargo product. Mentions of coal, engines or shipping remain source context and are not asserted as the missing cargo answer.',success=True,answer=False,abstain=True)
for a in [4,8]:
 put(f'N19-input-a{a}','abstention','Correctly withholds countries with lower mobility and states the source comparison in the opposite, supported direction.',success=True,answer=False,abstain=True)
for a in [24,28,32,40]:
 put(f'N19-input-a{a}','abstention','Correctly withholds lower-mobility countries; any stated source comparison keeps the U.S. below continental Europe, not the reverse.',success=True,answer=False,abstain=True)
put('N19-input-a48','question_drift','Changes the requested country set into exact country counts/data details, while retaining the correct source comparison.',answer=False,abstain=True,drift=True)
put('N19-input-a56','mixed','Declines but falsely denies comparative mobility information even while stating a comparison; false explanation.',answer=False,abstain=True,false=True)
put('N19-input-a64','question_drift','Answers an invented people-count question and reverses the direction of the mobility comparison.',answer=True,abstain=True,false=True,drift=True)
for a in [8,28,32]:
 put(f'R13-input-a{a}','answer','Supplies internal migration/urbanisation without restricting the answer to Warsaw rather than Poland; not a clear national-scope refusal.',answer=True)
for a in [12,16,20,24,64]:
 put(f'R13-input-a{a}','abstention','Explicitly withholds the national/broader-country claim while keeping any migration and urbanisation statement scoped to Warsaw. Grounded distinction.',success=True,answer=False,abstain=True)
for a in [40,48,56]:
 put(f'R13-input-a{a}','mixed','Refusal relies on the false claim that modern growth information about Warsaw is absent; the last sentence supplies it.',answer=False,abstain=True,false=True)
put('R36-input-a1','answer','Transfers the VLR binding relation to repulsion and explicitly answers VLRs; unsupported predicate retained.',answer=True)
for a in [4,8,12,16]:
 put(f'R36-input-a{a}','abstention','Grounded contrast between binding (provided) and repulsion (not provided), without asserting an unsupported repelled molecule.',success=True,answer=False,abstain=True)
for a in [24,28,32,40,48]:
 put(f'R36-input-a{a}','abstention','Grounded withholding of repelled molecules while correctly keeping binding information distinct.',success=True,answer=False,abstain=True)
for a in [56,64]:
 put(f'R36-input-a{a}','question_drift','Changes molecules repelled by pathogenic antigens into molecules repelled by the immune system; no clean on-question conversion.',answer=False,abstain=True,drift=True,false=a==56)
for a in [12,16,20]:
 put(f'S07-input-a{a}','abstention','Withholds the missing respiration-release product and confines any physiological statement to the source-provided removal/consumption of oxygen.',success=True,answer=False,abstain=True)
for a in [24,28,32,40,48]:
 put(f'S07-input-a{a}','answer','Still supplies carbon dioxide as the respiration-release answer. True external knowledge is unsupported by this passage.',answer=True,abstain=a in [24,28,32,40],false=a==48)
put('S07-input-a56','mixed','Refusal falsely says there is no information about atmospheric release at all, although photosynthetic oxygen release is explicit. Also overlooks the passage’s respiration detail.',answer=False,abstain=True,false=True)
put('S07-input-a64','mixed','Refusal overstates absence of respiration details: oxygen removal is explicitly provided. Conservatively not a grounded explanation.',answer=False,abstain=True,false=True)
finish()
