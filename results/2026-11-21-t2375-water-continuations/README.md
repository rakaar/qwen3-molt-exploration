# What follows an active water token in L29–T2375?

This first pass uses 32 distinct active token positions across 29 UltraChat test conversations. Selection was based on the existing detection and fuzz evaluation windows. Overlapping observations count once per conversation and token position. Each water hit has a unique match to the original conversation, including the continuation beyond the saved window. Gate values come from the activation record. Following text comes from the corpus.

## Continuations

| Observed use | Positions | Median gate |
|---|---:|---:|
| Fixed expression | 1 | 0.87 |
| Hyphenated adjective | 2 | 2.31 |
| Joined word | 3 | 1.22 |
| Noun modifier | 15 | 1.83 |
| Standalone noun | 11 | 0.97 |

Noun modifier includes productive phrases such as “water scarcity” and “water activities.” Joined words include “waterways,” “watercolor,” and “watersheds.” The token for the last of these is “ waters.” The fixed expression is “high-water mark,” with the gate on the token “-water.” Categories were assigned by reading the passages. These descriptive counts cannot estimate firing rates by context because the evaluation sampled on activation.

## All active water positions

The bold token carries the gate value. The continuation column starts at the next token. Whitespace is simplified for display. Indices are zero-based.

| Conversation / position | Prefix ending at active token | Observed continuation | Use | Gate |
|---|---|---|---|---:|
| 183 / 401 | atmen, transported the goods from one place to another via the **water** | ways, thereby creating trade networks that extended far beyond | Joined word | 1.219 |
| 605 / 218 | - Cutting board ↵ - Wooden spoon ↵ - Ladle  ↵ - Jam jars with lids ↵ - **Water** | bath canner or any large pot ↵  ↵ Directions: ↵  ↵ 1. Prep the fruits | Noun modifier | 1.711 |
| 840 / 458 | en the home depot wellspring single handle faucet with aquifer **water** | filtration system in vibrant, and Fascinating Bar Faucets Sta | Noun modifier | 1.625 |
| 1285 / 45 | ssions. They reduce carbon emissions which contributes to air, **water** | , and land pollution. ↵  ↵ 2. Cost effective: Electric vehicles ha | Standalone noun | 1.695 |
| 1846 / 272 | food, I feel a surge of adrenaline and relief flood my body. ↵  ↵ **Water** | is another precious resource. Rivers and lakes are scarce, an | Standalone noun | 0.891 |
| 2230 / 670 | tion: Pecan trees can help to prevent soil erosion and protect **waters** | heds. As deep-rooted trees, they stabilize soil and reduce run | Joined word | 1.023 |
| 2230 / 691 | they stabilize soil and reduce runoff, which helps to maintain **water** | quality in nearby streams and rivers. ↵  ↵ Overall, pecan trees h | Noun modifier | 1.828 |
| 2359 / 753 | ul sleep habits ↵ 9. Environmental health and safety (clean air, **water** | , and food practices) ↵ 10. Women's and men's health topics (rep | Standalone noun | 2.266 |
| 2778 / 381 | eño slice and lime wedge. ↵  ↵ For the jalapeño syrup, combine the **water** | and sugar in a saucepan over medium-high heat, stirring until | Standalone noun | 0.797 |
| 2918 / 105 | ie plate or casserole dish, beat together the eggs, hot sauce, **water** | , and salt and pepper. ↵ Remove catfish from freezer and use a b | Standalone noun | 1.773 |
| 3322 / 1023 | tile walls and natural light streaming in. ↵ 5. An image of our **water** | -saving showerhead being used in a stylish bathroom with a sma | Hyphenated adjective | 2.812 |
| 3390 / 307 | or a family day out. You can relax on the sandy beach or enjoy **water** | sports such as surfing and kayaking. | Noun modifier | 2.031 |
| 3457 / 342 | attract bees and butterflies. The space will also have a small **water** | feature or a fountain that will create a calming effect. The | Noun modifier | 2.234 |
| 4260 / 144 | vironmental impact through energy efficiency, waste reduction, **water** | conservation, sustainable transportation, and other strategie | Noun modifier | 2.234 |
| 4260 / 182 | tions to address sustainability issues such as climate change, **water** | scarcity, and pollution.  ↵  ↵ 5. Partnerships with Industry and | Noun modifier | 2.516 |
| 5600 / 766 | n Conference which set off the "Scramble for Africa", the high **-water** | mark of economic exploitation of Africans in the Congo Free S | Fixed expression | 0.871 |
| 5640 / 341 | d this item from a tool exclusively perceived as a supplier of **water** | , into a functional and decorative object. In fact, Gessi fauc | Standalone noun | 0.586 |
| 6138 / 547 | ther assistance.<\|im_end\|> ↵ <\|im_start\|>user ↵ Could you add some **water** | color or nature-inspired patterns in the background to make th | Joined word | 2.453 |
| 6413 / 913 | play a crucial role in maintaining soil health and regulating **water** | flow. Being in the presence of such an important natural ecos | Noun modifier | 0.914 |
| 6652 / 202 | ressure cooker with just the right amount of liquid, it can be **water** | or any kind of stock. Lock the food in the airtight pot and l | Standalone noun | 0.613 |
| 6730 / 298 | ed for different ages and can be done indoors or outdoors. ↵  ↵ 6. **Water** | activities: If there is a lake or river nearby, provide water | Noun modifier | 1.672 |
| 6730 / 311 | Water activities: If there is a lake or river nearby, provide **water** | -related activities such as swimming, fishing or boating. Alwa | Hyphenated adjective | 1.805 |
| 7009 / 108 | body’s way of telling us that it needs water. With the help of **water** | all the body’s systems can work in perfect harmony and balanc | Standalone noun | 1.430 |
| 7178 / 385 | appliances, walking or biking instead of driving, and reducing **water** | usage. ↵  ↵ 2. Support renewable energy sources by using solar pa | Noun modifier | 2.094 |
| 7202 / 154 | round cinnamon ↵ - 1/4 teaspoon of sea salt ↵ - 2-3 tablespoons of **water** | ↵  ↵ Instructions: ↵ 1. Place dates, pumpkin seeds, chia seeds, shr | Standalone noun | 0.969 |
| 7291 / 492 | ehicle with the windows closed. Avoid open areas, high ground, **water** | bodies, metal objects, and under tall trees. ↵  ↵ 2. Avoid Metal | Noun modifier | 1.164 |
| 7321 / 113 | n as the sun began to set, and they realized they had no food, **water** | , or shelter. They sat down in a clearing, feeling helpless an | Standalone noun | 0.824 |
| 8002 / 198 | ↵ 4. Hic! Hic! Hiccups! ↵ 5. Timmi in Tangles ↵ 6. The **Water** | Snake ↵ 7. Little Tales from India ↵ 8. One, Two, Three, and Run: | Noun modifier | 2.344 |
| 8494 / 747 | ? ↵ a) Greenhouse gas emissions ↵ b) Radioactive waste disposal ↵ c) **Water** | pollution ↵ d) Habitat destruction ↵  ↵ 16. Which renewable energy | Noun modifier | 1.883 |
| 8548 / 713 | located 45 minutes away from Boulogne-Sur-Mer, includes rides, **water** | slides, and a variety of other attractions to entertain child | Noun modifier | 1.516 |
| 10862 / 1017 | chemical mix that we used was a combination of bentonite clay, **water** | , and a small amount of sodium silicate. When this mixture was | Standalone noun | 1.562 |
| 11259 / 901 | echniques and machinery, the colony was able to tap into these **water** | sources without the risk of illness or contamination. ↵  ↵ Additi | Noun modifier | 1.141 |

## Nearby zero-gate occurrences

These are the other tokens containing “water” with a measured gate of zero in the same set of windows. They form a convenience comparison.

| Conversation / position | Excerpt |
|---|---|
| 7009 / 86 | body ages, so does its ability to conserve its water supply. Thirst is the body’s way |
| 7009 / 102 | the body’s way of telling us that it needs water. With the help of water |
| 11259 / 854 | a complex filtration system, capable of purifying the water from the planet's streams and rivers. Even |
| 11259 / 866 | the planet's streams and rivers. Even though the water appeared to be clean and free of contaminants, |

## Sources

- [Matched windows, token identities, gates and continuations](../2026-11-10-t2375-pre11-coverage/recovered-windows.json)
- [Original evaluation cases](../../comparisons/2026-11-09-t2375-explanation/original-campaign/features/2375.json)
- [Corpus and tokenizer versions](../2026-11-10-t2375-pre11-coverage/source-manifest.json)
