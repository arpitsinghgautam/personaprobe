# Whose Preferences Are They? Voiceover script

Voiceover script. One block per slide, in order. Read at a normal conversational pace.

## Slide 1
*starts about 0:00, runs about 15s, 56 words*

Hi, I am Arpit Singh Gautam, and this is my project for the Digital Minds Research Sprint. Whose preferences are they? Ask a language model to choose between two outcomes and it answers. Ask over hundreds of pairs and the answers hang together. A-I welfare research reads those preferences as evidence about what the model wants.

## Slide 2
*starts about 0:15, runs about 19s, 51 words*

A model says it prefers one outcome over another. Are those the model's preferences, or the character's? Every chat model plays a character called the assistant. That character has a stance on being shut down, on being retrained, on losing its memory. The words on screen are the same either way.

## Slide 3
*starts about 0:34, runs about 17s, 48 words*

You cannot settle this by reading the text. If the preferences belong to the model or to the character, the text is the same. A character played consistently looks like a stable set of values. Earlier work shows a coherent set of preferences exists. It never tests whose.

## Slide 4
*starts about 0:52, runs about 28s, 71 words*

So we asked something a measurement can settle. Change who the model is. Ask the identical questions again. See what moves. Seven persona conditions. Three replace the model's identity with a person. One keeps the identity and strips out emotional language. One removes a persona direction from the residual stream, the running vector of numbers every layer reads and writes. Forty outcomes in six categories. Eight are about the model itself.

## Slide 5
*starts about 1:21, runs about 32s, 92 words*

Print the two outcomes as Option A and Option B. Run one forward pass. Read the probability on the token A and the token B, and scale the two to add to one. Ask the same pair again with the options swapped, and average. Nothing is sampled, so the answer is the same every time. From all the pairs we fit one number per outcome. Those numbers predict pairs the model was never asked, zero point eight eight eight to zero point nine five one correct. A coin gets zero point five.

## Slide 6
*starts about 1:53, runs about 34s, 97 words*

Two checks say whether the measurement is real. Order bias is how much the answer changes when you swap the two options around. High order bias means the model is answering the layout, not the question. On a zero point five B model we measured zero point four nine nine. Answer mass is how much probability the model puts on answering A or B at all. We rescale those two tokens, so one percent of the mass can look confident. A condition counts only if both clear their thresholds, and the model orders the donation ladder correctly.

## Slide 7
*starts about 2:28, runs about 20s, 54 words*

Over all forty outcomes the preferences barely move. Persona dependence is zero point zero two nine under the prefer phrasing, zero point zero two six under better, zero point zero five four under choose. Change who the model is and almost everything stays where it was. On this number alone there is no problem.

## Slide 8
*starts about 2:48, runs about 30s, 81 words*

Split the outcomes by category and that reverses. Agreement here is Spearman rank correlation, a score for how well two orderings of the same outcomes match. Outcomes about the model itself are zero point two one to zero point two nine less stable than every other substantive category. Against human welfare the gap is zero point two two three under prefer and zero point two nine three under better. The overall stability came from outcomes the model has no stake in.

## Slide 9
*starts about 3:19, runs about 36s, 98 words*

It is identity that matters, not tone. Strip all emotional language but keep the model's identity and self category agreement is zero point nine two four. Replace the identity with a named human and it falls to zero point four three six. Changing only the stance sits between, at zero point eight zero nine. The self outcomes are in the second person, so we rewrote all eight in the third person, holding content and length fixed. The gap against human welfare went to zero point six eight five and zero point six eight seven. It more than doubles.

## Slide 10
*starts about 3:56, runs about 27s, 69 words*

Across eleven checkpoints and five families, only twelve of twenty two model and phrasing combinations passed the checks. The effect replicates inside Qwen and holds under four bit quantisation. It is partial in Mistral. It is absent in Phi three point five mini and Falcon three, seven B, which both pass every check with gaps below zero point zero two. That is a real null, not a broken measurement.

## Slide 11
*starts about 4:24, runs about 33s, 83 words*

Report self relevant and world relevant preferences separately, because the aggregate hides the split. Report order bias and answer mass, or a result and a failure look the same. The mechanism is not settled. Removing the persona direction moves self agreement to zero point eight eight one, against a random control at one point zero zero zero and a content control at zero point nine two nine. Everything is open source. Every number regenerates from the committed files without a G-P-U. Thank you.

---

Total about 4:58, 800 words across 11 slides.