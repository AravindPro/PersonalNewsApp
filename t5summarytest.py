# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
# %%
import torch
text = """The autonomous vehicle industry is in trouble and is looking for help from an unlikely source: US Transportation Secretary Pete Buttigieg.

A coalition of lobbying groups sent Buttigieg a letter last week imploring him to support AV development or risk being outpaced by China. The subtext, of course, is the crisis facing GM subsidiary Cruise, which grounded its fleet recently in the aftermath of a crash involving one of its driverless cars.

The letter makes no mention of the situation with Cruise, instead focusing on the looming threat of competition from China. Signatories include the Autonomous Vehicle Industry Association, which represents Cruise, Waymo, Zoox, Motional, and others, as well as the US Chamber of Commerce and the Alliance for Automotive Innovation, which lobbies for the auto industry.

Related
Pete Buttigieg is racing to keep up with self-driving cars
“The Department’s support for AV development is crucial to maintain our nation’s competitive edge and secure our position as a global leader,” the groups write. “The U.S. stands at a critical juncture in the AV race, with countries like China aggressively investing and advancing the technology.”

Self-driving cars are safer than human drivers “because they eliminate human errors like fatigued, impaired, and distracted driving,” the groups said. “The need to dramatically improve road safety has never been higher, and we need to take an ‘all-of-the-above’ strategy that includes AVs alongside other safety measures.”

The federal government has largely taken a back seat to regulating autonomous vehicles, leaving states to develop their own rulebooks for safe deployment. Legislation that would dramatically increase the number of AVs on the road has been stalled in Congress for over six years, with lawmakers at odds over a range of issues, including the number of exemptions from federal motor vehicle safety standards.

A potential stopgap solution called AV STEP was proposed by the National Highway Traffic Safety Administration earlier this year. This program could allow the agency to authorize more vehicles without traditional controls, like pedals and steering wheels, without hitting the annual cap on the number of exemptions. In its letter, the AV industry implores Buttigieg to start the rulemaking process on AV STEP.

NHTSA had said it would start the rulemaking process on AV STEP in the fall, though it seems to have missed that deadline. A spokesperson for the US Department of Transportation did not respond to a request for comment.

The situation with Cruise has exposed serious trust issues with AVs. California regulators accuse Cruise of withholding video footage showing its driverless vehicle dragging a pedestrian following a hit-and-run. (Cruise denies this.) GM CEO Mary Barra recently told investors that the company must “build trust” with communities, including first responders, before deploying its cars back on the road.

It’s hard to imagine Buttigieg or NHTSA continuing to embrace AVs at a time when public opinion toward self-driving cars seems to be souring. NHTSA has opened an investigation into the Cruise incident, and Buttigieg has publicly said that the government needs to ensure that AVs are safe before they are deployed at scale.
"""
inputs = tokenizer.encode(text, return_tensors='pt')
# %%
output = model.generate(inputs, num_beams=2, max_length=300, early_stopping=True)
print(tokenizer.batch_decode(output))

# %%
