# %%
from rank_bm25 import BM25Okapi
import pandas as pd
import numpy as np
# df = pd.DataFrame(columns=['array_column'])
# numpy_array = np.array([1, 2, 3, 4, 5])
# df.loc[0] = [numpy_array]
# df.to_csv('temp.csv')
from sentence_transformers import SentenceTransformer
EMBEDDING_MODEL = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2', cache_folder='./cache')


# %%
title = """Lord Ram has chosen PM Modi to build his temple: L.K. Advani"""
content = """Calling Prime Minister Narendra Modi an annanya bhakt (devoted disciple) of Lord Ram, Bharatiya Janata Party stalwart L.K. Advani said he was “just the charioteer of the Ram Rath Yatra” that began on September 25, 1990 to garner support for the construction of a Ram Temple in Ayodhya.


ALSO READ

Ram temple consecration ceremony invitation extended to President Droupadi Murmu
Speaking to Hindi magazine Rashtra Dharma, which is backed by the Rashtriya Swayamsewak Sangh (RSS), Mr. Advani referred to the “Ram Rath Yatra” which he took out 33 years ago “as the most decisive and transformative event in his political journey”, which allowed him to “rediscover India and, in the process, reconnect with himself”.

“Today the Rath Yatra completes 33 years. When we started the yatra on the morning of September 25, 1990, we did not know that the faith in Lord Ram with which we were starting this yatra would take the form of a movement in the country,” Mr. Advani was quoted as saying.

Also read | PM Modi begins special 11-day ritual before opening of Ram Temple in Ayodhya

The former Deputy Prime Minister said Mr. Modi, who was his then assistant, wasn’t very famous when the Rath Yatra was being taken out. Mr. Modi had remained with Mr. Advani throughout the Rath Yatra and the BJP veteran feels that at that very time “Lord Ram himself had chosen Mr. Modi, his devoted disciple, to build his Temple in Ayodhya”.

He also remembered former Prime Minister Atal Bihari Vajpayee in the article and said that his absence is being felt at the moment. “It was during the Rath Yatra, I felt that destiny had decided that one day a grand temple of Shri Ram would be built in Ayodhya,” Mr. Advani said, adding, “Now it’s only a matter of time.”

ALSO READ

Analysis | Different voices emerge in Congress despite party’s ‘no’ to Ram Temple pran pratistha
He was further quoted as saying that when the PM consecrates the Ram Lalla idol in the temple, he will represent every citizen of India. He further prayed that “the Ram Temple inspires all Indians to adopt the qualities of the Lord”.

Titled as “Ram Mandir Nirman, Ek Divya Swapna Ki Purti”, the article will be published in the January 16 edition of the magazine and copies would be shared with all those who will attend the consecration ceremony in Ayodhya.

According to the Vishva Hindu Parishad (VHP), the 96-year-old leader will attend the consecration ceremony in Ayodhya. His close associates told The Hindu that no confirmation on his travel to Ayodhya can be given at this point of time."""

# %%
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
def similar(c1, c2):
	e1 = EMBEDDING_MODEL.encode(c1)
	e2 = EMBEDDING_MODEL.encode(c2)
	print(cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1)))

# %%
t1 = "Farmers convention to chart action plan to defeat BJP"
t2 = """Back Back
What is Rabbit R1? All you need to know about the AI-powered smart assistant launched at CES 2024"""
t3 = "CES 2024: Rabbit's LAM pulls R1 out of the hat"
t4 = "Sri Lankan refugees | The long wait for Indian citizenship"
t5 = "Ram temple consecration ceremony invitation extended to President Droupadi Murmu"
t6 = "16 highlights from the 2024 Consumer Electronics Show"
t7 = "CES 2024: Transparent TV, dual-screen laptop, AI robot, pillow and 10 other cool gadgets at the biggest tech show of the year"
# %%
c2 = """Santa Monica-based startup Rabbit has announced the new Rabbit r-1 AI assistant at the Consumer Electronics Show (CES 2024) in Las Vegas. The R1 is an AI-powered device that features a 2.88-inch touchscreen display and runs on a MediaTek processor. The AI assistant can perform a variety of functions, including playing music, ordering groceries and sending messages.

Also Read |CES 2024 Day 3 highlights: From smart rings to headsets, everything announced
Explaining the idea behind Rabbit R-1, rabbit founder and CEO Jesse Lyu said, "We’ve come to a point where we have hundreds of apps on our smartphones with complicated UX designs that don’t talk to each other. As a result, end users are frustrated with their devices and are often getting lost. rabbit is now building towards an intuitive app-free experience with the power of AI."

“Large Language Models, like ChatGPT, showed the possibility of understanding natural language with AI; our Large Action Model takes it one step further: it doesn’t just generate text in response to human input – it generates actions on behalf of users to help us get things done."

As mentioned earlier, the Rabbit R1 features a 2.88-inch touchscreen and is run on the company's proprietary rabbit OS. Interestingly, unlike much of AI-based offerings which are based on large language models (LLM) like OpenAI's GPT or Google's Gemini, the Rabbit R1 instead is based on a ‘Large Action Model’ and has the ability to gauge users' intention and behaviour while using specific apps and mimicking those actions ‘reliably and quickly’, Rabbit claims.

How to pre-order Rabbit R1? 
The Rabbit R1 is priced at $199 and is available to pre-order from the company's official website: rabbit.tech. The company is planning to start shipping the smart devices by late March. Rabbit has announced that it has sold out its 2 batches of 10,000 rabbit r1 devices and the third batch is now available to pre-order from its website which are expected to be delivered around May to June 2024. """

c1 = """The national convention of the Samyukta Kisan Morcha (SKM), an umbrella organisation of more than 500 farmers’ outfits, will be held on January 16 in Jalandhar. The meeting will discuss alternative agriculture and industrial policies ahead of the Lok Sabha elections. The farmers’ leaders will also chart strategies to defeat the BJP in the polls.


The SKM said in a release that around 1000 delegates of the constituent organisations from across the country will participate in the convention. “The Convention is to expose the pro-corporate economic policies of the BJP-led Narendra Modi Government, detrimental to farmers and workers and people at large, causing large scale unemployment, price rise, poverty, indebtedness, and unbridled rural migration,” a statement said here on Friday.

ALSO READ

Farmers sit-in protest continue for second day in Punjab, Haryana
The meeting will also discuss the development narrative of the Centre based on GDP rate and the claims of the country becoming a three trillion-dollar economy. The SKM said such claims hide the decline in the per capita income, growing income inequality and denial of minimum support price to farmers and minimum wage to workers.

The SKM said two years have passed since they withdrew the agitations in Delhi, but the Prime Minister has not made good on any of the written assurances made to farmers including implementation of the M.S. Swaminathan Commission‘s recommendations, waiving the debt of farmers and farm workers, stopping privatisation of electricity and placing of smart meters on farms and in farmers households.

“Farmers who faced natural disasters that ruined crops did not get insurance and adequate compensation in large parts of the country since the Pradhan Mantri Fazal Bima Yojna functions for corporate profiteering rather than protecting farmers and agriculture,” the statement said."""
c3 = """AI startup Rabbit Inc. has unveiled the Rabbit R1 at CES 2024, and it's creating quite a buzz. This pocket-sized device promises to shake up the personal AI assistant space, changing how we interact with technology.

"Smartphones were supposed to be intuitive," says Rabbit Inc. CEO and Founder Jesse Lyu during the R1's presentation. The reality is they are not, over time your smartphone has become an aggregator of apps, with all your services tied to 100 different apps that you have to thumb through to get what you need.

This is the opportunity that Rabbit saw, a space for a middle-man-esque service for your digital life, a sort of concierge that can manage all your services for you."""
c4 = """Ganesan, 69, who lives in a camp for refugees from Sri Lanka, in Irumboothipatty village in Karur district of Tamil Nadu, describes November 30, 2023, as a day of “rebirth”. It was on this day that the Madurai Bench of the Madras High Court recognised him as an Indian. “I just couldn’t believe it,” he says, his subdued tone masking his joy. “I had been waiting for that moment for more than 30 years. It is as if I have got a second life.”

The Government of India considers all refugees in India “illegal migrants” and India does not have a law governing refugees. This means that Sri Lankan refugees, who came to India fearing violence and persecution during the civil war, which lasted from 1983 to 2009, are ineligible for Indian citizenship even if they have lived in India, like Ganesan has, for over 30 years. Camps in Tamil Nadu house 58,457 refugees (as on March 31, 2023). There are also 33,375 refugees living outside these camps.

"""
c5 = """President Droupadi Murmu on January 12 recieved the invitation for the consecration of Ram Temple in Ayodhya scheduled for 22nd January.

The invitation to the President was given by Nripendra Misra, Chairman, Ram temple construction committee and working president of Vishwa Hindu Parishad (VHP) Alok Kumar. The two were also accompanied by senior leader of Rashtriya Swayamsevak Sangh (RSS), Ram Lal.

ALSO READ

Ayodhya | Building a city around a temple
The president expressed immense joy on recieving the invitation and said that she will soon decide the time to come and visit Ayodhya, said a communique from the VHP.

The right wing organisation, which was at the forefront of the Ram temple movement, maintained that arrangements will be made for all guests invited for the temple inauguration.

As per VHP, over 7000 people from 150 categories are invited for the event which includes seers, politicians, businessmen, sportsmen, actors, families of Kar Sewaks etc."""
c6 = t6
c7 = """The 2024 Consumer Electronics Show (CES) had some really cool gadgets on display. Out of all the things showcased, there were quite a few gadgets that stood out. The show included futuristic TVs with amazing resolutions and immersive features, groundbreaking laptops that are both powerful and portable, and innovative robots with unique capabilities. The show also had smart home devices that are revolutionising the industry, and advancements in virtual reality that were mind-blowing. These are 10 of the cool and quirky gadgets that stole the show at CES 2024 and grabbed our attention.

by TOI Tech Desk
02/15LG Signature OLED T TV

LG Signature OLED T is a 77-inch transparent OLED TV. It can be fully transparent or adjusted like a traditional TV. When turned off, it can blend in with your home or be used as a fish tank, fireplace, or even a real-time window. It also has a holographic mode and can be placed in front of a window. Signature Series T can raise a black film for a traditional OLED viewing experience.

by Abhinav Kaustubh
03/15Samsung Music Frame

Samsung’s Music Frame is a wireless speaker that functions as a photo frame as well. This speaker is Dolby Atmos-compatible and equipped with SpaceFit calibration technology. You can use it as a TV speaker, rear speaker or subwoofer by pairing it with a Samsung TV or soundbar. The Music Frame also has the option to pair two speakers or connect up to five for synchronous play. It includes an IoT hub for smart home connectivity and comes with a Diasec matte acrylic plate Art Panel for a more professional and durable display.

04/15Asus Zenbook Duo

​Asus' Zenbook Duo is a dual-screen laptop with an OLED panel that is slightly larger than the competition, offering a 3K resolution, 120Hz refresh rate, and stylus support. It is powered by an Intel Core Ultra 9 185H CPU, up to 32GB of RAM, and a 1TB SSD. The Zenbook Duo is portable, measuring just 0.78 inches thick and weighing 3.64 pounds, and comes with a built-in kickstand that promotes a stacked setup. The keyboard can be stored inside the system, or placed in front of the system to double screen space.

05/15Lenovo ThinkBook Plus Gen 5 Hybrid

The ThinkBook Plus Gen 5 Hybrid from Lenovo combines a laptop and an Android tablet. The "Hybrid Station" base has an Intel Core Ultra 7 processor, Intel Arc GPU, up to 32GB of RAM, and a 1TB SSD. The detachable "Hybrid Tab" has a 14-inch, 2.8K OLED display that can function as a Windows 11 display or an Android tablet with Snapdragon 8+ Gen 1, 12GB of RAM, and 256GB of storage. The two systems don't share files. The Station can connect with an external monitor.

06/15MSI Claw

The first-ever gaming handled of MSI, the Claw, is a Windows gaming handheld, which comes equipped with Intel's Core Ultra chip, 16GB RAM, and up to 1TB of PCIe M.2 storage. The handheld has a 7-inch Full HD display with a 120Hz refresh rate, macro buttons, thumbsticks with RGB lights, and a 53Wh battery. It runs on Windows 11 and can run PC and Android games via its App Player.

07/15LG AI Agent

LG's AI Agent is a robotic home manager that anticipates and caters to its owner's needs. It evaluates and reports on your home's temperature, humidity, and air quality. With a two-legged wheeled design, the robot can travel through your home and respond to specific needs. It can also maximise your home's efficiency by patrolling for any appliances left on unnecessarily. The AI Agent greets users at the front door, assesses their mood through vocal and facial expressions, and plays music to suit their mood. It connects to smart appliances, lights, and outlets, learns your routines, and reports on the weather. LG intends for its AI Agent to "liberate users from the burden of housework" by automating your home's processes through learning from your daily life.

08/15Samsung Ballie

Samsung unveiled a new robot called Ballie that comes with a built-in projector. The robotic sphere will be available for purchase within the year. Ballie has a spatial LiDAR sensor and a 1080p projector, which can display movies, video calls, and greetings on surrounding surfaces. It can be controlled by voice commands and text messages and can send clips of what it sees. The robot is capable of automatically turning connected lights and gadgets on or off, as well as non-connected appliances with its infrared transmitter.

09/15Sony’s mixed-reality headset

Sony showed off a new mixed-reality headset at CES for spatial content creators. The standalone headset has a matte grey finish and dual 4K OLED microdisplays, with a flip-up visor and two front-facing cameras. It is powered by Qualcomm's Snapdragon XR2+ Gen 2 and has six cameras and sensors for user and space tracking. The headset comes with two controllers, and Sony wants creators to design 3D models in real time. It is scheduled to be released later this year, and pricing details are yet to be announced.

10/15Withings BeamO

Withings has launched the BeamO, a four-in-one health monitoring device that measures body temperature, blood oxygen saturation, provides medical-grade ECG readings, and serves as a digital stethoscope to monitor heart and lung functions. The device alerts users for signs of major health issues like AFib, asthma, COPD, among others. Users can log their readings and share them with healthcare professionals via the Withings app. The device has an eight-month battery life and supports multiple user profiles for an entire household.

11/15Rabbit R1

The R1 from Rabbit is a device powered by a natural-language operating system. R1's "Large Action Model" AI foundation learns a user's typical app and web experiences and reproduces them on a customised cloud platform. Rabbit hole is a private web portal that enables users to sign into their accounts, grant permissions, and more. Rabbit OS performs tasks with user permission, without storing their identity information or passwords. The R1's security extends to its industrial design, which features a downward-facing "Rabbit eye" camera and a microphone that activates when the side-mounted push-to-talk button is pressed. The device has a 2.88-inch touchscreen display, a scroll wheel for OS navigation, and a USB-C port for charging. It also has a 1,000mAh battery, 128GB of storage, and a SIM slot for mobile data.

12/15Razer Project Esther

Razer's Project Esther is the world's first HD haptics gaming chair cushion with 16 haptic actuators, ultra-low latency, and automatic audio-to-HD haptics conversion. It can be controlled for directionality, multiple-device integration, and multi-actuator experiences. Compatible with most gaming and office chairs, it provides an immersive experience that can be combined with a VR headset.

13/15Motion Pillow

The Motion Pillow is an AI-powered to help individuals dealing with snoring problems to get a better quality of sleep. The AI Motion System incorporated in the pillow is capable of detecting snoring patterns and inflates airbags slowly, which in turn lifts the user's head, thus opening their airway and reducing snoring. The accompanying app helps track and analyse sleep data, including snoring time, airbag operation time, sleep score, sleep time, and even recordings of your snoring. This information can be played back later to better understand your sleep patterns.

14/15Philips Smart Deadbolt

Philips showed off the Wi-Fi Palm Recognition Smart Deadbolt on the floor. The lock has a built-in palm scanner that unlocks the door when it detects your unique palm print. It can register up to 50 palm vein patterns and also allows you to use a PIN or a key if preferred. The device doubles as a doorbell and has proximity sensors that detect when you're reaching for the doorknob, automatically unlocking the door as you approach it. The lock is also compatible with Amazon Alexa and Google Assistant.

15/15Sony Afeela

Sony showcased their Afeela EV prototype car at CES 2024, which has a massive ultrawide dashboard screen and a bumper screen called the Media Bar. The car uses Unreal Engine 5.3 for displaying graphics, and it can also be used for playing games, watching movies, and AR navigation. Sony demoed a new game called Monster View, which encourages drivers to catch virtual creatures while driving. The Afeela may have features like "Smart Entry," Advanced Driver Assistance Systems, Spatial Audio support, and a noise-cancelling system."""
c8 = """The R1, the pocket-sized AI gadget from Rabbit that’s supposed to use your apps for you, has already sold out of its first batch — and its second batch, too.

In a Wednesday post on X (formerly Twitter), the startup Rabbit announced that it sold through its first 10,000-unit production run in just one day. “When we started building r1, we said internally that we’d be happy if we sold 500 devices on launch day,” Rabbit writes. “In 24 hours, we already beat that by 20x!” The first batch of preorders is expected to start shipping in March.

When we started building r1, we said internally that we'd be happy if we sold 500 devices on launch day. In 24 hours, we already beat that by 20x!

10,000 units on day 1!

Second batch available now at https://t.co/R3sOtVWoJ5
Expected delivery date is April - May 2024. pic.twitter.com/XqaHqqk36L

— rabbit inc. (@rabbit_hmi) January 10, 2024
The company put up a second batch for preorder, but on Thursday, that run of 10,000 devices sold out as well. A third batch is now available with an expected delivery date between May and June of 2024.

Rabbit introduced the R1 during CES on Tuesday, which comes with a small 2.88-inch touchscreen that runs on the company’s own Rabbit OS. It uses a “Large Action Model” that works as a “sort of universal controller for apps,” according to my colleague David Pierce, who got to try out the device during the showcase. This allows it to do things like play music, buy groceries, and send messages through a single interface without having to use your phone. It also lets you train the device how to interact with a certain app.

"""
c9 = """During CES 2024, Santa Monica-based AI startup rabbit introduced r1, a handheld and touchscreen artificial intelligence device that works like a walkie-talkie with camera functionality. The main premise of the r1 is that it does what any app can do without needing a smartphone. Want to arrange a flight? Done. Planning to order lunch or dinner? No problem. Need to book a taxi? Why not?

 

All the user needs to do is press the button on the side of the AI device, talk to the card-sized walkie-talkie about their request, and wait for the machine to get it all done. r1, in a nutshell, wants an app-free online experience by packing everything an app performs into one portable touchscreen device. rabbit’s r1 is mainly powered by two settings: its in-house OS called rabbit OS and the so-called Large Action Model or LAM, which is its system designed to make AI see and perform apps in the same way humans do, such as buying items and completing purchases.

 

It is also because of this LAM that rabbit’s r1 removes the need for users to download and use multiple apps since it and the OS navigate an app’s functionality on their own, like a personal assistant running errands. rabbit says that because of its OS and LAM, the AI walkie-talkie can understand and perform tasks more than just switching on and off lights, playing music, and giving weather updates.

rabbit r1 ai device ces 2024images and video stills courtesy of rabbit

 

 

R1 CO-DESIGNED WITH TEENAGE ENGINEERING
 

rabbit’s r1 and its AI will not work without an engineered and user-friendly body. To do that, the startup taps the design firm teenage engineering, whose portfolio ranges from synthesizers to audio devices and more. For r1, teenage engineering draws its design influences from the retro days of Tamagotchi and the portability of Post-it notes. Its luminous orange streak brightens up the device and catches attention right off the bat, an ideal setting for its 2.88-inch touchscreen display, 115-gram weight, push-to-talk button, and a scroll wheel of the same color used to navigate through options.

 

On the right side of the AI device, just above the scroll wheel, there is a camera named rabbit eye. May the viewers not be stunned once the camera flexes its ability to rotate and twist in 360-angle to capture and document the surroundings. Because of this feature, the user can take work video calls using r1 and can ask the AI device to scan and look up for them using physical objects in their surroundings. Take the remaining food in the user’s refrigerator as an example. The user just needs to hold up r1 and point the camera toward the refrigerator, and the AI device will soon suggest recipes using their leftover meals and remaining ingredients.

rabbit r1 ai device ces 2024rabbit’s r1 is an AI device able to execute tasks of potentially any smartphone app, now showcased at CES 2024

 

 

PRIVACY GUARANTEED WITH RABBIT’S AI DEVICE R1?
 

rabbit wants to highlight that r1 is a standalone handheld device that does not need a smartphone to be managed or for it to function. For specs fans, the AI device has both WiFi and cellular connectivity options since the user can insert their own SIM card in the dedicated slot. Inside its teenage engineering-designed casing, r1 boasts a 2.3 GHz MediaTek Helio P35 processor, 4 GB of memory and 128 GB of storage, and a USB-C port. rabbit claims that the battery charge can last for a day too.

 

Those who are worried about their privacy, rabbit says that they have the option to delete their stored data at any time and that they will also be informed and have control over which types of action are being delegated to the system. The startup assures them too that r1’s hardware does not have an always-active listening mode, that its microphone will not record unless the button on the side is pressed, and that its rotating camera’s default position is with its lens blocked.

 

Visitors to CES 2024 can experience rabbit’s r1 first-hand when they swing by their booth. In case interested fans are not able to make it, rabbit has already opened its pool of pre-sales starting January 9th, and as of January 10th, rabbit announced on X that its first batch sold 10,000 units already with the second batch now available for reservation. Each AI device is priced at 199 USD, and rabbit says that there’s no additional monthly subscription required. For US orders, shipping starts late March 2024 while global orders are sent out later this 2024.

"""
# 0.45 is a good criterion for similar in titles
# 0.6 is a good similarity criterion given the full content
# euclidean isn't that good a similarity method
# %%

# BM25 algo
from rank_bm25 import BM25Okapi
corpus = [t1, t2, t3, t4, t5, t6, t7]
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
# %%
query = t3
tquery = query.split(" ")
print(bm25.get_scores(tquery))
# %%
import pandas as pd
# df = pd.DataFrame(columns=["feedlink",])
df = pd.read_csv('data/rss.csv', index_col=0)
# data = [{"feedlink": "https://www.thehindu.com/news/national/feeder/default.rss"}]
# df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
# print(df)
# df.to_csv('data/rss.csv')

# %%
import feedparser
feed = feedparser.parse(
	'https://cdn.technologyreview.com/topnews.rss')


# %%
