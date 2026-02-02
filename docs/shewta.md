



Search Cartedo



Home
1

DMs
2

Activity
3

Files
4

Later
5

More
0

Cartedo






















Shweta Homji





Messages

Add canvas

Files

CanvasListFolder
Poovendhan
  4:50 PM
Hi Shweta, quick follow-up.
After reviewing with Rachit, we’re adjusting the approach to explore chunked / parallel generation for better efficiency, while keeping all locked sections and the overall flow unchanged.
I’m working through this now. Please let me know if it’s okay for me to continue on Ken’s simulation as part of the task.
Shweta Homji
  4:51 PM
@Poovendhan I have my meeting with Ken today, you need to let me know if this will get done by Wednesday 100% or not
4:53
I had already suggested chunked and parallel generation 2 weeks ago. I thought you were already doing that and had drastically reduced time with that approach based on your message on Dec 29th you had brought the generation down to 40-50 seconds. What new results are you expecting? I thought your issue was no longer generation time rather content validitiy and accuracy
Poovendhan
  5:04 PM
The current work is focused on content validity and accuracy, specifically consistent alignment between KLOs, submission questions, and resources, and passing validation checks reliably on a monolithic Prompt .
What I identified now is that validation agents were reprocessing the full JSON when checking accuracy because the JSON chunking was not properly scoped. That caused latency to increase during validation, not generation.
On Dec 29, generation was faster, but accuracy and structural issues remained. I understand that tradeoff clearly now and am fixing it properly.
With this scope, I can commit to completing this by Wednesday.
Shweta Homji
  11:55 AM
Please schedule a meeting at noon-12:30 with Rachit and I tomorrow to show us the process and the full execution of this
Poovendhan
  11:56 AM
Got it Shweta!
Shweta Homji
  12:00 PM
@Poovendhan are you ready to demo - is everything working or do you still need to work on things
Poovendhan
  12:01 PM
Hi Shweta, I want to clearly summarize what was completed and where things went wrong before the meet.
What we completed:
Fixed a critical realism KPI issue where only a subset of KPIs were being evaluated, which was causing extremely low scores.
Removed hardcoded industry assumptions (for example apparel) from generation paths and made industry detection fully dynamic via the factsheet.
Removed hardcoded string replacements that were forcing incorrect domain language.
Reworked realism checking to be tiered and more efficient, with clear pass and fail behavior.
These changes corrected the core domain fidelity and realism correctness issues in the pipeline.
Where it went wrong:
In the final phase, I attempted to optimize latency using chunked generation and later RAG.
That optimization removed some implicit global constraints that were holding alignment together across sections.
As a result, validation alignment dropped into the 70–80 range and latency resurfaced during validation because checks were reprocessing broader context than intended.
I understood the failure mode, but I was not able to restore those global constraints cleanly within the timeline without risking a misleading or unstable output.
I didn’t want to present something that looked complete but wasn’t structurally reliable. That’s why I stopped instead of sharing a partial result. I take responsibility for the miss, and I wanted to be explicit about what was fixed versus what did not land.
Shweta Homji
  12:01 PM
by when can you get this up and running,
Poovendhan
  12:04 PM
can we have a short huddle before the meet i feel nervous to present the current system and have some doubts also . I can have a stable version running by Friday, and I’ll share a clear checkpoint update sooner if anything blocks that.












Message Shweta Homji









Shift + Return to add a new line







Search Cartedo



Home
1

DMs
2

Activity
3

Files
4

Later
5

More
0

Cartedo



























Messages

Add canvas

Files

CanvasListFolder

Search messages in this channel (⌘F)

Shweta Homji
  5:26 PM
Hi @Poovendhan welcome to the team! Settle in and we'll do a whole round to introductions on Friday to introduce you during the team demos
Poovendhan
  5:29 PM
Hi Shweta, thanks a lot! Sure , I’ll settle in and I’m looking forward to work with y'all.
Shweta Homji
  6:40 PM
Please Ask Rachit to set you up on Msty and I'd also like you to set up a few agents for me on N8N - to improve our orchestration. I've got tons of interesting projects for you, just a bit swamped after our GESA win to submit additional things, let's connect on Friday -
Poovendhan
  6:43 PM
Sure shweta I will ask Rachit on that .
6:46
Also congratulations on the GESA win it was really deserved for the team and you.
Shweta Homji
  7:07 PM
Thanks! We're headed for the world championship next month
Poovendhan
  7:27 PM
That's so cool Shweta! we will win the world championship for sure .
Shweta Homji
  12:17 PM
PROMPT 1: learners will act as a junior consultant for an exciting Gen Z organic T-shirts brand, tasked with analyzing the U.S. market and providing a go/no-go market entry recommendation. Using only the simulation’s provided data, they will apply structured frameworks to assess market potential, competition, capabilities, finances, and risks before developing their  final strategy. ( Co-relate with Sample 1 output)
PROMPT 2: students will propose how a fast food brand should respond to its competitor's $1 menu by analyzing the competitor’s move, market impact, strengths, and four strategic options. Their goal is to propose a clear, realistic, and sustainable plan to protect or grow market share via an executive summary.( Co-relate with Sample 2 output)
PROMPT 3: Acting as a  strategic analyst at ThriveBite Nutrition,  learners will assess the viability of launching a new adaptogen-infused functional beverage targeting health-conscious consumers seeking stress relief. They will analyze product-market fit, estimate the market opportunity, benchmark competitors, evaluate internal capabilities, assess financial feasibility, and weigh potential risks using the resources provided. Students must deliver a concise executive summary recommending a go/no-go decision ( Co-relate with Sample 3 output) (edited) 
3 files
 

Download all

SAMPLE 1.pdf
PDF



SAMPLE 2.pdf
PDF



SAMPLE 3.pdf
PDF


You missed a huddleMISSED  12:17 PM
Shweta Homji was in the huddle for 0 m.
A huddle happened  12:18 PM
You and Shweta Homji were in the huddle for 7 m.

Poovendhan
 | Canvas
 
:headphones: Huddle notes: 12/18/25 with @Shweta and @Poovendhan
Canvas
Slack AI took notes for this huddle from 12:20:24 PM - 12:25:43 PM GMT+5:30. Meeting between @Poovendhan and @Shweta focused on project methodology, specifically addressing an LLM-based task for extracting textbook chapters with an emphasis on rapid, lean solution development. View huddle in channel
:handshake: Attendees
@Poovendhan and @Shweta
:star: Summary
Project Approach and Methodology
@Shweta emphasized that in a startup environment, one hour is equivalent to one month in traditional organizations, requiring rapid and efficient work processes.: [2:00]
The primary strategy is to start with the leanest possible solution and then incrementally add validation and complexity.: [3:00]
@Shweta recommended testing prompts across different LLM models to understand potential hallucinations and consistency.: [2:30]
Current Task: Textbook Chapter Extraction
@Poovendhan was initially concerned about potential hallucinations in LLM outputs when extracting textbook chapters.: [1:52]
@Shweta suggested first examining existing textbooks and then using a basic prompt across multiple models to validate results.: [2:00]
The goal is to either find a consistently working solution or document the multiple ways the approach does not work.: [4:25]
:white_tick: Action items
@Poovendhan to test prompt across different LLM models like Gemini, focusing on consistent and quick results.: [6:01]
@Poovendhan to request platform access from Raitt to fully engage with the project.: [5:31]
@Poovendhan to prepare a presentation demonstrating either a working solution or comprehensive documentation of potential issues by the next day.: [4:25]
@Poovendhan to explore the content-to-video project that was previously mentioned.: [3:54]
This tool uses AI to generate notes, so some information may be inaccurate. They're based on the huddle transcript and thread and can be edited anytime.

Huddle transcript
Huddle transcript



A huddle happened  12:25 PM
You and Shweta Homji were in the huddle for 21 m.
Shweta Homji
  12:35 PM
@Poovendhan
Key Learning Outcome - Assessment Criteria
Scenario ( Company info, challenge, Manager/ Avatar)
Type of Activity
Name
Resource
submission questions
Rubric - review question ( 5,4,3,2,1 star definition)
Interaction emails
Guidelines
SCOPE - tomorrow
KLO,  Scenario,
Resource and submission question












Message Shweta Homji









Shift + Return to add a new line







Search Cartedo



Home
1

DMs
2

Activity
3

Files
4

Later
5

More
0
Files


All files

Canvases

Lists
Recently viewed
Starred
Click the ⭐ star on any canvas or list to add it here for later.

:headphones: Huddle notes: 12/18/25 with @Shweta and @Poovendhan


headphones Huddle notes: 12/18/25 with @Shweta and @Poovendhan
Slack AI took notes for this huddle from 12:20:24 PM - 12:25:43 PM GMT+5:30. Meeting between @Poovendhan and @Shweta focused on project methodology, specifically addressing an LLM-based task for extracting textbook chapters with an emphasis on rapid, lean solution development. View huddle in channel
handshake Attendees
@Poovendhan and @Shweta
star Summary
Project Approach and Methodology
@Shweta emphasized that in a startup environment, one hour is equivalent to one month in traditional organizations, requiring rapid and efficient work processes.: [2:00]
The primary strategy is to start with the leanest possible solution and then incrementally add validation and complexity.: [3:00]
@Shweta recommended testing prompts across different LLM models to understand potential hallucinations and consistency.: [2:30]
Current Task: Textbook Chapter Extraction
@Poovendhan was initially concerned about potential hallucinations in LLM outputs when extracting textbook chapters.: [1:52]
@Shweta suggested first examining existing textbooks and then using a basic prompt across multiple models to validate results.: [2:00]
The goal is to either find a consistently working solution or document the multiple ways the approach does not work.: [4:25]
white_check_mark Action items
@Poovendhan to test prompt across different LLM models like Gemini, focusing on consistent and quick results.: [6:01]
@Poovendhan to request platform access from Raitt to fully engage with the project.: [5:31]
@Poovendhan to prepare a presentation demonstrating either a working solution or comprehensive documentation of potential issues by the next day.: [4:25]
@Poovendhan to explore the content-to-video project that was previously mentioned.: [3:54]
Horizontal Rule
This tool uses AI to generate notes, so some information may be inaccurate. They're based on the huddle transcript and thread and can be edited anytime.
Huddle transcript
Transcript

Transcript of huddle in Shweta Homji on 18 Dec from 12:20 PM to 12:25 PM Chennai, Kolkata, Mumbai, New Delhi.

This transcript is auto-generated, so some information may be inaccurate. It won’t be surfaced in search results.
@Poovendhan [1:47]: So I guess he was busy

@Shweta [1:51]: OK. All right, yeah.

@Poovendhan [1:52]: Doing, yeah, so I was doing the scrapping because I thought it was like too much of hallucination and or a validation was not very good for you.

@Shweta [2:00]: So I'll give you a tip, OK? But when the like so that you have to get things done very quickly. This is a startup. We basically one hour is like a one month for in a normal organizations, OK. So if you are even approaching it from a hallucination point of view, you made that assumption. You didn't quantify that assumption before. So first start out with the LLM across different models, you know, give a basic prompt and say this is the name of the textbook. You should be able to look at your own textbooks first, OK? The textbooks that you've used because you can go online

@Poovendhan [2:06]: Got it Yeah Mhm OK

@Poovendhan [2:26]: Got it. Yeah

@Shweta [2:30]: Today and you can look for textbook chapters, right? So the way to approach it is that you first go and look at a textbook and you should be able to see what are the textbook chapters. Then just give a basic LLM, you know, like give a basic prompt and misty across multiple models and see what is the outward. You're assuming there's hallucination. There is no hallucination. I did this prompt in like under 3 minutes, and you can consistently see it, and if there is hallucination, how will you even work on it? So you made it a much

@Poovendhan [2:31]: Mhm But Mhm

@Poovendhan [2:54]: Yeah

@Shweta [3:00]: Larger than like, you first have to do the leanest solution possible, lean, you know, after that you have to add a validation because the projects that I'm about to give you at the speed at which you're working on this one prompt, it'll take you a year. We don't have that kind of time, right? So, what you need to do is first, I'm not saying that we're going to leave LLM's completely on their own. That's not what I'm saying. I'm trying to tell you that first do it, then understand the problem, you what you're solving, like, you need to be able to art

@Poovendhan [3:05]: Got it, yeah Good Got it, yeah.

@Poovendhan [3:24]: OK

@Shweta [3:30]: Ic ul ate the problem. I don't want, right? So, and if I can do a prompt in 3 minutes, which takes you 3 days, there's something completely, then you haven't added any value straightforward, right? So you have to, you, the what I want to be able to.

@Poovendhan [3:31]: Go Yeah But yeah

@Shweta [3:45]: And this helps, that's why like I'm having this conversation with you. What I want you to do right now is I'm giving you today and tomorrow, OK?

@Poovendhan [3:50]: Yeah Yeah.

@Shweta [3:54]: Sit very defined time frame, and you were also working on another project where how about that? Where is that going? like with the content to video? Did you work on that at all?

@Poovendhan [4:04]: No, I still haven't, gone to the, I, I got like brief idea on the on boarding time. The ratchet was telling like disor your future projects will be coming to you. The video and the whisper, the chat integration, so yeah.

@Shweta [4:20]: OK. so this is how I want you to work, OK?

@Poovendhan [4:25]: Yeah

@Shweta [4:25]: Right now, I'm assigning you a a task which I'm giving you only today. Basically, we're gonna present to me tomorrow. It's OK. So when I say present it to me tomorrow, what I'm expecting is that it's done, right, that you've tried, so you should either know that one way that it works, every single time, like I want consistency and or the or you should know 100 ways that it doesn't work. You should be able to articulate the problem. So, but what I'm looking for is the solution. I've already done this, I'm

@Poovendhan [4:30]: Go Good Yes Yeah

@Poovendhan [4:49]: Go

@Shweta [4:55]: Actually giving it to you so that you understand what Cartiro does, OK?

@Poovendhan [4:56]: Yeah Got it. Yeah

@Shweta [5:00]: So I have given you right now. Let me just share my screen. There we go. OK. Can you see my screen?

@Poovendhan [5:10]: Yeah. Yeah

@Shweta [5:12]: OK, so this is the prompt that was given, like till here, like this is the prompt that the professor gave on the basis of which we created all of this, OK, so this is how Cartido works. A professor gives a you should get, do you have access to the platform?

@Poovendhan [5:15]: Yeah Mhm. No, not it, no

@Shweta [5:31]: OK, you need to ask Raitt for the platform. You need to like, or, have, OK, let me just introduce you to the whole team right now on dev. Are you on this dev chat? You should probably be there, OK, I'm gonna introduce you to the team and I'm gonna basically right now because tomorrow is the big reveal, the way I want to reveal a team member is that, hey, he's already come in and we've oriented him and he's working on this and blah blah blah and this is like that's why I was giving you like we gave you very basic projects start off with, but so the

@Poovendhan [5:42]: Yeah

@Poovendhan [5:58]: Got it. Yeah

@Shweta [6:01]: Textbook one, like if you just put first go ahead and do the prompt and see what is the output, OK? Look at it in Gemini look at different models. Your goal is that how can I really quickly deliver something consistently, right? Typically we don't use, chat GPT latest as a model we usually use specific like, static models, not the ones that they keep changing because the ones that are out there, the latest ones, etc.

@Poovendhan [6:08]: Got it. Yeah

@Shweta [6:31]: So you need like, like snapshot models, OK? Even in Misty. Do you know the difference? Hello? We then, I can't hear you. Hello

@Shweta [6:52]: Buyon








Search Cartedo



Home
1

DMs
2

Activity
3

Files
4

Later
5

More
0
Files


All files

Canvases

Lists
Recently viewed
Starred
Click the ⭐ star on any canvas or list to add it here for later.

:headphones: Huddle notes: 1/9/26 with @Shweta and @Poovendhan


headphones Huddle notes: 1/9/26 with @Shweta and @Poovendhan
Slack AI took notes for this huddle from 11:58:29 AM - 12:22:14 PM GMT+5:30. Shweta provides detailed guidance to Poovendhan on developing a systematic approach to validating AI simulation recontextualization, emphasizing the need for clear, measurable, and transparent performance reporting. View huddle in channel
handshake Attendees
@Shweta and @Poovendhan
star Summary
Methodology for AI Simulation Validation
@Shweta outlined a three-step approach to developing a robust validation process for AI simulations [10:56].
The primary goal is to achieve 100% accurate recontextualization without arbitrary time or run limitations [10:56].
Each agent must have clearly defined tasks, threshold values, and specific performance metrics [14:42].
Reporting and Accountability
@Shweta emphasized the need for a dashboard-style report that provides quick, actionable insights [16:31].
Reports should include pass/fail columns, reasons for failures, and suggestions for improvement [17:47].
The focus is on creating reports that non-technical domain experts can easily understand [21:54].
Collaborative Approach
@Shweta explained the complementary roles of technical (Rachid) and domain expertise (Shweta) in the project [21:00].
The goal is to develop @Poovendhan's skills to independently drive project progress [21:27].
white_check_mark Action items
@Poovendhan to create a simple table documenting agents, their tasks, and performance thresholds by Monday morning [19:09].
@Poovendhan to develop a dashboard showing performance across 20 different simulation scenarios [17:01].
@Poovendhan to focus on defining "perfection" in recontextualization and create clear metrics for evaluation [11:26].
Horizontal Rule
This tool uses AI to generate notes, so some information may be inaccurate. They're based on the huddle transcript and thread and can be edited anytime.
Huddle transcript
Transcript

Transcript of huddle in Shweta Homji on 9 Jan from 11:58 AM to 12:22 PM Chennai, Kolkata, Mumbai, New Delhi.

This transcript is auto-generated, so some information may be inaccurate. It won’t be surfaced in search results.
@Shweta [0:48]: OK, perfect. So you need to understand the scope of work that you are basically expected to complete, OK? So I know you've gone through multiple runs, etc. and at least by doing that you've understood the scope of work, which is that a human can select any simulation scenario and the existing simulation will get completely recontextualized, OK? Now,

@Poovendhan [1:15]: Exactly

@Shweta [1:16]: You don't have to think of, so the you have to think of it across all simulations. You have to help me gain confidence that the system that you have designed. It will work for engineering, it'll work for psychology, it'll work for nursing, for history, for business law, for computer science, whatever the heck it is. So your system needs to be completely topic agnostic, simulation agnostic, content agnostic, context agnostic, OK, but how do we create such a system so that it works every time, and that is where I had shared with you my process, and you can use yours completely, but

@Poovendhan [1:48]: Yeah Yeah.

@Shweta [1:59]: The part where you're struggling and I'm also struggling to give you specific feedback, is the way that you're presenting your information, I have to review your entire JSON. I'm not going to do that. What you need to do is lift that up to a level where you give me actionable insights. What does that mean? That means that how are you measuring the accuracy of the agents across, you know, your multiple runs. What does a run even mean to you? How many agents do you have? What are

@Poovendhan [2:26]: Yeah.

@Shweta [2:29]: They testing? How do you know this is perfect? How do you define what is perfect, right? So some of the things I already gave you that if you have to do like a semantic checker, then you have to ensure the domain fidelity and all of that is taken into consideration. You need to, and I, and I also taught you that go back to the LLM and say that this is what I need to do, chat with me and tell me what are some of the things I should be thinking about. Now, the problem with what you have submitted right now for me,

@Poovendhan [2:35]: Yeah Yeah.

@Poovendhan [2:58]: Mhm

@Shweta [2:59]: Is that it's not necessarily right or wrong, but you've given me your output and your JSON. I'm not going to review your Jason Povendan. I want you to give me reports and dashboards. OK, so because it was not clear to you.

@Poovendhan [3:07]: Go

@Shweta [3:12]: I have put together my thoughts in the format in which I would like to see it. What I've done for you is I have created a a starting template, you know, and, here, let me just share my screen and show you as well, because I have already done this in a different way, all right, but I don't want you to do it my way, but in order for me to make decisions, for example, can you see my screen now?

@Poovendhan [3:31]: Yeah Yeah, I could see him, yeah

@Shweta [3:38]: Right? So for example, just like a CICD pipeline, right? Like you basically look at different things. What I want is I want to make quick decisions for with you. Let me just get here. OK. So over here, you know, over here, this is the kind of table I want, right, where, let's say this is one of your, you have to show me your agents, right? I want an overall report which I have and I've put that that for you. I want an overall report that how many runs did you evaluate? What was the score across different

@Poovendhan [3:55]: I'm good Yeah.

@Shweta [4:08]: You giving me a dashboard, in essence, right? So you need to hold yourself accountable for what is it that you're doing and are you making any progress or not? And the only way that you can make, you can, you can be informed about the areas that you are making progress and the areas where you're not is if you have an agent helping you, OK? So, for example, over here, like this is one task, I'm not saying do it exactly like this, this is just to provide you guidelines so you

@Poovendhan [4:11]: Yeah What

@Poovendhan [4:32]: Yeah.

@Shweta [4:38]: Can adapt it to your mechanism. Right now, the confidence that I don't have in your work is mainly that I don't know what you're doing, like over here it's doing critical checks, right? These are the things that are critical. So you can do it as critical checks. You can do it as different agents, but you need to tell me that this is my agent number one. It does 5 tasks. These are the things it checks. This is the threshold value that I've put in. This is the result in 2 runs. You see what I'm doing here now? That almost always I get it, but once in a while I don't

@Poovendhan [4:50]: Yes Got it

@Poovendhan [5:07]: Yeah

@Shweta [5:10]: Get it. OK, why does it fail? Because the semantic replacement rule needs to be fixed, and then you take it back to the agent and say, OK, what is that semantic rule that needs to be changed, right? And over here it'll give you the insight that only one critical failure exists that leftover fast food references, fixing this, then you basically go and fix this, and you show me this report again. So every time you share an update with me, you basically give me reference. Shweta, I showed you the validation agent one, which this was the only one

@Poovendhan [5:35]: Yeah

@Shweta [5:40]: Failing, and after I've made the fix, I've done 30 runs and 30, like all of them are green. You see that? So now I have confidence not only on the fact that you've made progress, but are you also checking the right things, because you don't

@Poovendhan [5:46]: Good. Yeah. Yeah.

@Shweta [5:55]: Know what you're checking right now, right? You've left it to the agent, but how do you know that the simulation without a human in the loop, how are you making sure that it's 100%? Similarly, you have to look at, OK, these were not, not, these were non-negotiables. These were critical. And these were just non-blocking signals, right? So how will you create something like this? The reason I have given you this report, I've I've added that, over here like I've given, given you the whole thing and I've given you a sample prompt is so that this helps you start out, but don't just don't get married to this, OK? This is just a way for you to understand where I'm coming from, your approach is clearly different from mine. Go ahead with your approach. My approach is not always 100%, but I want

@Poovendhan [6:21]: OK Mhm Mhm

@Shweta [6:41]: You to take the ownership where, first, you're going to what failed, why it failed, you you're basically, this is what you're doing. This is the biggest thing that you're doing is you need to give me a report right now, which basically tells me that is what you have done reliable, there is no way a human can check the amount of content that our simulation generates manually. They cannot do it. We don't want them to do it. But how do we

@Poovendhan [7:07]: Yeah

@Shweta [7:11]: E vo ke confidence in them that I have done all the work on your behalf, right? So this is the way to do it, like, hey, I checked whether, you know, something happened or not, like, look, I had I've just made it just this is just random. I did it in 5 minutes to show you what I want, so don't go by this. Don't go creating this, OK? Create something for your. You need to show me like, like, you need to show me at a human level, how can I help you make the decisions?

@Poovendhan [7:28]: Yeah. OK. Yeah

@Shweta [7:41]: You're doing the JSON, you're doing everything back end, right? But how will you evoke confidence that a non techie like me, which is all our clients, right, all our professors have no understanding of tech, right? But if you tell them that

@Poovendhan [7:44]: Yeah Yes

@Shweta [7:55]: Hey, the reason why this failed is because there are some fast food terms that still remain, and then you say, would you like me to remove all the fast food terms, they say yes, you know, so you have to ensure that that yes.

@Poovendhan [8:07]: Mhm

@Shweta [8:10]: Right now you are doing so that you can bring this to 20 or 20 I will not need the human.

@Poovendhan [8:15]: Got it Sure

@Shweta [8:17]: It's like the it's like the git code, right? Like so you the get code review, like it tells you that, hey, I reviewed your code, this is what's messed up. This is what I recommend how you fix it. Would you like me to go ahead and fix it for you.

@Poovendhan [8:20]: Yeah

@Shweta [8:31]: Right? So

@Poovendhan [8:31]: Yeah

@Shweta [8:33]: Success I want to see Povendan is that every day you have to show me You need to hold yourself accountable to all of these agents, you need to show me what are you measuring? And you need to show me where is it failing. First, you get this right. I will tell you whether you're doing all of these.

@Poovendhan [8:46]: Yeah Good

@Shweta [8:53]: Like, is this even making sense or not, right?

@Poovendhan [8:55]: Yeah The thing is, actually perform this, the critical readings. Yeah, I guess you could hear me now.

@Shweta [8:57]: So. Hello

@Poovendhan [9:04]: Like

@Shweta [9:04]: Yeah

@Poovendhan [9:05]: Yeah, so

@Shweta [9:05]: Yeah, I can hear you. Yeah. Go on.

@Poovendhan [9:06]: So basically I had every agent that takes care

@Shweta [9:09]: Yeah, I can hear you

@Poovendhan [9:11]: Yes. So basically I had every agent that critically checks the domain fertility and if we have any misaligned the KLOs regarding like submission questions and the content in resources or the, the like the, if the submission questions are very easy like the copy paste thing, and I have every chicks that gives me like a summary of this report, but, when, when I was like performing the first time, the monolithic ones

@Shweta [9:33]: Hmm

@Poovendhan [9:41]: Which made the total report, like it was like generating the whole J song, not like hugs by chunks, and right now, which I have done was like chunking. So basically I have some like kelos and resources and all the sections in a hug and I was running parallelly.

@Shweta [9:48]: Mhm.

@Poovendhan [9:58]: So that's some, but I'm right now doing. And yeah, I guess I will try to implement this like in like UIE dashboard, so it'll be like good to see. Yeah.

@Shweta [10:11]: I don't think you're still understanding, so this is where I'm struggling. OK. You are forcing on the tasks. I don't care about the tasks, care about the tasks. I care about

@Poovendhan [10:15]: OK. Yeah Hello

@Shweta [10:53]: And then, can you hear me? Hi.

@Poovendhan [10:53]: Hello. Yeah, like I could hear you.

@Shweta [10:56]: OK. All I'm saying is that there are 3 tasks that you have to perform. The first one is that you have to, without without adding a time filter without adding the number of runs, OK? Without saying that, hey, I have to achieve this in 5 runs. You're not putting any limitations. Your number one goal is how do I perfectly contextualize the simulation with the new contact, OK, with the new scenario, right? There are no limits

@Poovendhan [11:01]: Sure. Mhm

@Poovendhan [11:24]: Yes Yeah

@Shweta [11:26]: There are The only thing that you need to achieve is when I do this task, 100% guarantee that the new content

@Poovendhan [11:34]: Yes

@Shweta [11:37]: Or the new simulation content will be perfect. This means you need to define what perfection is, what is it that you're checking, right? So that's what I told you that create your agents. Now don't

@Poovendhan [11:40]: Got it Yeah Sir, I guess you, yeah, I couldn't hear you properly. It's like breaking.

@Shweta [11:52]: All of that can change Hang on, let me call you back. Let me

@Poovendhan [12:32]: Hi I still can't hear you again

@Poovendhan [12:55]: Hello.

@Shweta [12:56]: Hi, can you hear me now? I'm just shifting my oh, yeah, OK. So, let's do baby steps.

@Poovendhan [12:57]: I could hear, yeah. Sure.

@Shweta [13:03]: You need to you the the simulation that I have given you is just one of the many simulation types, all right? So, step one, that you have to do is you need to give me a very simple glossary of all like I gave you, right, that there's a shorter, then there is a generation agent, then there is this you need to send me a very simple, plain text English no jason summary or glossary of what is the steps that

@Poovendhan [13:08]: Mhm Right Mhm

@Shweta [13:33]: You're following and what are the agents and within each agent, what are the tasks that each agent is running? What is your threshold value for that, and why? You understand? Because, because I, I understand the subject matter. I'm the domain expert, but you have to bring the domain experts you need to ask the right questions, so that we can define the right rules, OK? Now you have made a lot of progress on that, but I still don't have the confidence right now that the rules that you are following or the agents that you have been running in parallel are making sense. You should not jump onto the number of

@Poovendhan [13:43]: Yeah, I will It's good Sure

@Poovendhan [14:10]: Sure

@Shweta [14:12]: Runs or the time, etc. till you are very clear that you're performing all the right tasks. Right now, the problem is, even before you have the confidence that you are recontextualizing accurately, you are already jumping into the speed and time and rag and this and that. It's not about the process, it's not about the tools, it's number one about the accuracy. Can I do the task as desired, and that depends on how you define the task and how do you define the problem.

@Shweta [14:42]: And define the agents. OK, so you may have already achieved that moving, then I don't have the confidence yet because I want to read it in simple English, not your Jason. I want it as a dashboard, a very, and I don't want essays. I don't have the time. You've to make it into a simple table for me, just the way I've shared with you, which says, this is the agent, these are the subtasks, this is the definition of the task in column 2, and this is the threshold value. Here's why.

@Poovendhan [14:45]: OK Yeah. Good

@Poovendhan [15:01]: Sure

@Shweta [15:10]: OK So very simple English, do that. All right? After.

@Poovendhan [15:11]: You got it. Yeah Oh, I will be doing it with you.

@Shweta [15:15]: And the reason I'm pushing back on this is because every task that you do, you will need to think about what is it that you're doing, how are you going to break it into tasks? How are you going to ensure that all the like there is a difference between non-negotiables and difference between critical but optional and simply optional, etc. So that is the part what a prompt engineer needs to think about, OK?

@Poovendhan [15:38]: Yeah.

@Shweta [15:38]: So number one, think in terms of a human lens. You're not thinking efficiency, accuracy, whatever. You have nothing to go off like till you do this, you know, what does the work what am I doing? Like, how do I define first reframe the problem. Once you've done that, which you have, and I just want to know this because in your agents itself, I will be as the domain expert able to flag theirpo in, then you haven't thought about this. What about this? And what if this happens? How are you going to

@Poovendhan [15:52]: OK. Mhm Yeah

@Shweta [16:08]: Ensure that the resources generating the right metrics, how will you ensure that it's not giving the answer directly. How will you like lots of things, right? Right? So first give me a gloss, a simple glossary with tables of your agents and your schema. Step 1, step 2, step 3, what does that agent do? And if they are running in parallel, sure. Explain that to me. After that

@Poovendhan [16:16]: Yeah, exactly. Go.

@Shweta [16:31]: After that, you, if you, when you and I are in alignment that OK, these agents make sense, then I want you to create a dashboard of those agent tasks like I've shared with you as KPIs. These are your key performing indicators, that how do I know that I am doing something sensible. How do I know I'm making progress? How do I know that my model is so then you have to see how many runs and through that process you might add additional tasks to each of your agents, so grow that gloss

@Poovendhan [16:42]: Good

@Shweta [17:01]: Ary OK? You start with the glossary and then you grow that glossary and after that, what you need to show me is that I have done 20 runs, not of the same simulation. You have done 20 different simulation content, and you have to recontextualize it. Stooti will be your best partner to help you generate meaning. She has so many simulations. She can give you so many of them. We've done more than 20-300, OK? So the idea is, first glossary, 2 KP

@Poovendhan [17:01]: Go ahead Yeah Mhm

@Poovendhan [17:20]: OK OK

@Shweta [17:31]: I dashboard. What are you holding yourself accountable against? What are the metrics that you're looking at, like what I've shared with you, then you run it, and with the same simulation, with airline or that fast food simulation, you change the scenario 20 times.

@Poovendhan [17:36]: Yes

@Shweta [17:46]: OK.

@Poovendhan [17:46]: Got it. Yeah

@Shweta [17:47]: And in those 20 times, then you are going to give me that dashboard where you're appending an additional pass fail column, and reason why, to the same glossary that you have, which is what I've shared with you, that 19 on 20 and all of that, right?

@Poovendhan [17:55]: Sure Yeah, exactly, yeah

@Shweta [18:02]: And after we do that That's when I come in and I can be of any use to you, and you can be of use to me. That is the format in which I want the information, where you tell me that Shweta, here are the things that I am measuring based on what we discussed in the glossary and everything, it's constantly coming 19 on 20, and the reason why is that fast food elements are still, you know, no matter what I've done. So here are some of my suggestions that I want to try, but can you help me understand the rule that is this something that we can live with. Now you and I can have a meaningful conversation because you are coming

@Poovendhan [18:09]: Sure.

@Poovendhan [18:38]: Sure

@Shweta [18:39]: Informed with metrics we both know we have a common goal to go from 19 to 20, and once we do that, OK, so that is what I want. I'm not going to review your JSON and give you notes. This is not classroom. This is not tutorial. You have to figure out a way to get me the information so that I can be of use to you, and you can be off purpose to me, OK? So hopefully what I've given you, you should be able to process the next thing that I need from you, ASAP is hopefully by Monday

@Poovendhan [18:45]: It. Yeah. OK

@Shweta [19:09]: Morning, you can send it to me or sooner,

@Poovendhan [19:12]: Yeah.

@Shweta [19:13]: The first cut I want is a very simple table of the Agents that you are using, what is each agent performing and checking for? What is your schema? Very simple. I will not read beyond 3 pages, so you have to keep it very clean and very high level, and

@Poovendhan [19:22]: Because It

@Shweta [19:30]: And after that, you send that to me and after that you need to build that dashboard for me, like, like the gate code reviewing which I've shared with you where it shows 19 on 20 and the reason and how to fix. That is what you have to build, OK, so I think you are very close Bove then, OK, and it is important for you to try different things. It's important for you to try different things,

@Poovendhan [19:41]: Yeah That's it Yeah Yeah

@Shweta [19:55]: So that, you know, you get more contextual clarity. Now there will be meaning. Before this, you were shooting in the dark a little bit, but now you'll understand what you need to do, OK? So, can we agree that you have everything that you need to at least get the glossary for me and to get.

@Poovendhan [20:02]: Yeah Yeah. Mhm Yeah, exactly. I do have a thing, tomorrow I have a meeting with Rachid also on the morning, I guess. he will, he gave me the topic researches on right now which I have shared with you. yeah.

@Shweta [20:18]: OK OK Yeah, no, Raju is your Jason person. I don't care about that. I need accuracy. I do all the prompts, OK? So for me, you

@Poovendhan [20:28]: Yes. Sure

@Shweta [20:34]: Either are able to figure this out using your own schema. I, I don't want to spoon feed you, so I'd rather that whatever progress you've made, we capture that, right, because my understanding was that you had already achieved this in your technical interview, but and we just need to fine tune it for rules, but this is the way I want the information, so now you have more clarity. Do you have any questions for me?

@Poovendhan [20:40]: Yes, yes Mhm Yeah No soda if, no, I don't have any questions right now, yeah.

@Shweta [21:00]: And don't toggle back and forth between Ratchet and I, because we both do different functions. Rached's job is to ensure that so a prompt engineer does two things. One, you have to bring it to the adjason level, and the second is the core content. What is the problem you're trying to solve? I'm the problem. I'm the domain expert. He's the tech expert, right? So you will always work with both of us because Rachitt and I work together, and this role works at the intersection of both, right?

@Poovendhan [21:03]: Yeah. Exactly Mhm

@Poovendhan [21:20]: Yeah But

@Shweta [21:27]: So what I want to see Povendan is over the next, like, like 0 to 3 months that you are, you know, you are with us. I have kept you at an intern level, because I want to get you to a point where you can really thrive, and you can bring things to a level where I can make quick decisions, that it can make quick decisions, right? And then you basically get to drive a lot of the land grab stuff and set things in motion, but

@Poovendhan [21:37]: Yeah It should do

@Shweta [21:54]: You cannot go to tech till you solve the problem. So I will be your first gate, right? Because I can do this. I want you to do it so that I can focus on other things.

@Poovendhan [21:59]: Yes Got it

@Shweta [22:04]: Right? Right now I don't have the confidence that it's working because you're focusing on the wrong problem. See things that Rachid focuses on are how quickly is it coming, how the parallel structure is working. What is the orchestration? What is the overall agentic framework? How many agents do you have? Which models an API does he need to pro that's all tech. Tech is how you get it done, but the real meaning of prompts is what and why? OK, and how. So I think that

@Poovendhan [22:19]: Yes Mhm Exa

@Shweta [22:34]: You need to give me the confidence that you're able to get that done, otherwise all of this is meaningless. I'm gonna trash it, OK, so I don't want your work to get trashed, and I know that you will, and I'm throwing you into the deep sea so that you learn fast, OK? But

@Poovendhan [22:40]: Exactly Sure.

@Shweta [22:49]: Figure it out. I want to start with the glossary of your what you're doing? What are the agents, but in that kind of a format where it's a simple table so I don't need to read through paragraphs or scan through your JSON. I'm not gonna do that, OK? This is the way I want the information, and if you are send it to me, and whenever you're ready, and then ask me to review it, but let's get clear. Number one, glossary. #2, dashboard and the runs, OK?

@Poovendhan [23:02]: You what Mhm Yeah

@Shweta [23:16]: Cool. All right, all right. I'll talk to you later. Thanks for the doctor.

@Poovendhan [23:18]: Yeah Yeah, thank you, thank you, sir. I got like clear idea right now. thanks for the explanation here.

@Shweta [23:22]: Good Of course, yeah, I'm here for you and I at the same time I don't want to, you know, cripple your creativity by making you do everything according to me, but this is the way I want it reported, OK, so if it's unclear, just send me a message to Shweta and Povindan, what helps me is when you ask me specific questions, OK?

@Poovendhan [23:34]: Thank you Got it

@Shweta [23:42]: That's why I did this. Can you please review A, B, and C. That is hyperspecific, right, because I'm pulled in a billion directions. So as specific and binary you can make for me, it'll be better. The best way to communicate with me is Shweta, this is the problem I'm facing. I don't know how to proceed. Can you meet me for 15 minutes? Or Swetan, this is the problem I'm facing. I've used GPT or like, you know, to come up with solutions. These are the three solutions. It does any work for you? That is a very binary. The more binary you make for me,

@Poovendhan [24:02]: OK

@Shweta [24:12]: The more valuable you become to me poy then, OK?

@Poovendhan [24:15]: Got it, so

@Shweta [24:16]: Cool. OK, bye

@Poovendhan [24:17]: Thank you, see you. Bye








Search Cartedo



Home
1

DMs
2

Activity
3

Files
4

Later
5

More
0
Files


All files

Canvases

Lists
Recently viewed
Starred
Click the ⭐ star on any canvas or list to add it here for later.

:headphones: Huddle notes: 12/22/25 with @Shweta and @Poovendhan


headphones Huddle notes: 12/22/25 with @Shweta and @Poovendhan
Slack AI took notes for this huddle from 4:36:35 PM - 5:01:14 PM GMT+5:30. Meeting between @Shweta and @Poovendhan focused on two key projects: refining simulation scenario generation and developing a dynamic audio message system for learning simulations. View huddle in channel
handshake Attendees
@Poovendhan and @Shweta
star Summary
Ken's Simulation Scenario Project
@Shweta wants @Poovendhan to develop a multi-agent system for transforming 5 existing simulations by changing scenarios while maintaining domain and context fidelity [1:25].
The project requires creating verification agents with high accuracy thresholds (starting at 98%), focusing on checking domain specifics, context preservation, and resource quality [2:33].
Key verification criteria include ensuring resources are self-contained, under 1500 words, and do not directly provide answers but enable student inference [8:10].
Audio Message Generation Project
@Shweta wants to replace static video introductions in simulations with dynamic, 60-90 second voice messages that provide contextual information [16:49].
The project requires developing an architecture that can generate voice messages through an API (potentially 11 Labs), convert them to audio, and integrate them back into the Cartito platform [17:24].
The goal is to create more engaging, human-like voice messages that also serve as an anti-cheating mechanism [16:49].
Project Approach and Expectations
@Shweta emphasizes an immersive learning approach, wanting @Poovendhan to solve problems independently while providing guidance to prevent getting lost [19:06].
Regular status updates are expected, with @Shweta promising quick reviews and course corrections [14:39].
The projects are seen as opportunities for significant professional growth and skill development [13:51].
white_check_mark Action items
@Poovendhan must set up 6 agents for the simulation project, with a one-line description of each agent's role and threshold values by the end of the week [13:21].
@Poovendhan needs to demonstrate the ability to swap simulation scenarios quickly and accurately by January 5th [12:36].
@Poovendhan must develop an end-to-end architecture for generating and integrating voice messages, including cost analysis and technical implementation [21:20].
@Poovendhan should request platform access from Rachid and explore the learner experience by going through a simulation [16:19].
@Poovendhan to explore multiple text-to-speech tools to create a library of human-like voices [20:50].
Horizontal Rule
This tool uses AI to generate notes, so some information may be inaccurate. They're based on the huddle transcript and thread and can be edited anytime.
Huddle transcript
Transcript

Transcript of huddle in Shweta Homji on 22 Dec from 4:36 PM to 5:01 PM Chennai, Kolkata, Mumbai, New Delhi.

This transcript is auto-generated, so some information may be inaccurate. It won’t be surfaced in search results.
@Poovendhan [0:15]: Yeah

@Shweta [0:18]: OK, but then I just wanted to touch base with you. I think what you've done is, you know, pretty good. I just want you to make sure that, first I wanted to make sure that have you understood the next task and the previous task is not entirely finished. I do want justification on that, but what I wanted to provide you with was a little bit of context, OK, because, that way it'll help you finish the task and figure out what is it that I'm trying to measure, OK?

@Poovendhan [0:34]: Yeah Just OK. Yeah

@Shweta [0:48]: So what you are doing is, is, kind of what you were doing in your interview itself, where

@Poovendhan [0:48]: Yeah, sure Yeah

@Shweta [0:57]: We already have a simulation, the sample 123 that I have sent you, we already have those finished simulations, all right? Now, the professors typically get very busy by the end of the semester and other things, so they don't really have time to very quickly create new simulations, but.

@Poovendhan [1:03]: Exactly. Mhm So we will have template, yeah, so yeah.

@Shweta [1:18]: Yeah, so, hmm, go on. Yeah.

@Poovendhan [1:21]: So I just went through the product. It's tote on last week at the end. So I got like clear idea on what I have to work on, like, you know, like template based simulation right now. So that's what she wants to do.

@Shweta [1:25]: Yes I'll tell you that, yeah, I want to reduce your scope of work, which is why I wanted to tell you, currently we have 5 simulations for a professor called Ken, KEN, OK? So, Ken is a market is a management and strategy professor who has done 5 simulations with us. Now he wants to repeat those simulations, but he wants to change the scenario, which means that it could be a different kind of company, all

@Poovendhan [1:37]: Yeah. OK OK

@Poovendhan [1:53]: OK OK

@Shweta [2:03]: Right? So this is a very, that is what I sent you, that I want you to be able to change the entire context, which is what you have done, all right, but when you do this, there are a lot of things that you have to, check, and this is where I want you to write verification and validation, you know, like a checker agent and like a, yeah, so that's where I want you to, the main thing that you have to do is you have to write to those

@Poovendhan [2:03]: OK Mhm Yeah

@Poovendhan [2:26]: Exactly, yeah

@Shweta [2:33]: Agents and you have to show me that my threshold value for passing that, OK, my threshold value for each of those agents you set at a very high point. You basically say you start with 98%, and if you are not getting it, you come down to 95%, but what is it that you're checking, OK? So this is where the information becomes very, very crucial, the number one thing, so smaller things like submission question or rubric,

@Poovendhan [2:34]: Sure Yeah Mhm. Got it.

@Poovendhan [2:53]: Yes

@Shweta [3:03]: Et c Over there it is very easy because they are shorter and they are less content, so contextualizing them is much easier. The biggest challenge you will have is on the resource, right?

@Poovendhan [3:04]: Yeah Exactly Yes The resources, yeah, that's all. I was like still stuck with that little.

@Shweta [3:16]: Yes And I have already written all of these prompts, all right, and I could give them to you, but I don't want to because I want you to do it, that way you will identify more problems because your dedicatedly doing it. I understand the domain, which is why I want you to, I want to push you into, OK, go figure it out yourself, because when you do it over and over again, I know that you will be able to think more deeply about the Carterdo product, so that when you start building these,

@Poovendhan [3:21]: OK Yeah.

@Poovendhan [3:45]: OK

@Shweta [3:47]: It'll be easier. But what I really want Povendan, is that success for you by 5th of by the end of this, like by 5th of Jan, you have to show me 3 things. Your biggest task is going to be that I will give you the 5 simulations, and you do you have access to the platform now?

@Poovendhan [3:50]: Mhm Yeah

@Poovendhan [4:08]: No, still not it. I guess to be forgot I have to follow up with her again. Yes.

@Shweta [4:10]: Oh Don't follow up with Tutti. She can't give you access, like, she would have if she had a spare account, but just directly reach out to Rachid and say Shweta said to give me access to the platform so I can go and create my own thing, and I'll tell you why, OK?

@Poovendhan [4:15]: OK See. Mhm

@Shweta [4:27]: I'll, I want you to go and create it step by step, because then you will also be like, OK, this is crazy. I don't understand it. This is so not intuitive, and we're building the next version. I want you to be able to as a brand new user, say that I'm completely lost. I didn't understand 1234, like, give us a checklist, you know, so that the new platform that the team is busy building, we can, show it to you and say, hey, did it solve your problems? But document your problems somewhere that I didn't understand this like whether it's a bug, whether it's a feature

@Poovendhan [4:45]: What Yeah. Got it.

@Shweta [4:57]: Whether it's like chaos, you know, because when you understand the problem deeply, you're gonna be able to solve it better, right? But your main task, Ovendan, is that and you'll make, you'll really empower me if you can do this, is that I'm going to give you the PDF, which has all the content for all 5 of Ken's simulations, all right?

@Poovendhan [5:04]: Sure, sure, yeah Sure Yeah

@Shweta [5:18]: But you need to create like I want some kind of observability and acceptance, values to say that the work that you have done, like otherwise I'll have to manually check it, right? So the way, yeah, so the way I approach it, I'm sharing my approach. You are welcome to use any approach that you have, but the things that I want you to do are the ones that that I have written. First, you will first you will have a generation agent, right? Then you will have a check

@Poovendhan [5:26]: Got it Got it, yeah, I get it, yeah. Yeah

@Poovendhan [5:47]: Sure

@Shweta [5:48]: Agent, then you will have a validation agent that the checker agent basically sends only the incremental changes that the generation agent needs to regenerate, and then finally validation agent is the one that's like sort of summarizing everything and saying, Is this good to go or not, right? You, you do it in your own way, but I want to see that when I change the domain or the company or the scenario my domain fidelity

@Poovendhan [5:49]: Mhm

@Poovendhan [6:07]: Got it. Yeah

@Shweta [6:18]: Is what is domain fidelity mean? It means that When you're at a fast food, it is a, you know, $1.01 menu or like $50 or 50% off on a burger or like take a happy meal or like, you know, things like that. This is domain for fast food, but when you go on an airline, you don't have a $1.01 menu there. You have, you know, you have 50% off on seats or you have upgrade to or you might have, loyalty points. Yeah, all of that, right? So domain fidelity

@Poovendhan [6:27]: Yeah Yeah Yeah. Exactly

@Poovendhan [6:47]: Yeah, call me if you see something, yeah.

@Shweta [6:52]: Means that the domain of the industry is like for example, in telcos, it is ARPU, average, you know, revenue per user or like counted rates, your mobile bundle, and so on. So first, domain fidelity. Second, context fidelity that are you able to, like, if the goal was go no go decision and choosing between 4 strategic options. Are you still doing that? You know, that was the main goal of the topic, hm,

@Poovendhan [6:59]: Mhm. Yeah Yeah

@Poovendhan [7:19]: Exactly. Yeah, yeah, yeah

@Shweta [7:22]: Right. Then comes the third part that does the resource, so you'll need to create another agent which says that does the resource contain all the information the student needs to answer the submission questions.

@Poovendhan [7:35]: That, yeah

@Shweta [7:37]: At the same time, so is it self contained. That's number 1. 2nd is that is the resource within like 1500 words, OK. Third, I'm just making some rules up for you, so you'll have to make more. The third rule could be that are you sure that the resource does not have the answer, because I have seen that when I was writing prompts that if you say self-contained, then the AI with a temperature one setting will be like, you know, oh, OK, I have to make sure everything is there to answer the question. The best way to do that

@Poovendhan [7:41]: Mhm Got it Yeah

@Shweta [8:07]: Is to answer the question and give it in the resource.

@Poovendhan [8:09]: Exactly, that's what I was facing right

@Shweta [8:10]: Yes, of course, and good. See, you're gonna face this. So that's where you need to do an inference map first, OK? An inference map is that if your checker agent is, you create a checker agent, you need to write it as, Hey, I want this the resource to be self-contained, but it should not carry the answer, it should basically have all the dots for inference to connect the dots, but it doesn't really give the connected dots to the student. So you need to show me, like I don't want to manually check anything. You need

@Poovendhan [8:11]: OK. Yeah Mhm

@Poovendhan [8:33]: Exactly Yeah Got it

@Shweta [8:40]: To show me this is my agent. Look, it's saying that this is done, and then I want to, I will look at your agent rules, OK, because by Jan 5thuvendan, you what you need to accomplish for me is that as soon as Ken says, Hey, use these 5 scenarios and not the one that you used for me last time. We should be able to within minutes, you know, get everything rendered.

@Poovendhan [8:51]: Got it It Good So what do you think like, is there any like criteria for the time limit and the

@Poovendhan [9:10]: Like any kind of tokens. Do you have any idea?

@Shweta [9:12]: I think that, I think what you should do first is meet the threshold value, OK?

@Poovendhan [9:18]: Got it

@Shweta [9:19]: 1st, 1st ensure that your work is getting done, you have 100% observability and you trust, trust the validation of the system, OK? Don't worry about the time and the tokens just yet, right? After you've got me to 98, 99%, you know, accuracy. Let's look at the 2% in like that where we are losing and see why we are losing because your agent should say that I'm not reaching 100% because of what, you know, our

@Poovendhan [9:23]: Mhm. Got it. Yeah Got it. Yeah

@Shweta [9:49]: Paths are threshold value for acceptance may be 96% or 98%, whatever you set it at, but it's going to fail at the 100% mark, right?

@Poovendhan [9:54]: Yeah Mhm, yeah

@Shweta [9:58]: It's failing when it's telling you I'm passing it through, but this is why it's not reached 100%, so I want to, when you're not able to reach that, I want to see why not, right? And we might see that, oh, you know, the word limit is like 150 word limit is the problem. I can make that decision. That's where the human in the loop comes in, me, I'm the human in the loop. I can tell you ignore this, right? So once you get to that point, then you will have, OK, sweatta, the average number of tokens are $1500 or like 155

@Poovendhan [10:03]: Exactly yeah Mhm Exactly, yeah Got it

@Poovendhan [10:19]: Mhm

@Shweta [10:28]: 0000 and the average time it's taking is 21, like no, like 9 minutes. OK, now you got, you've achieved one control variable. Now then the other ones are time and tokens. Now you will optimize the prompt. How will you do it? You can make it into multiple smaller agents that run concurrently. You can do multiple different things, but you all for X first, which is get the aid.

@Poovendhan [10:34]: Yeah Yeah Yeah

@Poovendhan [10:54]: We should get output

@Shweta [10:56]: Yeah, and the output first and then decide which model, Gemini is very speedy, but after a point, I, I right now my favorite is Gemini, and in opening I also try to stick with like certain snapshot models, and I want to.

@Poovendhan [10:56]: Exactly, yeah Mhm Yeah, that's what I was going to the, the, initially I was like setting a four-room mini because it was like very fast.

@Shweta [11:14]: OK Oh, no, no, hoomini is bogus. Like it gives you nothing. Like for me is OK, but you'll run into a lot of problems in photomini, you know.

@Poovendhan [11:18]: Yeah Yeah Exactly, yeah, yeah, yeah. Now I changed to 4 4.1 right now. I'm like trying with the with it.

@Shweta [11:30]: I think that if you, if you go to, one of the snaps, don't go, that's not the snapshot model. The snapshot models, go to the dated snapshot models because they don't change, right? You can't use SAGBT latest. You can't use 5 latest, 40 latest, no, none of the latest models, nothing, because those are constantly changing, we'll have to, we'll just get lost in prompt adaptation. That's not what we want to do. And

@Poovendhan [11:38]: Yeah, yeah Exactly. Mhm. Yeah.

@Poovendhan [11:55]: Yeah

@Shweta [11:56]: In Misty

@Poovendhan [11:57]: Mhm.

@Shweta [11:58]: In Misty when you write the prompt, you can split it. I don't know if you've seen it or not, but you should.

@Poovendhan [12:02]: Yeah, I was like testing that totally with the different mos right now, yeah.

@Shweta [12:04]: Very good So I write all the prompts outside, OK? And then I bring them in to Misty and give them input and my prompt and I executed across like 8 or 9 different models that I typically use, so that I can have a main model, I can have a backup model, and I can, like, you know, a complete like and a complete fallback model if nothing else than this one. So right now, if you, your first step, so that you don't like die of

@Poovendhan [12:10]: Mhm Yeah Mhm OK

@Shweta [12:36]: Stress, is that just ensure that these 5 simulations I can, you will need to demo to me on 5th of January that I can give you any topic, like I will come and say, OK, uve then just change the entry level prompt scenario to this. Swap out the scenario for this, I want these 5 simulations. I'm not giving you the world. I'm giving you only these 5 simulations where I want to change the scenario, you are going to basically

@Poovendhan [12:37]: OK. Go

@Shweta [13:06]: And I can give you a scenario in any words, OK? So I'm not changing the context. I'm changing the scenario, all right? So the input prompt is probably not changing, it's not changing. The scenario is changing.

@Poovendhan [13:10]: So yeah. Mhm Got it. Yeah

@Shweta [13:21]: OK, so you need to do this, and you need to set up your different you need to then, before, like whenever you can, before the end of this week, ideally, you need to tell me that, OK, I have set up 6 agents, like whatever is your schema. I have a semantic layer, I have a domain fidelity layer, I have whatever, all your checker agents, and you need to in one line, tell me what that what is the role of that agent? You design it yourself. Nobody's gonna give

@Poovendhan [13:38]: Yeah. Yeah

@Shweta [13:51]: You this opportunity. I'm telling you this is a very fun project. It's intense learning, but you will grow 100 eggs with this exercise. Give it a shot. I have already done it, so that is why I can tell you how, where you might be making mistakes, which is why I can send you a whole TED Talk of questions that did you think of this? Did you?

@Poovendhan [13:54]: Sure, yeah Exactly.

@Shweta [14:10]: Done that, right

@Poovendhan [14:10]: I get it. Yeah, I get it here.

@Shweta [14:11]: Right? But I want you to do it. And if I show you how I've done it, you won't learn. OK? So, so I hope you see this as an immersive learning exercise and this is the problem you need to solve, so you will send me your agentic structure, you will send me all the agents in one line description and what you have set as the threshold value, you will need to send me your

@Poovendhan [14:16]: Yeah, exactly, yeah

@Poovendhan [14:34]: Got it

@Shweta [14:39]: Output whenever you are ready, but when you report to me on Jan 5th, you're going to show me that it is done, you know, and that what does that mean to you? Don't wait till Jan 5th, in the middle the way you're updating me, keep updating me. You will also benefit from it because I will review your work very quickly, tell you, OK, you're going down the wrong road, OK? You don't want to, you, while I want you to go figure it out yourself. I also don't want you to get lost in the weeds, OK?

@Poovendhan [14:42]: Sure Mhm OK Sure

@Poovendhan [14:59]: Go you Mhm

@Shweta [15:08]: All right. So, this is one project, but you will, this will be a stressful project, so I'm giving you fun ones on the side. The other part is

@Poovendhan [15:15]: Yeah, right now I'm seeing the audio one right now.

@Shweta [15:18]: Hm Let me explain what I want in the audio. Today, when we create the simulations, we have we have recorded AI videos which are static, all of that is static content, and it's very boring because we only have about 40 avatars or so, but there are lots of simulations that are done in the same course. They don't want to use the same avatar. So what you are doing for me is that imagine when the student and and Povendan you have to ask Rach

@Poovendhan [15:25]: Mhm It's a Yeah. Yeah

@Poovendhan [15:42]: Mhm Yeah

@Shweta [15:49]: For the links to the learner platform and the creation platform. First, you can even get in touch with Vinoth. I'll just introduce you to Wynoth right now. You are, but I don't know if Vinoth can set you up though. Rachut will have to set you up, so do that, OK? And I will just introduce you to rail and Vino and ask Ratchet to give you rail is my co-founder, and he is the head of

@Poovendhan [15:57]: Got it Sure Mhm

@Poovendhan [16:17]: Yes

@Shweta [16:19]: Learning. You need to ask rail that rail can you show me, like the journey of the students so that then, like, or give me once ratchet gives you access, go do one of the simulations like rail can give you access to it. Why? Because I want you to experience it as a learner, all right? And we're trying to improve the learner experience, but today's learner experience happens like this. You're you start the simulation, the first thing that happens is a video call with a manager. It's a

@Poovendhan [16:20]: OK True. Yeah.

@Poovendhan [16:46]: Mhm Yeah.

@Shweta [16:49]: Recorded video audio like video message, OK? So the manager tells you some blah blah blah, hey, welcome to the team. Some generic random nonsense, which is not like which was very cool two years ago, but we have to get rid of it now. Today, what I want is as soon as you start the simulation, you should be greeted with, like, hey, as you'd like a, a voice message, you know, which is about the company, so you don't need to read anything. This also becomes like an anti-cheating hack because if it's a voice message that you cannot download, then how will you take

@Poovendhan [16:52]: Yeah Yeah Yeah, exactly Mhm

@Shweta [17:19]: All that context and you'll have to type something in your LLM to cheat also, right?

@Poovendhan [17:23]: Exactly, yeah.

@Shweta [17:24]: So the voice message that you are working on, when we generate the simulation, it'll automatically generate a 60 to 92nd pitch, OK, like a voice message, right? That voice message needs to go through an API to a tool, I don't know, 11 labs or whatever you can get, right? Where it dynamically creates and sends it back into Cartito.

@Poovendhan [17:36]: Yeah Yeah Mhm. Got it. Yeah.

@Shweta [17:49]: So you have to think about the architecture as well, that, if the, if I have created a script that my AI has generated for that video audio message, then that script goes to 11 labs.

@Poovendhan [17:52]: Mhm. Mhm Lemon labs and the audio should be trans I mean like speech.

@Shweta [18:05]: And then Yes, and then how does that MP4 file or whatever come back to Cartido, where will it be nested? Like, where will it get saved? Where is it in our S3 bucket, or where, where is it going to get saved, like as an identity, and so on, right? so that is the part that I want you to do the whole story that any tool that you select, the problem that you're trying to solve is, I will give you from our AI generated like of

@Poovendhan [18:22]: Yes Mhm Got it

@Shweta [18:38]: Video, like an audio message, you will take that through the API 11 Labs generates it and then brings it back into Cartido, which gets rendered.

@Poovendhan [18:43]: Mhm Got it Yeah

@Shweta [18:47]: What mechanism you want which container you want, we can test that with Rachid later, but the functionality should exist. This is a this is a criteria for acceptance, OK?

@Poovendhan [18:53]: Sure Yeah. Got it here

@Shweta [18:59]: If you have any doubts, you send me your, like status update.

@Poovendhan [19:05]: One case somebody, yeah, I'll send you to somebody or like, you know.

@Shweta [19:06]: Yeah Yeah. I, I don't micromanage, but I will step in to guide you so that you don't feel like you're like shepherd less, but at the same time, I'm, I hope I've made what I want you to do very clear. You're taking notes as well, so you let me know how I can help you, OK? And by sending me updates, OK? And what you're doing right now is perfectly fine, but by Jan 5th, my expectation is that your textbook prompt

@Poovendhan [19:10]: OK. Yeah, got it Yeah, exactly, yeah. Sure, sure, sure

@Poovendhan [19:30]: Mhm

@Shweta [19:39]: You should be, that's a very simple prompt you were overcomplicate.

@Poovendhan [19:42]: Yeah, I guess I finished, yeah, I got the answer. it's so fucking.

@Shweta [19:47]: OK

@Poovendhan [19:47]: But yeah, but before that I was going through the trailer trailer, the task manager and be using some kind of EFPA to extract the textbook. OK. And our fallback is the one we are gonna gentle the open EI ones.

@Shweta [20:04]: Yeah, I'm.

@Poovendhan [20:04]: But the results are coming perfect right now. Yeah.

@Shweta [20:06]: Yeah. Look, I'll tell you what we're doing. When you go to chat GPT or any of these LLMs, you basically say, hey, how can I help you today? Some of the things to get you started. Do an image of this, do it like you have these prompt pills, right? You can see.

@Poovendhan [20:12]: Mhm Exactly, yeah

@Shweta [20:20]: In lieu of that, we're telling a professor, don't know how to create, OK, give me the name of your textbook and use any of the chapters. Here is a quick list of chapters. It's not even a big deal. Your accuracy doesn't even need to be 90%. It needs to be 88%. Anyway, all so why overcomplicate it? So the textbook prompt is very basic if you can't do it or do I already wrote that promise. It's a very basic prompt. It'll work, but I want you to your biggest project, which you need to impress me with your skills, whichever way you want to do it, which I'm really excited to see how

@Poovendhan [20:29]: Exactly, yeah Mhm OK Sure

@Shweta [20:50]: You're going to approach it is Ken 5 simulations. You have 123. I have to give you the remaining 2, OK? So I will give you that and the second thing you have to do is that whole whether you use 11 labs or anything else. I like 11 labs a lot, OK, but you find me something where the voice is not robotic. It doesn't say oh hi booby in the mic and da da da da da, not like that. It has to be more human, you know, like a human voice. So so far 11 Labs has been my favorite, but you choose anything that you can, you choose multiple so that

@Poovendhan [20:51]: OK OK It Mhm Mhm

@Poovendhan [21:18]: So

@Shweta [21:20]: Increases our library where we can have different voices, but you need to show me the end to end, that how will it go from us to 11 laps, how will it come back? and what does it cost, etc.

@Poovendhan [21:22]: Good Mhm Got it, got it, sir. So basically, after finishing that, can I upload it through the Git up right? So I got the access, but some axes are missing for some repose. So I have to ask that question.

@Shweta [21:39]: Mhm. And you need to talk to Rache. Yeah, you ask Rache. All tech access and everything, Raut will give because I mean that's how the VAPT and privacy works are in the company, so he, single person approval, so tech is all rugged. I'm all context, OK? So you need to figure out, so by Jan 5th, again I'm saying, and it's not like between now and Jan 5th you go invisible. I'm here. I always check your work whenever you send me something, I reply to it, and it might take me within like a day to respond, but I.

@Poovendhan [21:43]: Exactly Both Mhm. Go to

@Poovendhan [22:08]: Go to together. Mhm

@Shweta [22:12]: Will OK, so keep me in the loop, but I by Jan 5th, I want this done. And I also want you to get that audio message thing, right? Now if you are struggling, if you don't know how to get it done, you need to first go out to the land graft community. You also have your brother, figure it out on your own first and then me, I have done all of this, but

@Poovendhan [22:12]: Sure, sure, sure Mhm. Got it Mhm. Got it. Mhm. Mhm

@Poovendhan [22:30]: Got it OK.

@Shweta [22:34]: If I answer everything, then I feel like you won't learn, OK?

@Poovendhan [22:38]: Yeah, I get it, yeah

@Shweta [22:39]: And I need an automation, like I need that I can just like lean on you and say, hey Puvendan, can you just get this done and you're like, Sure, I'll need a day to figure it out, and I'll do it, you know. That's what I need, OK?

@Poovendhan [22:49]: Sure. Got it. Got it, sure

@Shweta [22:51]: But at the same time, don't stress out and don't feel like, oh my God, I'm dying, I'm going to disappoint people. No, you're, we are your faith, OK? Are you sorted? Any questions for me right now?

@Poovendhan [22:57]: No, yeah Got it, yeah All right, I don't have any questions, just, I'm just figuring out the simulation again. So, yeah, what I think is I need more access to chat GPT and other things because I'm like

@Shweta [23:08]: OK.

@Poovendhan [23:15]: Finishing that opens right and I don't have money to pay it. So that's what I mean. Yeah, yeah.

@Shweta [23:19]: Why should you do that? Misty should Misty, you should use our APIs. What not? I should do use our money, like your, your money, I should use us.

@Poovendhan [23:24]: Yeah, yeah, yeah, I'm using the Yeah, the ones I'm using the Google Ye Studio. Basically they have like multiple features, code executions and everything like.

@Shweta [23:34]: Mhm.

@Poovendhan [23:39]: So that's where I'm like facing issue, but yeah. Yeah, maybe I should go with the EPAs, EPKs right now.

@Shweta [23:46]: Absolutely, use the API, yeah, and if it's, if you're blowing up and, other things, then Rachad will let you know, like, you know, but this is what your job is, so obviously you need access, so get ready to give you API access to like, but he should have already hooked you up on Misty. Didn't he give you the impressions?

@Poovendhan [23:56]: Got it Yeah, she, she gave the EP is, but I feel like, you know, maybe I am using too much of EPs maybe I'm exploiting too much of money so.

@Shweta [24:09]: No, they are fine, but then we are not a small dinky company. It's all right. You can just go onto Misty and use it. This is your job. This is what you have to do day in day out so that I don't have to do it, OK?

@Poovendhan [24:10]: Yeah. Got it, got it.

@Shweta [24:20]: All right. Anything else?

@Poovendhan [24:21]: No, nothing else, that's it. If I

@Shweta [24:22]: So you will follow up with it. You need to nag Rachchet if you don't have access. Nobody's gonna come and interview. This is how the company works. Like you ask for it, you nag, you make it happen. OK, if after nagging 3 times, you're not getting it from Raja, please message me.

@Poovendhan [24:24]: Take that Got it Got it. Yeah. Sure, sure, sweetheart. I'll get it.

@Shweta [24:38]: OK, all right, bye

@Poovendhan [24:39]: Thank you. Bye