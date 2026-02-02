

Poovendhan <> Rachit - Transcript
00:00:00
 
Poovendhan Velrajan: Okay. Okay. What would be my
Rachit Sharma: Hey, how are you?
Poovendhan Velrajan: Hello, Richard. Can you hear me now?
Rachit Sharma: Yes, I can hear you. Uh we have real also here. We also need uh help from you.
Poovendhan Velrajan: Hello.
Rachit Sharma: So we uh get over call itself.
Poovendhan Velrajan: Yeah.
Rachit Sharma: So, yeah,
Poovendhan Velrajan: Yeah.
Rachit Sharma: you wanted I guess uh two helps right regarding some GS and the generation. So, what was
Poovendhan Velrajan: Yeah. Exactly. So basically right now what I done was I got like five simulation
Rachit Sharma: it?
Poovendhan Velrajan: from sweater. So it's for k simulation I guess. So I needed to generate the lesson package towards that using the prompt system right now. So basically I developed an monolytic prompt.
Rachit Sharma: Sure.
Poovendhan Velrajan: Basically uh it generates the lesson package towards the guidelines and everything right now. Yeah. So, I could show you a demo right now. So, right now I have loaded the sample base simulation.
 
 
00:08:20
 
Poovendhan Velrajan: Right now it's in the local system and right now I do have the prompt which I was given uh for the simulation and we give the max ripples that we are going to give right now because it takes like too much of uh latency so I'm giving three and yeah so it gets like simulated right now. So Rashid uh the big problem over here is the latency. Uh my previous results I was like getting like uh yeah so it takes like too much of tokens and uh it gives like partial results right now. So I was like seeing like how could I like improve the system overall right now. Yeah.
Rachit Sharma: Okay.
Poovendhan Velrajan: So these are my previous results I got. So I got like some cases where I went like 300 to 400 seconds know and I got like whole package.
Rachit Sharma: Okay. So, uh right now do not worry about the latency like even if it takes five six minute that's uh something that we
Poovendhan Velrajan: Yeah.
Rachit Sharma: can worry about once you have achieved the efficiency.
 
 
00:09:57
 
Rachit Sharma: So right now worry about the efficiency on how much are you how much efficient solution are you
Poovendhan Velrajan: Got it.
Rachit Sharma: getting.
Poovendhan Velrajan: Yeah. So I did get like 90% right now on the pass results for each and every guidelines but I'm like hoping to get like 98 minimum. So overall score in some cases I was like failing 60% some 70% and 80%. But most of the cases I got like 90 and uh right now. So I'm like hoping to get like 98% above for the validation. So what
Rachit Sharma: Can you show me the uh partial data that you're
Poovendhan Velrajan: else? Yeah, sure.
Rachit Sharma: getting?
Poovendhan Velrajan: I could share you right now. Uh, it generates to JSON file. So, could I share you to Slack? Right.
Rachit Sharma: Yeah.
Poovendhan Velrajan: Yeah, I did send you. Yeah.
Rachit Sharma: Okay.
Poovendhan Velrajan: Uh I did get the guidelines from sweater on how the results should be like the kos should be related to I mean the submission questions should be related to kos and uh the content fability and the resources and everything.
 
 
00:13:12
 
Poovendhan Velrajan: So I got checked up with sweater every day. I would like send her the updates and the prompt system on what I have worked and not. So yeah.
Rachit Sharma: So uh then how did you do it in the PC assignment that you did for
Poovendhan Velrajan: Yeah.
Rachit Sharma: us?
Poovendhan Velrajan: So over there I done like I locked the JSON format. So I splitted every uh fields and I recontextrated but uh while I'm doing this I done like monolytic because the uh results were there like failing uh on that system. So I was like trying this monolithic whole prompt system so it got like a better result than the previous ones. Yeah.
Rachit Sharma: Okay, give me a second again. Uh, did you try chunking? Did you try chunking chunking the data set into different buckets so that
Poovendhan Velrajan: What?
Rachit Sharma: you first uh transform the first bucket then the second bucket like basically divided into
Poovendhan Velrajan: Yeah.
Rachit Sharma: chunks
Poovendhan Velrajan: So basically splitted the whole format. So basically when we get the base JSON we fix the structure hold the whole structure and uh split it and uh after that we will like merge it but I didn't try to split like the JSON right now.
 
 
00:14:57
 
Rachit Sharma: Okay,
Poovendhan Velrajan: Yeah.
Rachit Sharma: try doing that.
Poovendhan Velrajan: So let I I guess I should try.
Rachit Sharma: Try doing that because you can work on it parallelly like your assessment criteria,
Poovendhan Velrajan: Yeah. Sure.
Rachit Sharma: your kos scenario, all of those things can be done parallelly and simulation flow can be done parallelly. That way you do not have to uh wait for the latency and also your context gets small.
Poovendhan Velrajan: Good.
Rachit Sharma: You don't have to do the whole chunk together. Also link into it. Uh you if you could use uh rag or if you could u use embeddings for it or even some vector DB so that your structure remains the same and you know where data is stored
Poovendhan Velrajan: Sure.
Rachit Sharma: that you can bring it back any time.
Poovendhan Velrajan: Mhm.
Rachit Sharma: So try these approaches and let me know by the end of the day whatever you feel like just create small small
Poovendhan Velrajan: Sure,
Rachit Sharma: PCs to test it together.
Poovendhan Velrajan: sure, sure, Richard. Yeah, I will do that.
 


Jan 7, 2026
JSON LEASSON PACKAGE GENERATOR 
Invited Poovendhan Velrajan Rachit Sharma Shweta Homji
Attachments JSON LEASSON PACKAGE GENERATOR  
Meeting records Transcript 

Summary
Poovendhan Velrajan reported high latency issues when using chunking and embeddings due to their monolithic architecture, which processes the entire JSON before working in chunks, a setup they shared with Rachit Sharma. Rachit Sharma reviewed Poovendhan Velrajan's LangSmith setup for the "lesson simulation package" and advised on a new JSON structure, emphasizing the necessity of preserving all IDs, objects, arrays, and their structures within the new format. The main talking points revolved around resolving the latency issues, the analysis of the current data structure, and the implementation of a new JSON structure.

Details
Notes Length: Standard
Challenges with Chunking and Embeddings Poovendhan Velrajan reported issues with using chunking and embeddings on Monday, noting that the latency was very high due to their monolithic architecture. They explained that attempting to use chunking resulted in too many validations and repeats, thus increasing latency. Poovendhan Velrajan shared the architecture, which includes a generate node and seven different checkers, with Rachit Sharma, and demonstrated that the system initially re-contexts the whole JSON before processing in chunks (00:00:00).
Analysis of LangSmith and Data Structure Rachit Sharma reviewed Poovendhan Velrajan's LangSmith setup for the "lesson simulation package," which Poovendhan Velrajan felt was messed up after attempting parallelization and chunking (00:10:07). Poovendhan Velrajan confirmed they had converted a PDF shared by SWE into a whole package system JSON for recontextualization (00:11:59). Rachit Sharma offered to share a new JSON file, advising Poovendhan Velrajan that they would be provided with the JSON directly, eliminating the need for them to convert anything (00:13:54).
New JSON Structure and Preservation of IDs Rachit Sharma shared a new JSON structure and instructed Poovendhan Velrajan to use it to create the structure, referencing the structure previously discussed in an interview. Poovendhan Velrajan observed that the new JSON structure was very different from what they had been working on but believed the method of sending batch by batch could work (00:15:24). Rachit Sharma emphasized the necessity of preserving all IDs, objects, arrays, and their structures within the new JSON, which is a new rule that Poovendhan Velrajan will have to follow while re-working their solution (00:22:25).
Next Steps and Meeting Schedule Poovendhan Velrajan confirmed that the simulation is based on "topic vizard data". Rachit Sharma requested that Poovendhan Velrajan send the new implementation and the Lang link by five o'clock, noting that they should use the method from their interview. Poovendhan Velrajan agreed to send a proper report and scheduled a follow-up meeting after five, confirming they would use RAG implementation in the update (00:23:19).

Suggested next steps
Poovendhan Velrajan will send the implementation, the lang link, and a proper report (mentioning the use of RAG) by 5 PM.
Poovendhan Velrajan will use the new JSON shared by Rachit Sharma to create the structure and try to use chunking or embedding while preserving the entire JSON structure, including all the IDs, objects, and arrays.
Poovendhan Velrajan will schedule a meeting on Rachit Sharma's calendar after 5 PM once the implementation is done.

You should review Gemini's notes to make sure they're accurate. Get tips and learn how Gemini takes notes
Please provide feedback about using Gemini to take notes in a short survey.
