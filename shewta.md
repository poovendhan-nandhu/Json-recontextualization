



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



