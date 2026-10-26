"""Build a capability-matched training dataset for the LLM router.

Unlike the arena dataset (labels = noisy 2025 battle outcomes), this dataset
teaches the router what KIND of assistant suits a prompt:

  (prompt, capability description, label)

- Prompts are grouped by the capability they need (coding, casual chat, ...).
- Descriptions are name-free "capability cards" per archetype, written in the
  style of OpenRouter descriptions, plus REAL OpenRouter descriptions
  (name-scrubbed) anchored to archetypes by keyword rules so the model also
  recognizes the exact text style seen at inference time.
- Labels are by construction: 1.0 matching archetype, 0.0 mismatching, and a
  graded score for sensible adjacencies (e.g. a deep-reasoning model for a
  coding prompt is decent, not perfect).

Output: data/router_training_pairs_capabilities.csv
"""

import csv
import json
import random
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = PROJECT_ROOT / "data" / "model_profiles.json"
OUTPUT_CSV = PROJECT_ROOT / "data" / "router_training_pairs_capabilities.csv"

SEED = 42

# --------------------------------------------------------------------------
# Capability archetypes: description cards (name-free, OpenRouter-ish style)
# and prompts that need that capability.
# --------------------------------------------------------------------------

ARCHETYPES: dict[str, dict[str, Any]] = {
    "reasoning": {
        "cards": [
            "A frontier reasoning model built for hard problems: competition mathematics, scientific analysis, formal proofs, and long multi-step chains of thought. Slower and more expensive than everyday models, but reliable where correctness matters.",
            "This assistant specializes in deep reasoning. It excels at mathematics, logic puzzles, research questions, and planning tasks that require thinking through many steps before answering.",
            "A state-of-the-art model for advanced reasoning, quantitative analysis, and academic work. It takes its time on hard problems and is optimized for accuracy over speed.",
            "Designed for maximum reasoning depth: olympiad-level math, causal analysis, strategy, and any question where the answer must be derived rather than recalled.",
        ],
        "prompts": [
            "A farmer has chickens and rabbits, 35 heads and 94 legs in total. How many of each animal does he have?",
            "Prove that the square root of 2 is irrational.",
            "Factory A produces 45 machines in 20 minutes; factory B produces 30 in 15. Which produces more in 2 hours?",
            "What is the time complexity of this algorithm and why? Explain with a recurrence relation.",
            "Three switches outside a room, one bulb inside. You may enter once. How do you tell which switch works?",
            "Derive the closed-form solution for the nth Fibonacci number.",
            "I have 8 balls, one heavier. What is the minimum number of weighings on a balance scale to find it?",
            "Evaluate whether this argument is valid: all cats are mammals, some mammals swim, therefore some cats swim.",
            "Solve this AMC problem: for how many integers n is n^2 - 3n + 2 prime?",
            "Design an experiment to test whether a new drug improves memory, including controls and sample size reasoning.",
            "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Explain the logic.",
            "A train leaves at 3pm travelling 60km/h; another at 4pm at 80km/h. When does it catch up?",
            "Explain the mathematical difference between correlation and causation with a formal example.",
            "What is the optimal strategy in this auction game theory scenario?",
            "Estimate the number of piano tuners in Chicago, showing your reasoning step by step.",
            "Analyze the time-space tradeoff of memoization vs tabulation for dynamic programming.",
            "Is the following proof by induction correct? If not, identify the flawed step.",
            "Calculate the probability of drawing at least one ace in five cards, showing the combinatorics.",
            "Compare Bayes frequentist and Bayesian interpretations of probability applied to this medical test result.",
            "A cable of 80m is cut; one part is 12m longer. Set up and solve the equations.",
            "Reason through the Ship of Theseus paradox and defend one position with logical arguments.",
            "Find the bug in this mathematical induction argument that all horses are the same color.",
            "What is the Nash equilibrium of this two-player game matrix?",
            "Step through the Euclidean algorithm for gcd(1071, 462) and explain why it terminates.",
        ],
    },
    "coding": {
        "cards": [
            "An expert programming model for complex software engineering: writing and refactoring production code, debugging tricky failures, and explaining architecture across many languages and frameworks.",
            "This assistant is optimized for developers. It writes, reviews, and debugs code, designs APIs, works with git histories and stack traces, and explains engineering tradeoffs.",
            "A strong coding model suited to building real applications: distributed systems, algorithms, testing strategy, and performance optimization, not just snippets.",
            "Built for serious software work: from quick fixes to full project scaffolding, with deep knowledge of programming languages, tooling, and infrastructure.",
        ],
        "prompts": [
            "Write a complex PyTorch script to train a neural network with distributed data parallel.",
            "Can you debug this React useEffect hook that causes an infinite loop?",
            "Write a Python script to scrape a website using BeautifulSoup.",
            "How do I implement a binary search tree in C++ with deletion?",
            "Refactor this 200-line Java class to use the repository pattern.",
            "My Kubernetes pod keeps crashing with OOMKilled. Help me diagnose the memory leak.",
            "Write a Rust function that parses an RFC 3339 timestamp without external crates.",
            "Design a REST API for a blog with pagination, auth, and rate limiting.",
            "Why does this SQL query do a full table scan? Here is the EXPLAIN output.",
            "Implement a rate limiter using Redis and Lua with token bucket semantics.",
            "Convert this callback-based Node.js code to async/await with proper error handling.",
            "Write unit tests for this currency converter, covering edge cases and rounding.",
            "My Docker build is 1.2GB. How do I create a multi-stage build to shrink it?",
            "Write a Go worker pool with graceful shutdown and context cancellation.",
            "Fix this race condition in my Python threading code and explain the fix.",
            "Set up GitHub Actions CI for a monorepo with path-based triggers.",
            "Write a parser for this custom log format using Python's re module.",
            "How should I structure a large TypeScript codebase for a fintech app?",
            "Optimize this pandas pipeline that currently takes 40 seconds to run.",
            "Implement consistent hashing for a distributed cache in Java.",
            "Write a Terraform module for an S3 bucket with versioning and encryption.",
            "Debug this segmentation fault in my C program. Here is the core dump backtrace.",
            "Create a WebSocket server in Python with heartbeat and reconnect logic.",
            "Review this pull request diff and flag security issues and code smells.",
        ],
    },
    "general": {
        "cards": [
            "A balanced general-purpose assistant for everyday questions, drafting, planning, light analysis, and explanations across many topics. A good default when no special capability is needed.",
            "This model handles a wide variety of daily tasks well: answering questions, writing emails, making plans, summarizing content, and helping with decisions.",
            "A versatile everyday assistant that balances quality, speed, and cost. Great for mixed workloads where you want one reliable model for everything.",
            "Well-rounded and dependable for general assistance: research help, document drafts, brainstorming, explanations, and routine knowledge work.",
        ],
        "prompts": [
            "Write a polite email to my boss asking for next Friday off.",
            "Help me plan a 3-day trip to Rome on a mid-range budget.",
            "What is the difference between a 401k and an IRA?",
            "Draft a wedding toast for my younger sister.",
            "How do I renew a passport in the UK?",
            "Give me a weekly meal plan for a family of four with grocery list.",
            "Explain how compound interest works in simple terms.",
            "What should I look for when buying a used car?",
            "Help me write a LinkedIn summary for a career change into data science.",
            "Summarize the pros and cons of renting vs buying a home.",
            "Create a study schedule for a certification exam in six weeks.",
            "What are some good team-building activities for a remote company?",
            "Draft a complaint letter about a delayed flight requesting compensation.",
            "Explain the point system in this board game so a beginner understands.",
            "Help me decide between two job offers using a comparison framework.",
            "What is the process for registering a small business in Canada?",
            "Write a birthday message for a friend who loves hiking.",
            "Recommend a beginner-friendly fitness routine for a busy parent.",
            "How should I store winter clothes to avoid damage?",
            "Give me talking points for a performance review self-assessment.",
            "Explain the rules of cricket to an American.",
            "Plan a first date itinerary for a rainy Saturday in a big city.",
            "What is a healthy lunch I can meal-prep with just five ingredients?",
            "Help me write an obituary template for a family member.",
        ],
    },
    "casual": {
        "cards": [
            "A friendly conversational companion for casual chat, small talk, jokes, and emotional support. Warm and personable rather than technical.",
            "This model is built for relaxed conversation: venting about your day, silly hypotheticals, jokes, and companionship. It keeps the chat light and engaging.",
            "An approachable chat partner focused on natural, fun conversation and being a good listener rather than crunching hard problems.",
            "Designed for social interaction: banter, empathy, storytelling games, and open-ended chit-chat where personality matters more than raw intelligence.",
        ],
        "prompts": [
            "Tell me a joke about pigeons.",
            "I had such a long day, can we just chat for a bit?",
            "What's your favorite pizza topping and why?",
            "mowisz po polsku? opowiedz mi coś zabawnego",
            "Pomówmy o czymś miłym, mam gorszy dzień",
            "Tell me a fun fact that will impress my friends at dinner.",
            "If animals could talk, which would be the rudest?",
            "I'm bored. Entertain me.",
            "What's a good would-you-rather question?",
            "Keep me company while I wait for my train.",
            "Say something to cheer me up, I failed my driving test.",
            "What hobby should I pick up just for fun?",
            "Tell me about a weird historical tradition.",
            "Recommend me a comfort movie for a rainy evening.",
            "Lets play 20 questions, you go first.",
            "What's the most useless superpower you can imagine?",
            "I just want to vent about my noisy neighbors.",
            "Chat with me like an old friend would.",
            "Co robisz? opowiedz mi jakiś śmieszny historię",
            "Make up a story about a dragon who is afraid of heights, just for fun.",
            "What song should I put on while cooking dinner?",
            "Ask me a random question, I feel like talking.",
            "Tell me your favorite terrible pun.",
            "I got a puppy today!! Tell me something cute about puppies.",
        ],
    },
    "creative": {
        "cards": [
            "A creative writing model with a knack for voice and style: fiction, poetry, screenplays, marketing copy, brand naming, and vivid brainstorming.",
            "This assistant shines at imaginative work: stories with strong narration, taglines, worldbuilding, song lyrics, and ideas nobody else at the table came up with.",
            "Built for words that sell or move people: ad copy, product descriptions, speeches, and creative concepts, with range across tones and formats.",
            "A novelist-grade writing partner for drafting, rewriting, and riffing, strongest where originality and stylistic control matter.",
        ],
        "prompts": [
            "Write a short story about a lighthouse keeper who receives letters from the sea.",
            "Give me 10 name ideas for an artisan coffee brand.",
            "Write a villanelle about autumn endings.",
            "Draft a punchy product description for handmade leather boots.",
            "Write the opening scene of a noir thriller set in Warsaw.",
            "Help me write a best man speech that is funny but not embarrassing.",
            "Create a world where gravity reverses every seventh day. Describe daily life.",
            "Rewrite this paragraph in the style of Hemingway.",
            "Write lyrics for a breakup chorus, indie-folk mood.",
            "Brainstorm taglines for a fitness app targeting busy parents.",
            "Write a chilling two-sentence horror story.",
            "Draft a creative brief for a sneaker launch campaign.",
            "Write a monologue for a stage play about a retired magician.",
            "Pen a persuasive op-ed opening paragraph about urban cycling.",
            "Give me a plot twist for my detective novel involving the victims diary.",
            "Write product copy for a smart garden sensor, playful tone.",
            "Compose an ode to my old bicycle.",
            "Script a 30-second radio ad for a local bakery.",
            "Describe a sunset so vividly a blind reader could feel it.",
            "Write a children's bedtime story about a shy thundercloud.",
            "Generate 15 YouTube title ideas for a bread-baking channel.",
            "Write a wedding vows template inspired by mountain climbing.",
            "Invent a myth explaining why cats knead blankets.",
            "Draft the back-cover blurb for a sci-fi novel about memory merchants.",
        ],
    },
    "multilingual": {
        "cards": [
            "A multilingual model strong in translation and cross-language work, with native-quality handling of many languages, idioms, and regional nuance.",
            "This assistant is optimized for language tasks: translating documents, localizing content, polishing non-English writing, and conversing fluently outside English.",
            "Built for global communication: accurate translation that preserves tone, cultural adaptation of marketing text, and dependable grammar across dozens of languages.",
            "A language specialist model for translating, transcreating, and chatting naturally in multiple languages, sensitive to register and idiom.",
        ],
        "prompts": [
            "Translate this contract clause from English to legal German.",
            "Przetłumacz ten wiersz na angielski zachowując rym.",
            "把这封商务邮件翻译成英文，语气要正式。",
            "Localize our app store description for the Japanese market.",
            "Traduis cet article de blog en anglais, ton professionnel.",
            "Translate this manga dialogue into natural English slang.",
            "Übersetze diese Homepage-Inhalte ins Englische, SEO-freundlich.",
            "How do you say 'it's raining cats and dogs' in Japanese, and is there an equivalent idiom?",
            "Traduzi esse e-mail para português do Brasil, tom informal.",
            "Translate this customer support ticket from Hindi to English.",
            "My Polish grandmother's recipe uses old units. Translate and modernize it.",
            "Check the grammar and naturalness of my Spanish cover letter.",
            "Переведи это стихотворение на английский, сохрани ритм.",
            "Translate our privacy policy into Simplified Chinese accurately.",
            "Which of these two Arabic translations sounds more formal?",
            "Translate this Italian song lyric and explain the wordplay.",
            "Localize our game UI strings for Korean, keeping them short.",
            " Traduis ce CV en anglais britannique.",
            "Write a formal business email in Korean requesting a meeting.",
            "Translate this court judgment excerpt from French to English.",
            "Jak przetłumaczyć 'przychodzi walczył z king kong' na angielski z sensem?",
            "Adapt this slogan for the German market without losing the pun.",
            "Translate my Airbnb listing into Spanish and French.",
            "Explain the difference between formal and informal address in this Swedish email.",
        ],
    },
    "docs": {
        "cards": [
            "A document-work model for reading and condensing long text: summaries, structured extraction, comparison across sources, and question answering grounded in the provided material.",
            "This assistant excels at taming walls of text: meeting notes, contracts, research papers, and reports turned into briefs, bullet points, tables, and action items.",
            "Built for summarization and grounded analysis of provided documents, careful to stick to the source and cite what it uses.",
            "A patient reader for lengthy material: extracts key numbers, obligations, dates, and decisions, and answers questions strictly from the text you give it.",
        ],
        "prompts": [
            "Summarize this 40-page PDF report into one page of key findings.",
            "Extract all payment terms and deadlines from this contract.",
            "Turn these messy meeting notes into structured minutes with action items.",
            "Compare these three vendor proposals and build a feature matrix.",
            "What are the main claims of this paper and what evidence supports them?",
            "Summarize this customer call transcript and flag churn-risk signals.",
            "Extract every date and obligation from this lease agreement into a table.",
            "Condense this 90-minute interview into 10 highlight quotes.",
            "Make a TL;DR of this legal ruling in plain language.",
            "Redact and summarize this medical case report for a lay audience.",
            "Extract the entities and relationships from this news article into JSON.",
            "Summarize this earnings call and list guidance changes vs last quarter.",
            "Which clauses in this ToS differ from the previous version?",
            "Build an FAQ from this product documentation chapter.",
            "Summarize this podcast transcript in 5 bullets with timestamps.",
            "What does this insurance policy actually cover? Quote the relevant sections.",
            "Condense this RFP into a checklist of mandatory requirements.",
            "Summarize these lecture slides into revision notes.",
            "Compare the two witness statements and note contradictions.",
            "Extract all action items and owners from this Slack thread export.",
            "Turn this recipe blog post into a clean structured recipe card.",
            "What commitments did each party make in this partnership memo?",
            "Summarize this regulatory filing's risk factors by severity.",
            "Create an executive brief from this quarterly business review deck text.",
        ],
    },
    "vision": {
        "cards": [
            "A multimodal model that understands images alongside text: screenshots, photos, diagrams, charts, handwriting, and UI mockups, reasoning about what it sees.",
            "This assistant can look at pictures and answer questions about them: extract text from photos, explain charts, identify objects, and debug UI from screenshots.",
            "Built for visual understanding: document scans, whiteboard photos, product images, plots, and anything where the answer is in the image.",
            "A vision-capable model that reads diagrams, compares image pairs, describes scenes in detail, and turns visual content into structured data.",
        ],
        "prompts": [
            "Here is a screenshot of my app. Why does the layout break on mobile?",
            "What does this error dialog say and how do I fix it? (image attached)",
            "Read the total amount from this receipt photo.",
            "Explain what this line chart about sales trends shows.",
            "Identify the plant in this photo and how to care for it.",
            "Transcribe this handwritten letter into clean text.",
            "Which of these two UI mockups has better accessibility? Explain visually.",
            "Describe this architecture diagram and list the components.",
            "What breed is this dog approximately?",
            "Convert this scanned form into JSON with all field values.",
            "Is this mole something a dermatologist should look at?",
            "Explain the joke in this comic strip.",
            "What's written on the store sign in this street photo, in Japanese?",
            "Analyze this heatmap: where are users clicking most?",
            "Compare these two X-rays and describe the differences.",
            "Extract the table from this photo of a printed page as CSV.",
            "What circuit is shown in this schematic and what does it do?",
            "Rate the plating of this dish photo for a food blog.",
            "What font is used in this logo, approximately?",
            "Tell me what's funny in this meme image.",
            "This is my room. Suggest furniture rearrangements with reasons.",
            "Read the barcode number in this image.",
            "Describe this satellite image: what land features do you see?",
            "What Excel formula would produce the result shown in this screenshot?",
        ],
    },
    "agentic": {
        "cards": [
            "An agentic model built for tool use and autonomous workflows: function calling, multi-step API orchestration, browser and shell tasks, and recovering when a step fails.",
            "This assistant plans and executes multi-step tasks with tools: calling APIs, querying databases, running code, and chaining results into a finished outcome.",
            "Optimized for agency: structured tool calls, long task horizons, state tracking between steps, and knowing when to ask for clarification versus act.",
            "A model for automation and orchestration: scaffolds plans, invokes tools with correct arguments, verifies outputs, and adapts when environments change.",
        ],
        "prompts": [
            "Book me the cheapest flight to Tokyo next month using the travel API.",
            "Search the web for the latest CUDA pricing and update my spreadsheet.",
            "Every morning at 8am, check these three RSS feeds and email me a digest.",
            "Here is an SQL database. Answer: which region grew fastest last year?",
            "Run this Python data pipeline and fix any errors it hits, then rerun.",
            "Use the weather and maps APIs to plan a photo route avoiding rain.",
            "Scrape competitor prices daily and alert me on drops over 10 percent.",
            "Call the CRM API to enrich these 50 leads with company details.",
            "Automate posting my blog updates to three social networks.",
            "Monitor this log stream and page me on error spikes.",
            "Chain these three internal APIs to onboard a new customer end to end.",
            "Take a natural-language request and generate the tool calls as JSON.",
            "Control a browser: log in, download the invoice, rename it by date.",
            "Build a workflow: new GitHub issue becomes a Jira ticket with labels.",
            "Query the stock API, compute moving averages, then chart the result.",
            "Batch-translate 200 product descriptions via the translation endpoint.",
            "When an order email arrives, extract items and update inventory.",
            "Orchestrate: fetch repo, run tests, if green then deploy to staging.",
            "Use web search to fact-check this article and annotate each claim.",
            "Manage my calendar: find a slot all five attendees share and send invites.",
            "Handle this customer refund flow across payment and email tools.",
            "Watch this directory and transcribe any new audio files.",
            "Pull Kaggle data, clean it, and post summary stats to Slack.",
            "Given this goal and these tools, produce a step-by-step execution plan.",
        ],
    },
    "data": {
        "cards": [
            "A data-analysis model comfortable with numbers: SQL, spreadsheets, statistics, experiment interpretation, and turning messy tables into charts and decisions.",
            "This assistant works well with structured data: writing queries, cleaning datasets, choosing the right statistical test, and explaining what the numbers actually mean.",
            "Built for analytics: cohort analysis, forecasting basics, A/B test readouts, pivot-table logic, and sanity checks that catch bad data before it misleads.",
            "A quantitative helper for data work: from quick spreadsheet formulas to regression diagnostics and clear executive summaries of findings.",
        ],
        "prompts": [
            "Write a SQL query for monthly retention cohorts from these tables.",
            "Which statistical test should I use for this A/B test and why?",
            "This pivot table shows a 3% lift. Is that significant?",
            "Clean this CSV: fix dates, split names, handle missing zip codes.",
            "Explain what a p-value of 0.03 means for my manager.",
            "Build a forecast for next quarter's sales from this series.",
            "What Excel formula computes weighted average margin per region?",
            "Interpret this regression output: which predictors matter?",
            "Find the anomaly in this daily traffic data.",
            "Create a cohort chart spec from this event log table.",
            "How do I detect and handle outliers before running the t-test?",
            "Write a pandas groupby showing week-over-week growth per category.",
            "Our churn rate rose 2 points. Walk me through a diagnostic plan.",
            "Convert this business question into a metric definition and SQL.",
            "Explain selection bias in my survey sample of 200 users.",
            "What sample size do I need to detect a 5% MDE at 95% confidence?",
            "Sanity check: does this revenue total match the sum of the regions?",
            "Design a dashboard layout for tracking a SaaS funnel.",
            "Compute CAC and LTV from this marketing spend and signup table.",
            "Is this correlation between ad spend and sales spurious?",
            "Write a query to find the second-highest salary per department.",
            "Explain Bayesian A/B testing vs frequentist for our use case.",
            "Normalize this table to 3NF and show the resulting schema.",
            "Which chart type best shows distribution shifts between two periods?",
        ],
    },
    "fast": {
        "cards": [
            "A fast, lightweight model tuned for speed and cost: great for high-volume, latency-sensitive work like autocomplete, classification, routing, and quick lookups where depth matters less.",
            "This assistant prioritizes instant responses at low price. Ideal for bulk processing, simple questions, formatting tasks, and short back-and-forth chat.",
            "An economical workhorse for scale: classification, extraction, short answers, and drafting at a fraction of frontier-model cost.",
            "Built for throughput: tiny latency, tiny bill. Perfect when you need a million decent answers rather than ten brilliant ones.",
        ],
        "prompts": [
            "Classify these 500 support tickets as bug, feature, or billing.",
            "Give me a one-line answer: capital of Australia?",
            "Autocomplete this product title: 'Wireless noise-cancelling...'",
            "Extract the company names from these 200 headlines.",
            "Quick: what's 15% tip on 84 dollars?",
            "Tag these customer reviews positive, negative, or neutral.",
            "Format these 50 addresses into a standard layout.",
            "Short answer: can ducks fly?",
            "Generate slug URLs for this list of article titles.",
            "Is this email spam or not? One word.",
            "Give me a random hex color.",
            "Deduplicate this keyword list.",
            "Pick the correct spelling: necessary, neccessary.",
            "What day of the week was Jan 1, 2020?",
            "Split these paragraphs into sentences at scale, 10k files.",
            "Translate 'thank you' into Spanish.",
            "Summarize this tweet in five words.",
            "Generate alt text for 300 product images in bulk.",
            "Correct the typos in this short note.",
            "Route this incoming message to sales or support.",
            "Count the words in this paragraph.",
            "Pick a random number between 1 and 100.",
            "What's the boiling point of water in Fahrenheit?",
            "Convert these 12 timestamps from UTC to EST.",
        ],
    },
    "education": {
        "cards": [
            "A patient tutoring model that explains step by step, adapts to the learner's level, checks understanding with questions, and builds intuition rather than just giving answers.",
            "This assistant is built for learning: clear explanations from basics upward, worked examples, analogies, common-mistake warnings, and encouragement without judgment.",
            "Designed for teaching: breaks concepts into digestible steps, adjusts difficulty, and turns homework struggles into understanding.",
            "A supportive academic coach for students: Socratic questioning, exam preparation plans, flashcard generation, and feedback on essays and proofs.",
        ],
        "prompts": [
            "Explain photosynthesis to a 10-year-old.",
            "I don't understand integrals. Start from the very beginning.",
            "Walk me through solving 2x + 5 = 15 like I'm 12.",
            "Why do we divide fractions by flipping them? Explain intuitively.",
            "Make me 20 flashcards for Spanish past tenses.",
            "Check my essay introduction and tell me how to improve it.",
            "Explain recursion using a real-life analogy.",
            "Quiz me on the French Revolution, one question at a time.",
            "What's the difference between weather and climate? Simple explanation.",
            "Help me build a 30-day plan to learn Python basics.",
            "Explain big-O notation with drawings described in words.",
            "I keep failing stoichiometry problems. Diagnose my misconception.",
            "Teach me the German cases with a memorable story.",
            "Explain how the internet works to my grandmother.",
            "Give me practice problems on sine and cosine at my level.",
            "Explain why the sky is blue without jargon.",
            "How should a beginner read a financial statement? Teach me.",
            "Turn this textbook chapter into a study guide with questions.",
            "Use the Socratic method to help me discover what inertia means.",
            "Explain DNA transcription like I'm watching a cooking show.",
            "Grade my paragraph and explain each correction kindly.",
            "What is the easiest way to memorize the periodic table groups?",
            "Explain supply and demand using sneaker reselling.",
            "Help me prepare for a chemistry final in two weeks.",
        ],
    },
    "long_context": {
        "cards": [
            "A long-context model able to hold entire books, codebases, or day-long transcripts in working memory at once, answering questions that require connecting details thousands of words apart.",
            "This assistant specializes in very large inputs: whole novels, complete legal files, and multi-hour conversation logs, without losing track of earlier content.",
            "Built for materials too big for ordinary models: cross-references facts across a full manuscript or repository and stays consistent from the first page to the last.",
            "An extended-context specialist that can ingest and reason over book-length documents in one pass, ideal for deep analysis of lengthy material.",
        ],
        "prompts": [
            "Here is my entire novel draft. Which characters' motivations contradict each other across chapters?",
            "Analyze this full quarterly earnings transcript and track every promise made about hiring.",
            "I'm pasting our complete codebase. Where is the authentication logic inconsistent?",
            "Compare the first and last chapters of this book for character growth.",
            "Here is a five-hour meeting transcript. List every decision and who made it.",
            "Review this whole contract and find clauses that conflict with each other.",
            "This is the full chat history from our project. Summarize how the requirements evolved.",
            "Read this entire scientific paper and check whether the conclusion follows from all the results sections.",
            "Here is the complete deposition. Which statements contradict the earlier testimony?",
            "Trace the hero's journey across the full trilogy text I've attached.",
            "Find every place in this lengthy manual where the safety instructions contradict the diagrams.",
            "Here is a year of my journal entries. What recurring themes do you see?",
            "Cross-reference this whole audit file and flag transactions that share counterparties.",
            "I've pasted the entire play. Map who is on stage in every scene.",
            "Analyze the full podcast back-catalog transcripts and describe how the host's views changed.",
            "Here is the complete medical history. Which symptoms cluster together over time?",
            "Check this whole Wikipedia article chain for statements that disagree with each other.",
            "Track the budget discussions across this full committee archive.",
            "Read the entire terms of service and privacy policy together and find conflicts.",
            "Here is the full research dataset description. Are any variable definitions inconsistent?",
            "Summarize the plot of this entire season's episodes I've transcribed.",
            "Which arguments in this full philosophical text are left unresolved?",
            "Here is our complete support ticket history. When did the bug first appear?",
            "Analyze this whole legal precedent file and extract the reasoning chain.",
        ],
    },
    "roleplay": {
        "cards": [
            "A roleplay model that stays in character with consistent personality, memory of the scene, and collaborative storytelling across long interactive sessions.",
            "This assistant is built for immersive character play: adopting personas, keeping voice consistent, and improvising dialogue in games, stories, and simulations.",
            "Designed for roleplay and simulation: plays fictional characters, historical figures, or custom personas convincingly, and reacts to unexpected player choices.",
            "A creative partner for interactive fiction and character scenarios, balancing dramatic tension with the user's direction.",
        ],
        "prompts": [
            "Pretend you are a medieval blacksmith. I walk into your shop.",
            "Let's roleplay: you are a hard-boiled detective, I'm your new partner.",
            "Act as my strict but fair wizard mentor in this adventure.",
            "You are the ship's AI on a stranded colony vessel. Begin.",
            "Roleplay a job interview where you are a skeptical CFO.",
            "Be my D&D dungeon master for a one-shot in a haunted forest.",
            "Pretend to be my future self from ten years ahead.",
            "You're a Roman senator in the last days of the Republic. Convince me.",
            "Let's do a customer service simulation. You're the unhappy client.",
            "Play my rival chef in a cooking competition scene.",
            "Act as a pessimistic AI assistant from a dystopian future.",
            "Roleplay a first date where you're an alien learning human customs.",
            "You are the Sorting Hat. Sort me, but argue with me first.",
            "Simulate a negotiation where you're a used car dealer.",
            "Be the ghost haunting my new apartment. Start the scene.",
            "Pretend we're old army buddies reunited after twenty years.",
            "You are my sarcastic sidekick in a superhero scenario.",
            "Play a strict Victorian governess interviewing me.",
            "Let's run a heist planning scene. You're the cautious safecracker.",
            "Act as a sentient sword I just pulled from a stone.",
            "Roleplay a podcast host interviewing me about my 'new book'.",
            "You're the final boss. Monologue before we battle.",
            "Simulate my therapist in a practice session for a difficult conversation.",
            "Be the narrator of a choose-your-own-adventure in feudal Japan.",
        ],
    },
    "image_gen": {
        "cards": [
            "An image generation model that creates pictures from text descriptions: illustrations, concept art, product photos, logos, and stylistic edits rendered from a prompt.",
            "This assistant produces original images: photorealistic scenes, paintings, character designs, and graphics, following detailed style and composition instructions.",
            "Built for visual creation: turning ideas into rendered artwork, iterating on layouts and styles, and generating visuals for articles, decks, and games.",
            "A text-to-image model for design work: mockups, textures, book covers, and marketing visuals, with control over mood, palette, and framing.",
        ],
        "prompts": [
            "Draw a watercolor fox reading a book under a lantern.",
            "Generate a logo concept for a mountain coffee brand.",
            "Create concept art of a derelict space station orbiting a gas giant.",
            "Make me a cozy pixel-art bedroom scene.",
            "Illustrate a children's book page: a brave little tugboat.",
            "Generate a hero image for my blog post about deep work.",
            "Draw this character sheet: a ranger with a mechanical hawk.",
            "Create a product mockup of a matte black water bottle on stone.",
            "Paint a cyberpunk street market in the rain, neon reflections.",
            "Generate a cute mascot for my python coding channel.",
            "Make an isometric illustration of a data center.",
            "Draw a tarot card of The Programmer.",
            "Create album cover art for an ambient music EP, minimalist.",
            "Generate a pattern of interlocking leaves for fabric printing.",
            "Illustrate the scene from my story: the lighthouse in the storm.",
            "Make a stylized portrait of my cat as a renaissance noble.",
            "Generate textures for a brick wall, seamless tile.",
            "Draw a map of a fantasy city with distinct districts.",
            "Create an infographic-style illustration of a coffee journey.",
            "Generate a background of soft gradient waves for my slides.",
            "Paint the same valley in four seasons, side by side.",
            "Design a sticker of an astronaut drinking bubble tea.",
            "Make line art of a bicycle for a coloring book.",
            "Generate a fake screenshot of a retro operating system desktop.",
        ],
    },
    "audio": {
        "cards": [
            "An audio model for speech and sound: transcription of recordings, voice synthesis, meeting notes from audio, and analysis of music or spoken content.",
            "This assistant works with sound: turning speech into accurate text with speaker labels, generating natural voices, and summarizing what was said in long recordings.",
            "Built for voice tasks: dictation, subtitles, podcast transcription, pronunciation feedback, and text-to-speech with natural pacing.",
            "A speech-capable model that handles interviews, lectures, and calls: precise transcripts, timestamps, and who-said-what attribution.",
        ],
        "prompts": [
            "Transcribe this hour-long interview with speaker labels.",
            "Generate subtitles for my tutorial video, with timestamps.",
            "Summarize what was decided in this meeting recording.",
            "Turn this voicemail into text and flag the callback number.",
            "Write a narration script and read it in a calm documentary voice.",
            "Transcribe this lecture and mark where the slides change.",
            "Which customer objections come up most in these sales calls?",
            "Convert my podcast episode into a blog post draft from the audio.",
            "Read this bedtime story aloud in a soothing voice.",
            "Transcribe and translate this Spanish voice note to English.",
            "Extract the action items from this recorded standup.",
            "What song structure does this track follow? Verse, chorus, bridge.",
            "Generate a voiceover for my product demo, energetic tone.",
            "Take dictation while I brainstorm, and clean it up as you go.",
            "Timestamp every mention of 'budget' in this earnings call.",
            "Transcribe these focus group recordings and group responses by theme.",
            "Read my French homework sentences aloud slowly for practice.",
            "Analyze this call center audio: how often do agents interrupt?",
            "Make a transcript of this sermon with paragraph breaks.",
            "Convert this dictation into a formatted medical note.",
            "Transcribe the lyrics of this song demo accurately.",
            "Summarize this audiobook chapter by chapter.",
            "Detect the moments the customer sounded frustrated in this call.",
            "Narrate my meditation script with long, calm pauses.",
        ],
    },
    "research": {
        "cards": [
            "A research model grounded in sources: literature reviews, citation-aware answers, claim verification, and structured synthesis across many references.",
            "This assistant is built for scholarly work: finding and weighing evidence, comparing studies, flagging disagreements in the literature, and citing what it uses.",
            "Designed for investigation with receipts: traces claims to their origin, distinguishes consensus from contested findings, and lays out conflicting evidence fairly.",
            "A careful research partner for questions that need sources: systematic summaries, fact-checking, and reading lists organized by subtopic.",
        ],
        "prompts": [
            "What does the current literature say about intermittent fasting and muscle gain?",
            "Find sources comparing monorepo vs polyrepo outcomes and summarize.",
            "Fact-check this viral claim about microplastics in bottled water.",
            "Do the studies agree on spaced repetition for language learning?",
            "Build me a reading list on transformer alternatives, ordered by influence.",
            "What's the evidence for and against cold showers improving mood?",
            "Compare the methodology of these three climate studies.",
            "Has any peer-reviewed work replicated this psychology finding?",
            "What did commentators across the spectrum say about this policy?",
            "Cite sources for the claim that remote work reduced productivity.",
            "Summarize the state of research on caffeine and anxiety.",
            "Which papers first introduced attention mechanisms? Trace the lineage.",
            "Verify these five statistics from my draft article.",
            "What are the strongest objections to this paper's methodology?",
            "Research the history of the eight-hour workday with citations.",
            "Is there consensus on minimum effective exercise duration?",
            "Gather evidence on whether four-day work weeks affect output.",
            "What studies exist on screen time and toddler language delay?",
            "Cross-check this biography's claims against historical records.",
            "Compile what surveys say about developer satisfaction by language.",
            "Find contradictory studies on the minimum wage employment debate.",
            "What did clinical trials conclude about this supplement?",
            "Trace the origin of the ten-thousand-steps-a-day figure.",
            "Research whether urban trees measurably reduce summer temperatures.",
        ],
    },
}

# Sensible cross-capability scores: prompt archetype -> (card archetype, score)
ADJACENCIES: dict[str, list[tuple[str, float]]] = {
    "reasoning": [("coding", 0.4), ("data", 0.4), ("education", 0.4), ("research", 0.4)],
    "coding": [("reasoning", 0.4), ("agentic", 0.35), ("fast", 0.2)],
    "general": [("casual", 0.35), ("education", 0.3), ("docs", 0.3)],
    "casual": [("general", 0.35), ("multilingual", 0.25), ("creative", 0.25), ("roleplay", 0.3)],
    "creative": [("general", 0.35), ("casual", 0.25), ("roleplay", 0.35)],
    "multilingual": [("casual", 0.3), ("general", 0.3)],
    "docs": [("data", 0.3), ("general", 0.3), ("reasoning", 0.25), ("long_context", 0.5)],
    "vision": [("general", 0.2), ("image_gen", 0.2)],
    "agentic": [("coding", 0.4), ("data", 0.3)],
    "data": [("reasoning", 0.4), ("coding", 0.35)],
    "fast": [("general", 0.3), ("casual", 0.25)],
    "education": [("general", 0.4), ("reasoning", 0.35)],
    "long_context": [("docs", 0.5), ("reasoning", 0.3)],
    "roleplay": [("casual", 0.4), ("creative", 0.4)],
    "image_gen": [("vision", 0.25)],
    "audio": [("vision", 0.2), ("multilingual", 0.2)],
    "research": [("reasoning", 0.5), ("docs", 0.4)],
}

# Keyword rules to anchor real (scrubbed) OpenRouter descriptions to archetypes.
# First match wins; order matters (specific modality rules before general ones).
REAL_DESC_RULES: list[tuple[str, list[str]]] = [
    ("image_gen", ["image generation", "text-to-image", "generate images", "image model", "illustration", "diffusion", "art generation"]),
    ("audio", ["audio", "speech", "voice", "transcription", "music", "tts", "asr"]),
    ("vision", ["vision", "image", "multimodal", "screenshot", "photo", "ocr", "video"]),
    ("agentic", ["tool", "function calling", "agent", "autonomous", "orchestrat"]),
    ("long_context", ["long context", "extended context", "large context", "million token"]),
    ("research", ["research", "citation", "literature", "academic", "sources"]),
    ("coding", ["coding", "code", "software", "developer", "programming", "debug"]),
    ("reasoning", ["reasoning", "math", "science", "olympiad", "proof", "think"]),
    ("docs", ["summariz", "document", "retrieval", "rag", "context window"]),
    ("multilingual", ["multilingual", "translat", "languages", "language model for"]),
    ("creative", ["creative", "writing", "story", "poetry", "copywriting", "novel"]),
    ("data", ["sql", "data analysis", "analytics", "spreadsheet", "statistic"]),
    ("roleplay", ["roleplay", "role-play", "persona", "companion", "character"]),
    ("fast", ["fast", "affordable", "lightweight", "cheap", "speed", "efficient", "low cost"]),
    ("casual", ["chat", "conversation", "companionship", "everyday chat", "small talk"]),
    ("education", ["tutor", "education", "teaching", "student", "explain"]),
    ("general", ["general", "versatile", "everyday", "balanced", "wide range", "broad"]),
]

# Same scrubbing rules as build_scrubbed_arena.py so real descriptions and
# synthetic cards share one text style (no model or provider names).
MODEL_ROOTS = [
    "gpt", "chatgpt", "o1", "o3", "o4", "claude", "sonnet", "opus", "haiku",
    "llama", "qwen", "mistral", "mixtral", "magistral", "gemma", "gemini",
    "phi", "grok", "yi", "command", "nemotron", "deepseek", "kimi",
    "minimax", "nova", "hunyuan", "ernie", "glm",
]
PROVIDERS = [
    "OpenAI", "Anthropic", "Google", "Meta", "Mistral", "Cohere", "Qwen",
    "Microsoft", "Alibaba", "xAI", "Tencent", "DeepSeek", "MoonshotAI",
    "Moonshot", "MiniMax", "Amazon", "Nvidia", "Salesforce", "AI21", "Z.ai",
]
ROOTS_PATTERN = re.compile(r"(?i)\b(?:" + "|".join(MODEL_ROOTS) + r")[a-zA-Z0-9\-\.]*")
PROVIDER_PATTERN = re.compile(r"(?i)\b(?:" + "|".join(PROVIDERS) + r")\b")

# Product-name residue that survives scrubbing looks like "This model 2.5
# Flash" or "5.6 Sol". Words containing a digit or a model-suffix word are
# deleted from the description (not the whole entry). Word boundaries keep
# normal English ("production", "maximum", "professional") safe.
MODEL_SUFFIX_WORDS = (
    "flash", "pro", "super", "ultra", "plus", "max", "mini", "nano", "turbo", "lite",
)
RESIDUAL_NAME_PATTERN = re.compile(
    r"(?i)\b(?:" + "|".join(MODEL_SUFFIX_WORDS) + r")\b|\d"
)

# Abstract codenames ("Sol", "Argon", "Fable", "Glimpse") are neither digits
# nor suffix words, so the cleaner also uses a kill-list built from the actual
# catalog: every alphabetic token (>= 3 chars) that appears in any model id or
# display name, minus capability words that legitimately occur in names and
# must survive in descriptions.
NAME_WORDS_TO_KEEP = {
    "chat", "coder", "code", "coding", "vision", "video", "audio", "voice",
    "music", "reasoning", "reasoner", "thinking", "thought", "instruct",
    "instruction", "latest", "preview", "experimental", "online", "base",
    "search", "assistant", "expert", "app", "guard", "smart", "open",
    "science", "math", "deep", "think", "web", "text", "image",
    "agent", "agents",
}

VOCAB_COMMON_WORD_EXEMPT = 3.5  # zipf scale: ordinary English never enters the kill-list

# Words protected from the rare-English-word deletion pass below: capability
# and domain vocabulary that frequency lists rank low but must never be cut.
DOMAIN_ALLOWLIST = {
    "multimodal", "agentic", "orchestration", "orchestrate", "autonomous",
    "transcription", "transcribe", "summarization", "translation",
    "localization", "multilingual", "autocomplete", "refactoring",
    "debugging", "latency", "throughput", "inference", "embedding",
    "embeddings", "tokenizer", "tokenization", "benchmark", "benchmarks",
    "workhorse", "olympiad", "combinatorics", "brainstorming", "persona",
    "personas", "roleplay", "fanfiction", "screenplay", "screenplays",
    "mockup", "mockups", "infographic", "infographics", "isometric",
    "watercolor", "photorealistic", "dystopian", "renaissance", "tarot",
    "monologue", "voicemail", "dictation", "narration", "narrator",
    "audiobook", "podcast", "deposition", "stoichiometry", "photosynthesis",
    "recursion", "calculus", "spreadsheet", "spreadsheets", "cohort",
    "cohorts", "churn", "onboarding", "workload", "workloads", "codebase",
    "codebases", "monorepo", "polyrepo", "microservices", "webinar",
    "copywriting", "workflow", "workflows", "socratic", "study",
    "consistently", "dependable", "reliable", "trustworthy",
}

# Rare-word detection is optional so the script still runs without the
# `wordfreq` package installed (it is in requirements.txt).
try:
    from wordfreq import zipf_frequency as _zipf_frequency
except ImportError:  # pragma: no cover
    _zipf_frequency = None

RARE_WORD_THRESHOLD = 3.0  # zipf scale; ~1 occurrence per few thousand words


def build_codename_vocabulary(profiles: dict[str, dict[str, Any]]) -> frozenset[str]:
    """All alphabetic tokens that appear in any model id or display name.

    Ordinary English words (zipf >= 3.5) are exempted: deleting "agent" or
    "apply" from descriptions would mangle capability text, and real model
    names are already handled by the root/provider scrubbing.
    """
    vocab: set[str] = set()
    for model_id, profile in profiles.items():
        for token in re.split(r"[^a-zA-Z]+", f"{model_id} {profile.get('name', '')}".lower()):
            if len(token) < 3 or token in NAME_WORDS_TO_KEEP:
                continue
            if (
                _zipf_frequency is not None
                and _zipf_frequency(token, "en") >= VOCAB_COMMON_WORD_EXEMPT
            ):
                continue
            vocab.add(token)
    return frozenset(vocab)


def is_rare_english_word(word: str) -> bool:
    """True for uncommon English words - likely abstract model codenames.

    Only lowercase alphabetic words of 4+ characters are candidates;
    hyphenated tokens, capitalized words and acronyms, and domain vocabulary
    are never deleted by this rule.
    """
    if _zipf_frequency is None:
        return False
    if len(word) < 4 or not word.isalpha() or not word.islower():
        return False
    if word in DOMAIN_ALLOWLIST:
        return False
    return _zipf_frequency(word, "en") < RARE_WORD_THRESHOLD


def strip_residual_tokens(desc: str, codenames: frozenset[str] = frozenset()) -> str:
    """Delete whitespace-delimited words that contain a digit, a model-suffix
    word, a catalog codename, or a rare non-dictionary word:
    "This model 2.5 Sol Flash is..." -> "This model is..."."""
    kept = [
        word for word in desc.split()
        if not any(char.isdigit() for char in word)
        and not RESIDUAL_NAME_PATTERN.search(word)
        and word.lower() not in codenames
        and not is_rare_english_word(word)
    ]
    cleaned = " ".join(kept)
    # Lone leftover letters from names like "Command A" -> "This model A"
    cleaned = re.sub(r"(?i)(this model) [a-z] (?=[a-z])", r"\1 ", cleaned)
    return cleaned


def scrub_description(desc: str, display_name: str = "") -> str:
    """Remove model names and providers from a description."""
    desc = ROOTS_PATTERN.sub("This model", desc)
    desc = PROVIDER_PATTERN.sub("its developer", desc)
    if len(display_name) > 3:
        desc = re.sub(re.escape(display_name), "This model", desc, flags=re.IGNORECASE)
    return desc


def classify_description(desc: str) -> str | None:
    text = desc.lower()
    for archetype, keywords in REAL_DESC_RULES:
        if not keywords:
            continue
        if any(k in text for k in keywords):
            return archetype
    return None


# Strict keyword rules for labeling real prompts mined from the arena CSV.
# Ambiguous prompts match nothing and are skipped - mislabeled prompts are
# worse than fewer prompts. Order: specific before general.
PROMPT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("data", re.compile(r"(?i)\b(sql|dataframe|dataset|regression|p-value|spreadsheet|statistics|pivot table)\b")),
    ("image_gen", re.compile(r"(?i)\b(draw|sketch|illustrat\w*|generate an image|create an image|digital art|painting of)\b")),
    ("audio", re.compile(r"(?i)\b(transcrib\w*|transcription|text to speech|voice ?over)\b")),
    ("multilingual", re.compile(r"(?i)\b(translat\w*|przetłumacz|翻译|übersetz\w*|traduis|переведи|traduc\w*)\b")),
    ("docs", re.compile(r"(?i)\b(summar\w*|tl;?dr|key points)\b")),
    ("research", re.compile(r"(?i)\b(citations?|literature|peer.?reviewed|fact.?check\w*)\b")),
    ("coding", re.compile(r"(?i)\b(code|coding|python|javascript|typescript|java\b|c\+\+|rust\b|golang|api|bug|debug|function|regex|docker|kubernetes|git\b|compile|css|html|react|numpy|pytorch|tensorflow|programming|script)\b")),
    ("creative", re.compile(r"(?i)\b(story|poem|poetry|lyrics|screenplay|slogan|tagline|novel|fiction|write a song)\b")),
    ("roleplay", re.compile(r"(?i)\b(roleplay|role-play|pretend you are|act as)\b")),
    ("long_context", re.compile(r"(?i)(entire (book|transcript|document|codebase|novel)|whole (book|transcript|document)|full document)")),
    ("reasoning", re.compile(r"(?i)\b(prove|equation|derivative|integral|theorem|probability|how many ways|puzzle)\b")),
    ("education", re.compile(r"(?i)\b(explain like|eli5|teach me|for beginners|simple terms)\b")),
    ("casual", re.compile(r"(?i)\b(joke|funny|bored|chat with me|opowiedz|żart|how are you)\b")),
]


def classify_prompt(prompt: str) -> str | None:
    for archetype, pattern in PROMPT_RULES:
        if pattern.search(prompt):
            return archetype
    return None


def main() -> None:
    rng = random.Random(SEED)

    # Normalise the synthetic cards with the same residue-stripping the real
    # descriptions get (a no-op for the shipped cards, but keeps future edits
    # from quietly reintroducing name-like tokens).
    for spec in ARCHETYPES.values():
        spec["cards"] = [strip_residual_tokens(c) for c in spec["cards"]]

    archetypes = list(ARCHETYPES)
    rows: list[list[Any]] = []

    # 1. Synthetic capability cards: by-construction labels
    for arch, spec in ARCHETYPES.items():
        others = [a for a in archetypes if a != arch]
        adjacent = dict(ADJACENCIES.get(arch, []))
        for prompt in spec["prompts"]:
            rows.append([prompt, rng.choice(spec["cards"]), 1.0])
            for other in rng.sample(others, 4):
                rows.append([prompt, rng.choice(ARCHETYPES[other]["cards"]), 0.0])
            for other, score in adjacent.items():
                rows.append([prompt, rng.choice(ARCHETYPES[other]["cards"]), score])

    synthetic_count = len(rows)

    # 2. Real OpenRouter descriptions, scrubbed and keyword-anchored
    anchored = 0
    residue_stripped = 0
    if PROFILES_PATH.is_file():
        with PROFILES_PATH.open("r", encoding="utf-8") as f:
            profiles: dict[str, dict[str, Any]] = json.load(f)
        codenames = build_codename_vocabulary(profiles)
        for model_id, profile in profiles.items():
            short_id = model_id.split(":")[0]
            if short_id != model_id:  # skip :batch/:free twins of the same model
                continue
            desc = str(profile.get("description") or "").strip()
            if len(desc) < 30:
                continue
            arch = classify_description(desc)
            if arch is None:
                continue
            desc = scrub_description(desc, str(profile.get("name") or ""))
            if not desc:
                continue
            stripped = strip_residual_tokens(desc, codenames)
            if stripped != desc:
                residue_stripped += 1
            desc = stripped
            if len(desc) < 40:  # stripping can gut a description; keep usable ones only
                continue
            anchored += 1
            for prompt in rng.sample(ARCHETYPES[arch]["prompts"], 2):
                rows.append([prompt, desc, 1.0])
            others = [a for a in archetypes if a != arch]
            for other in rng.sample(others, 2):
                prompt = rng.choice(ARCHETYPES[other]["prompts"])
                rows.append([prompt, desc, 0.0])
    else:
        print(f"[profiles] {PROFILES_PATH} not found - skipping real-description anchoring")

    # 3. Real prompts mined from the arena dataset (140k battles), labeled
    # with strict keyword rules. Prompts that mention any model by name are
    # about models, not routing examples, and are skipped.
    arena_csv = PROJECT_ROOT / "data" / "router_training_pairs_scrubbed.csv"
    arena_prompt_counts: dict[str, int] = {}
    if arena_csv.is_file():
        seen_prompts: set[str] = set()
        arena_prompts: dict[str, list[str]] = {a: [] for a in ARCHETYPES}
        with arena_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if not row:
                    continue
                prompt = row[0].strip()
                if len(prompt) < 15 or prompt in seen_prompts:
                    continue
                seen_prompts.add(prompt)
                if ROOTS_PATTERN.search(prompt) or PROVIDER_PATTERN.search(prompt):
                    continue
                archetype = classify_prompt(prompt)
                if archetype is not None and len(arena_prompts[archetype]) < 40:
                    arena_prompts[archetype].append(prompt)

        for archetype, prompts in arena_prompts.items():
            if not prompts:
                continue
            arena_prompt_counts[archetype] = len(prompts)
            others = [a for a in archetypes if a != archetype]
            adjacent = dict(ADJACENCIES.get(archetype, []))
            for prompt in prompts:
                rows.append([prompt, rng.choice(ARCHETYPES[archetype]["cards"]), 1.0])
                for other in rng.sample(others, 4):
                    rows.append([prompt, rng.choice(ARCHETYPES[other]["cards"]), 0.0])
                for other, score in adjacent.items():
                    rows.append([prompt, rng.choice(ARCHETYPES[other]["cards"]), score])
        print(f"Mined real arena prompts per archetype: {arena_prompt_counts}")
    else:
        print(f"[arena] {arena_csv} not found - skipping real-prompt mining")

    rng.shuffle(rows)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence1", "sentence2", "label"])
        writer.writerows(rows)

    labels: dict[float, int] = {}
    for row in rows:
        labels[row[2]] = labels.get(row[2], 0) + 1
    print(f"Success! Generated {len(rows)} capability-matched pairs "
          f"({synthetic_count} synthetic + {anchored} anchored real descriptions; "
          f"residue tokens stripped from {residue_stripped} of them).")
    print(f"Label distribution: {dict(sorted(labels.items()))}")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
