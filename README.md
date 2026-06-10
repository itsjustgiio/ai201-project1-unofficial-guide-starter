# The Unofficial Guide - Project 1

## Domain

This project is a Retrieval-Augmented Generation system for student-facing CCNY information. The domain is student-generated and student-useful knowledge about The City College of New York, including Reddit-style student advice, professor reviews, advising complaints, course workload discussions, club resources, and student activities information.

This knowledge is valuable because official college pages usually explain policies and programs, but they do not fully capture what students ask each other day to day: which classes feel difficult, how registration actually goes, what professor reviews emphasize, how students balance clubs with coursework, and what involvement opportunities are worth exploring. A RAG system is a good fit because answers should come from collected documents, not from the model's general memory.

---

## Document Sources

I collected 10 local text documents in the `documents/` folder. Some are manually copied Reddit or Rate My Professors content, and others are CCNY resource pages or handbook text.

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Beaver Handbook 2025-2026 | Club handbook text export | `documents/beaver_handbook_2025_2026.txt` |
| 2 | CCNY CS Department | Official department information | `documents/ccny_cs_department.txt` |
| 3 | CCNY CS Faculty Advising | Official advising information | `documents/ccny_cs_faculty_advising.txt` |
| 4 | CCNY advisors discussion | Reddit thread copied manually | `documents/reddit_ccny_advisors_unhelpful.txt` |
| 5 | CCNY calculus discussion | Reddit thread copied manually | `documents/reddit_ccny_calc_discussions.txt` |
| 6 | CCNY student experience discussion | Reddit thread copied manually | `documents/reddit_ccny_really_like.txt` |
| 7 | General CUNY student threads | Reddit threads copied manually | `documents/reddit_cuny_general_student_threads.txt` |
| 8 | Transfer freshman CS path | Reddit thread copied manually | `documents/reddit_transfer_freshman_cs_ccny.txt` |
| 9 | CCNY CS professor listings | Rate My Professors copied manually | `documents/rmp_ccny_cs_professor_listings.txt` |
| 10 | CCNY school reviews | Rate My Professors copied manually | `documents/rmp_ccny_school_reviews.txt` |

---

## Chunking Strategy

**Chunk size:** 450 words per chunk.

**Overlap:** 65 words.

**Why these choices fit your documents:** The corpus mixes short student opinions with longer resources such as the Beaver Handbook. A 450-word chunk is large enough to keep a student comment, review theme, or handbook section understandable on its own, while still being focused enough for semantic retrieval. The 65-word overlap helps preserve context when a discussion or handbook section crosses a chunk boundary.

**Preprocessing:** `ingest_and_chunk.py` strips HTML tags, removes script/style blocks, unescapes HTML entities, normalizes spacing and smart punctuation, removes common navigation/cookie/share boilerplate, and removes repeated short lines that look like site headers or footers.

**Final chunk count:** 64 chunks across 10 documents.

The chunk inspection report was written to `data/reports/chunk_inspection.md`. A few early handbook chunks still contain table-of-contents material, but the representative chunks also include substantive club policy, event planning, student organization, and professor review content.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` through `sentence-transformers`.

This model was chosen because it runs locally, does not require an API key, is fast enough for a small class project, and is commonly used for lightweight semantic search. The embedding script stores chunks in ChromaDB with metadata for source ID, source path, title, chunk index, and word positions.

I used a cosine-distance ChromaDB collection named `ccny_unofficial_guide_chunks_cosine` and normalized embeddings before indexing and querying. This made the distance scores much more useful than the first default-distance test.

**Production tradeoff reflection:** If this system were being deployed for real CCNY students, I would compare embedding models based on retrieval accuracy for informal student language, latency, cost, context length, and multilingual support. Multilingual support could matter because CCNY has a diverse student body, and stronger models might retrieve better results from slang, nicknames, professor names, and short review-style text. I would also consider whether to use a hosted embedding model for higher accuracy or keep a local model for privacy, cost control, and simpler deployment.

---

## Grounded Generation

**System prompt grounding instruction:**

```text
You are a grounded question-answering assistant for a CCNY unofficial guide.

You must follow these rules:
1. Answer using only the provided retrieved document chunks.
2. Do not use outside knowledge, even if you think you know the answer.
3. If the chunks do not contain enough information, say exactly: "I don't have enough information on that."
4. Do not invent professor names, course details, policies, dates, or student opinions.
5. Keep the answer concise and specific to the retrieved evidence.
```

The generation layer is implemented in `query.py`. It retrieves chunks first, formats only those chunks into the LLM prompt, and uses Groq's `llama-3.3-70b-versatile` with `temperature=0`. It also checks retrieval distance before calling the LLM; if the best match is above `0.5`, it returns the fallback answer instead of asking the model to guess from weak context.

**How source attribution is surfaced in the response:** Source attribution is appended programmatically from retrieved metadata rather than relying on the LLM to cite sources correctly. The Gradio app displays the answer in one box and a separate "Retrieved from" box listing each retrieved source, file path, chunk number, and distance score.

**Local test note:** End-to-end LLM generation was not run in this workspace because no `.env` file with `GROQ_API_KEY` is present. Retrieval, indexing, chunking, and the grounded fallback behavior were tested locally.

---

## Evaluation Report

I ran the five planned questions from `planning.md` against the Chroma retriever after indexing 64 chunks. The table below records retrieval behavior and the answer the system would be able to support from retrieved evidence. Generation still requires adding `GROQ_API_KEY` to `.env`.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What advice do current students give incoming CCNY CS freshmen? | Register early and stay on top of coursework. | Retrieved a transfer/CS freshman thread mentioning Transfer Explorer, course mapping, CCNY's lab-focused CS structure, and asking advisors for help. Also retrieved a student experience thread recommending networking, study groups, and choosing professors carefully. | Partially relevant | Partially accurate |
| 2 | What complaints do students have about registration? | Limited seats and scheduling conflicts. | The grounded system declined: "I don't have enough information on that." Retrieval was pulled toward Beaver Handbook club-registration chunks instead of course-registration complaints. | Off-target | Inaccurate |
| 3 | What traits do students praise in highly rated professors? | Clear explanations and helpful feedback. | Retrieved professor listing data and a student thread saying professor choice matters, some professors provide support, and lower-level CS courses have better professors than some later courses. | Partially relevant | Partially accurate |
| 4 | What opportunities do CCNY clubs provide? | Leadership, networking, and campus involvement. | Retrieved Beaver Handbook chunks explaining that clubs provide leadership development, communication/teamwork practice, networking, friendships, mentorship, events, funding, meeting space, and campus involvement. | Relevant | Accurate |
| 5 | What challenges do students face balancing classes and clubs? | Time management and heavy workloads. | Retrieved handbook chunks warning that over-dedication to club activities can affect academic performance and recommending planning meetings, goals, schedules, and member availability carefully. | Relevant | Accurate |

**Retrieval quality:** Mostly relevant, with one clear failure on the registration question.  
**Response accuracy:** Accurate for club questions, partially accurate for CS/professor questions, and inaccurate for the registration question.

---

## Failure Case Analysis

**Question that failed:** What complaints do students have about registration?

**What the system returned:** `I don't have enough information on that.`

**Root cause (tied to a specific pipeline stage):** The failure happens at the retrieval stage. The query uses the word "registration," and the corpus contains many Beaver Handbook chunks about club registration, club forms, and CampusGroups. Those chunks outranked the one school-review chunk that actually mentions course-registration complaints such as required classes not being offered sufficiently, classes meeting at the same time, and relying on e-permit. Because the best retrieved distance was above the confidence threshold and the context was off-target, `query.py` correctly declined instead of generating a shaky answer.

**What you would change to fix it:** I would add more course-registration-specific documents, such as student threads about course seats, CUNYfirst enrollment, waitlists, and scheduling conflicts. I would also consider adding metadata filters or query rewriting so "registration" can be interpreted as course registration instead of club registration when the question mentions classes, seats, schedules, or enrollment.

---

## Spec Reflection

**One way the spec helped you during implementation:** The planning document kept the pipeline concrete. Because it named the embedding model, top-k value, ChromaDB vector store, chunk size range, and overlap range, the implementation could be split cleanly into ingestion/chunking, embedding/retrieval, and grounded generation files. It also forced the code to preserve metadata early, which matters later for source attribution in the interface.

**One way your implementation diverged from the spec, and why:** The plan originally described a mix of Reddit, Rate My Professors, club directory, student activities, and handbook sources. The final corpus uses those categories, but some sources are manually copied text exports instead of live scraped pages because Reddit and Rate My Professors content can be difficult to scrape reliably. I also changed the Chroma setup to use cosine distance with normalized embeddings after the first retrieval test produced high distance scores.

---

## AI Usage

**Instance 1**

- *What I gave the AI:* I gave the AI the Milestone 3 instructions, the `planning.md` Documents section, the Chunking Strategy section, and the architecture diagram.
- *What it produced:* It produced `ingest_and_chunk.py`, a local document pipeline that discovers `.txt`, `.md`, `.html`, `.htm`, and `.pdf` files, saves raw text, cleans content, chunks text, prints inspection samples, and writes `chunks.jsonl`.
- *What I changed or overrode:* I made the chunking numbers concrete at 450 words with 65 words overlap because those values sit inside the planned ranges. I also made the script stop when no documents exist instead of generating placeholder data.

**Instance 2**

- *What I gave the AI:* I gave the AI the Retrieval Approach section from `planning.md` and asked it to implement embedding and ChromaDB retrieval using `all-MiniLM-L6-v2` and top-k retrieval.
- *What it produced:* It produced `embed_and_retrieve.py`, which loads chunks, embeds them with `sentence-transformers`, stores them in ChromaDB with source metadata, and prints retrieved chunks with distance scores for test queries.
- *What I changed or overrode:* I adjusted retrieval so it checks whether the Chroma collection is empty before loading the embedding model, loads the model from the local cache when possible, uses cosine distance, and normalizes embeddings for more meaningful scores.

**Instance 3**

- *What I gave the AI:* I gave the AI the Milestone 5 grounding requirement, the desired output format of answer plus source list, and the requirement to build a simple Gradio interface.
- *What it produced:* It produced `query.py` for grounded generation through Groq and `app.py` for a Gradio web UI.
- *What I changed or overrode:* I made source attribution programmatic in Python instead of depending on the LLM to cite sources. I also tightened the weak-retrieval guard so the system says it does not have enough information when retrieval distance is above `0.5`.

---

## Demo Notes

To run the project:

```powershell
python ingest_and_chunk.py
python embed_and_retrieve.py index
python app.py
```

Then open `http://127.0.0.1:7860`.

Before recording the final 3-5 minute demo, create a `.env` file from `.env.example` and add `GROQ_API_KEY`. The demo should show at least three queries with the "Retrieved from" source box visible, including one strong club-related answer and the registration failure case where the system declines because retrieval is off-target.
