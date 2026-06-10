# The Unofficial Guide - Project 1

## Domain

This project is a Retrieval-Augmented Generation system for student-facing CCNY information. The intended domain is student-generated and student-useful knowledge about The City College of New York, including Reddit-style student advice, professor reviews, registration complaints, workload discussions, club resources, and student activities information.

This knowledge is valuable because official college pages usually explain policies and programs, but they do not fully capture what students ask each other day to day: which classes feel difficult, how registration actually goes, what professor reviews emphasize, how students balance clubs with coursework, and what involvement opportunities are worth exploring. A RAG system is a good fit because those answers should come from collected documents, not from the model's general memory.

Important current limitation: the code pipeline is implemented, but the `documents/` folder currently contains no real source documents beyond `.gitkeep`. Because of that, the system cannot yet produce grounded answers.

---

## Document Sources

The planned sources from `planning.md` are listed below. The files have not yet been collected into `documents/`, so the current corpus size is zero documents.

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | CCNY Reddit - Campus Life | Student discussion | `documents/ccny_reddit_campus_life.txt` |
| 2 | CCNY Reddit - CS Advice | Student discussion | `documents/ccny_reddit_cs_advice.txt` |
| 3 | CCNY Reddit - Registration | Student discussion | `documents/ccny_reddit_registration.txt` |
| 4 | CCNY Reddit - Professors | Student discussion | `documents/ccny_reddit_professors.txt` |
| 5 | CCNY Reddit - Workload | Student discussion | `documents/ccny_reddit_workload.txt` |
| 6 | Rate My Professors Reviews | Professor reviews | `documents/rmp_general_reviews.txt` |
| 7 | CS Professor Ratings | Professor reviews | `documents/rmp_cs_professor_reviews.txt` |
| 8 | Clubs Directory | Club resource | `documents/clubs_directory.txt` |
| 9 | Student Activities Resources | Campus involvement resource | `documents/student_activities.txt` |
| 10 | 2025-2026 Club Handbook | PDF handbook | `documents/beaver_handbook_2025_2026.pdf` |

The ingestion script supports `.txt`, `.md`, `.html`, `.htm`, and `.pdf` files, so these planned sources can be added as plain text exports or PDFs before rerunning the pipeline.

---

## Chunking Strategy

**Chunk size:** 450 words per chunk.

**Overlap:** 65 words.

**Why these choices fit your documents:** The planned corpus mixes short student opinions with longer resources such as a club handbook. A 450-word chunk is large enough to keep a student comment, review theme, or handbook section understandable on its own, while still being focused enough for semantic retrieval. The 65-word overlap helps preserve context when a discussion or handbook section crosses a chunk boundary.

**Preprocessing:** `ingest_and_chunk.py` strips HTML tags, removes script/style blocks, unescapes HTML entities, normalizes spacing and smart punctuation, removes common navigation/cookie/share boilerplate, and removes repeated short lines that look like site headers or footers.

**Final chunk count:** 0 chunks. The chunk count is zero because no real documents have been added to `documents/` yet. Running `python ingest_and_chunk.py` currently stops with: `No source documents found in documents. Add .txt, .md, .html, or .pdf files, then run this script again.`

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` through `sentence-transformers`.

This model was chosen because it runs locally, does not require an API key, is fast enough for a small class project, and is commonly used for lightweight semantic search. The embedding script stores chunks in ChromaDB with metadata for source ID, source path, title, chunk index, and word positions.

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

The generation layer is implemented in `query.py`. It retrieves chunks first, formats only those chunks into the LLM prompt, and uses Groq's `llama-3.3-70b-versatile` with `temperature=0`. It also checks retrieval distance before calling the LLM; if the best match is too weak, it returns the fallback answer instead of asking the model to guess.

**How source attribution is surfaced in the response:** Source attribution is appended programmatically from retrieved metadata rather than relying on the LLM to cite sources correctly. The Gradio app displays the answer in one box and a separate "Retrieved from" box listing each retrieved source, file path, chunk number, and distance score.

---

## Evaluation Report

I ran all five planned questions from `planning.md`. Because the vector store is empty, every query failed at retrieval before generation could run. This is a real evaluation result: the interface and generation code exist, but the system is not ready to answer until source documents are added, chunked, embedded, and indexed.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What advice do current students give incoming CCNY CS freshmen? | Register early and stay on top of coursework. | Error: ChromaDB collection is empty. Run `python embed_and_retrieve.py index` first. | Off-target: no retrieval occurred | Inaccurate |
| 2 | What complaints do students have about registration? | Limited seats and scheduling conflicts. | Error: ChromaDB collection is empty. Run `python embed_and_retrieve.py index` first. | Off-target: no retrieval occurred | Inaccurate |
| 3 | What traits do students praise in highly rated professors? | Clear explanations and helpful feedback. | Error: ChromaDB collection is empty. Run `python embed_and_retrieve.py index` first. | Off-target: no retrieval occurred | Inaccurate |
| 4 | What opportunities do CCNY clubs provide? | Leadership, networking, and campus involvement. | Error: ChromaDB collection is empty. Run `python embed_and_retrieve.py index` first. | Off-target: no retrieval occurred | Inaccurate |
| 5 | What challenges do students face balancing classes and clubs? | Time management and heavy workloads. | Error: ChromaDB collection is empty. Run `python embed_and_retrieve.py index` first. | Off-target: no retrieval occurred | Inaccurate |

**Retrieval quality:** Off-target for all five questions because no documents have been indexed.  
**Response accuracy:** Inaccurate for all five questions because the system did not return the expected grounded answers.

---

## Failure Case Analysis

**Question that failed:** What complaints do students have about registration?

**What the system returned:** `ERROR: The ChromaDB collection is empty. Run python embed_and_retrieve.py index first.`

**Root cause (tied to a specific pipeline stage):** The failure happens at the retrieval stage. The ingestion stage has not produced `data/chunks/chunks.jsonl` because there are no source files in `documents/`. Since there are no chunks, the embedding stage cannot populate ChromaDB. When `query.py` calls the retriever, ChromaDB has an empty collection, so no relevant chunks can be returned and generation cannot be grounded.

**What you would change to fix it:** I would collect the ten planned documents into the `documents/` folder, starting with manually saved `.txt` files for Reddit/RMP-style sources and PDFs or text exports for club resources. Then I would run `python ingest_and_chunk.py`, inspect the cleaned document sample and five representative chunks, run `python embed_and_retrieve.py index`, and rerun all five evaluation questions. If top distances are still above about 0.6-0.7 or chunks look noisy, I would adjust cleaning and chunk size before recording the demo.

---

## Spec Reflection

**One way the spec helped you during implementation:** The planning document kept the pipeline concrete. Because it named the embedding model, top-k value, ChromaDB vector store, chunk size range, and overlap range, the implementation could be split cleanly into ingestion/chunking, embedding/retrieval, and grounded generation files. It also forced the code to preserve metadata early, which matters later for source attribution in the interface.

**One way your implementation diverged from the spec, and why:** The spec assumed that the document collection would exist before evaluation, but the workspace did not contain the planned source documents. Because of that, the implementation diverged by adding explicit fail-fast checks instead of silently continuing with an empty or fake corpus. This makes the current evaluation unsuccessful, but it keeps the project honest and prevents the LLM from producing answers that are not grounded in retrieved documents.

---

## AI Usage

**Instance 1**

- *What I gave the AI:* I gave the AI the Milestone 3 instructions, the `planning.md` Documents section, the Chunking Strategy section, and the architecture diagram.
- *What it produced:* It produced `ingest_and_chunk.py`, a local document pipeline that discovers `.txt`, `.md`, `.html`, `.htm`, and `.pdf` files, saves raw text, cleans content, chunks text, prints inspection samples, and writes `chunks.jsonl`.
- *What I changed or overrode:* I made the chunking numbers concrete at 450 words with 65 words overlap because those values sit inside the planned ranges. I also made the script stop when no documents exist instead of generating placeholder data.

**Instance 2**

- *What I gave the AI:* I gave the AI the Retrieval Approach section from `planning.md` and asked it to implement embedding and ChromaDB retrieval using `all-MiniLM-L6-v2` and top-k retrieval.
- *What it produced:* It produced `embed_and_retrieve.py`, which loads chunks, embeds them with `sentence-transformers`, stores them in ChromaDB with source metadata, and prints retrieved chunks with distance scores for test queries.
- *What I changed or overrode:* I adjusted the retrieval function so it checks whether the Chroma collection is empty before loading the embedding model. This avoids wasting time trying to download or load the model when there is no indexed data to search.

**Instance 3**

- *What I gave the AI:* I gave the AI the Milestone 5 grounding requirement, the desired output format of answer plus source list, and the requirement to build a simple Gradio interface.
- *What it produced:* It produced `query.py` for grounded generation through Groq and `app.py` for a Gradio web UI.
- *What I changed or overrode:* I made source attribution programmatic in Python instead of depending on the LLM to cite sources. I also added a weak-retrieval guard so the system says it does not have enough information when retrieval distance is too high.

---

## Demo Notes

To record the 3-5 minute demo after documents are added, run:

```powershell
python ingest_and_chunk.py
python embed_and_retrieve.py index
python app.py
```

Then open `http://127.0.0.1:7860`.

The demo should show three queries with the "Retrieved from" source box visible, one strong answer, one failure or weak retrieval case, and a quick walkthrough of this evaluation report. At the current stage, a demo would only show the empty-index failure, so the source documents need to be collected before recording the final submission video.
