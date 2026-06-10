# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

The domain I chose is student-generated knowledge about The City College of New York (CCNY). This includes information shared by students through Reddit discussions, professor reviews, club resources, and student handbooks.

This knowledge is valuable because many of the questions students have are not fully answered by official college websites. While CCNY provides information about courses, clubs, and academic policies, students often rely on other students to learn about topics such as professor teaching styles, course difficulty, registration strategies, workload expectations, campus life, and extracurricular opportunities.

This information is difficult to find through official channels because it is spread across many different sources, including Reddit threads, Rate My Professors reviews, student guides, and organization documents. A Retrieval-Augmented Generation (RAG) system can bring these sources together and make them searchable through natural language questions, helping students quickly find practical advice and experiences from other students.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| #   | Source                       | Description                                       | URL or location               |
| --- | ---------------------------- | ------------------------------------------------- | ----------------------------- |
| 1   | CCNY Reddit - Campus Life    | Student discussions about classes and campus life | reddit.com/r/CCNY             |
| 2   | CCNY Reddit - CS Advice      | Advice for incoming CS students                   | reddit.com/r/CCNY             |
| 3   | CCNY Reddit - Registration   | Student experiences with registration             | reddit.com/r/CCNY             |
| 4   | CCNY Reddit - Professors     | Opinions and recommendations on professors        | reddit.com/r/CCNY             |
| 5   | CCNY Reddit - Workload       | Discussion of course difficulty and workload      | reddit.com/r/CCNY             |
| 6   | Rate My Professors Reviews   | Student reviews of CCNY professors                | ratemyprofessors.com          |
| 7   | CS Professor Ratings         | Reviews of CS faculty                             | ratemyprofessors.com          |
| 8   | Clubs Directory              | Student organizations and activities              | clubs_directory.txt           |
| 9   | Student Activities Resources | Campus events and involvement opportunities       | student_activities.txt        |
| 10  | 2025-2026 Club Handbook      | Official CCNY club handbook                       | beaver_handbook_2025_2026.pdf |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** 350-500 words per chunk. The Milestone 3 script uses 450 words per chunk by default.

**Overlap:** 50-75 words. The Milestone 3 script uses a 65-word overlap by default.

**Reasoning:**  
My sources include short student discussions, professor reviews, Reddit posts, and longer documents like the CCNY Student Club Handbook. Since many of the sources are review-heavy or discussion-based, smaller chunks help keep each student opinion or topic focused. The overlap helps preserve context when a comment, professor review, or handbook section continues across chunk boundaries. This strategy should make retrieval more accurate because the system can pull specific advice about professors, registration, clubs, or workload instead of retrieving very large sections with unrelated information.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** all-MiniLM-L6-v2 via sentence-transformers

**Top-k:** 5 chunks per query

**Production tradeoff reflection:**  
For this project, I chose all-MiniLM-L6-v2 because it is free, lightweight, fast, and works locally without needing an API. This is a good fit for a small student project with a limited number of documents. If this system were deployed for real CCNY students, I would compare stronger embedding models based on retrieval accuracy, context length, multilingual support, latency, and cost. Since CCNY has many multilingual students, multilingual support could be important. I would also consider whether a larger model retrieves better answers from informal student language, Reddit posts, and professor reviews.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| #   | Question                                                        | Expected answer                                 |
| --- | --------------------------------------------------------------- | ----------------------------------------------- |
| 1   | What advice do current students give incoming CCNY CS freshmen? | Register early and stay on top of coursework.   |
| 2   | What complaints do students have about registration?            | Limited seats and scheduling conflicts.         |
| 3   | What traits do students praise in highly rated professors?      | Clear explanations and helpful feedback.        |
| 4   | What opportunities do CCNY clubs provide?                       | Leadership, networking, and campus involvement. |
| 5   | What challenges do students face balancing classes and clubs?   | Time management and heavy workloads.            |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Student-generated sources such as Reddit posts and professor reviews may contain biased, inconsistent, or conflicting information. Different students may have very different experiences, making it difficult to determine a single correct answer.

2. The retrieval system may return irrelevant chunks if a query contains ambiguous terms or if multiple documents discuss similar topics. This could lead to responses that are only partially grounded in the most relevant sources.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Inestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

```text
+--------------------+     +----------------+     +----------------------+     +----------------+     +----------------+
| Document Ingestion | --> | Chunking       | --> | Embedding + Vector   | --> | Retrieval      | --> | Generation     |
| Python Loaders     |     | Text Splitter  |     | Store (ChromaDB)     |     | Top-k Search   |     | LLM + Sources  |
+--------------------+     +----------------+     +----------------------+     +----------------+     +----------------+
```

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use
     - What you'll give it as input
     - What you expect it to produce
     - How you'll verify the output matches your spec -->

**Milestone 3 — Ingestion and chunking:**

I will use ChatGPT to help implement the document ingestion and chunking pipeline. I will provide my Documents section, Chunking Strategy section, and Architecture diagram from planning.md. I will ask it to generate Python code that loads my CCNY documents, cleans unnecessary text, and creates chunks using my specified chunk size and overlap. I will verify the output by inspecting sample chunks and ensuring they are readable, self-contained, and match my chunking specifications.

**Milestone 4 — Embedding and retrieval:**

I will use ChatGPT to implement the embedding and retrieval components. I will provide my Retrieval Approach section and Architecture diagram. I will ask it to generate code that uses all-MiniLM-L6-v2 to create embeddings, stores them in ChromaDB, and retrieves the top 5 most relevant chunks for a query. I will verify the output by testing evaluation questions and checking whether the retrieved chunks are relevant to the query and come from the correct source documents.

**Milestone 5 — Generation and interface:**

I will use ChatGPT to help build the generation pipeline and user interface. I will provide the project requirements, Retrieval Approach, and grounding requirements. I will ask it to generate code that sends retrieved chunks to an LLM, generates answers using only the retrieved context, and displays answers with source citations through a simple Gradio interface. I will verify the output by testing both in-scope and out-of-scope questions and ensuring answers include source attribution and do not rely on information outside the retrieved documents.
