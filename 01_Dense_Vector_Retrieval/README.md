<p align = "center" draggable="false" ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

<h1 align="center" id="heading">Session 1: Dense Vector Retrieval</h1>

### [Quicklinks]()

| 📰 Module Sheet                                                                 | ⏺️ Recording | 🖼️ Slides | 👨‍💻 Repo       | 📝 Homework | 📁 Feedback |
| :------------------------------------------------------------------------------- | :----------- | :-------- | :------------ | :---------- | :---------- |
| [Dense Vector Retrieval](../00_Docs/Modules/01_Dense_Vector_Retrieval/README.md) |[Recording!](https://us02web.zoom.us/rec/share/sHWvo0Nd1aI0SEhKecOLEX9kFGVJJAdYfsKiuTmm8t85W48Z2lnjpnzTy8jAd8R5.PwuqibGwAZhvDd8c) <br> passcode: `C62n^@Q!`| [Session 1 Slides](https://canva.link/htfqf8i39yejyhn) | You are here! | [Session 1 Assignment](https://forms.gle/Z9qskfVaAvPjn6gz8) | [Feedback 6/2](https://forms.gle/21a2uoL9DVZPwgJP6) |


## 🏗️ How AIM Does Assignments

> 📅 **Assignments will always be released to students as live class begins.** We will never release assignments early.

Each assignment will have a few of the following categories of exercises:

- ❓ **Questions** - these will be questions that you will be expected to gather the answer to. These can appear as general questions, or questions meant to spark a discussion in your breakout rooms.

- 🏗️ **Activities** - these will be work or coding activities meant to reinforce specific concepts or theory components.

- 🚧 **Advanced Builds (optional)** - Take on a challenge. These builds require you to create something with minimal guidance outside of the documentation.

## Main Assignment

In this assignment, you will build a vector RAG application using LangChain v1, OpenAI embeddings, and Qdrant.

The main notebook is:

```text
01_Cat_Health_Vector_RAG_LangChain_Qdrant.ipynb
```

The notebook uses the bundled cat health guideline PDF in `data/cat_health_guidelines.pdf`.

### Setup

From this folder, install the environment with uv:

```bash
uv sync
```

Then open the notebook in Cursor or VS Code and select the Python/Jupyter environment created by uv.

You will also need an OpenAI API key available when running the notebook.

---

## 🏗️ Activity #1: Embedding Similarity

Run the embedding similarity primer in the notebook.

You will compare embeddings for terms like:

- `king`
- `queen`
- `banana`
- `cat`
- `veterinarian`
- `cat health guidelines`

#### ❓Question #1

Why is cosine similarity useful for dense vector retrieval?

##### ✅ Answer:
 Dense vector retrieval is powered by embedding models which convert text into high-dimensional vectors, and cosine similarity helps us measure how closely related these vectors are in meaning.Cosine similarity rationalizes that the smaller the angle between two vectors, the more similar they are in terms of direction, which correlates with semantic similarity in embedding space.
---

## 🏗️ Activity #2: Build the Vector RAG Pipeline

Run the notebook sections that:

1. Load the PDF into LangChain `Document` objects
2. Split the document into chunks
3. Embed the chunks
4. Store the chunk embeddings in in-memory Qdrant
5. Retrieve relevant chunks with similarity scores
6. Generate an answer grounded in retrieved context

#### ❓Question #2

Why is metadata important for a RAG application?

##### ✅ Answer:
A RAG application without metadata can find similar text; however, a RAG application with metadata can find the more relevant and contextually accurate text and prove it. Embeddings tell you which chunks are semantically close to the user query, but the metadata is what helps you understand why those chunks are relevant and filter them based on additional criteria.

#### ❓Question #3

What tradeoff do we make when choosing chunk size and chunk overlap?

##### ✅ Answer:
The tradeoffs are between retrieval precision, semantic continuity, and cost. 

Chunk size controls how much context lives in each vector:
- larger chunk size carry more context but produce blurrier embeddings with less precision
- smaller chunk size retrieve more precisily but lack the surrounding context needed to answer 

Overlap controls how much each chunk re-includes from its neighbors:
- more overlap prevents ideas from being split across chunk boundaries but results in duplicate content, costing storage and tokens
- less overlap keeps the index lean but can drop sentences that span chunk boundaries


 The combination of the two is what affects how the document's meaning is partitioned across chunks:
- Smaller chunks + low overlap: best for answers that live in single sentences. high precision, low recall, low cost
- Larger chunks + low overlap: low precision, decent recall, low cost. However, vectors are too generic to rank well 
- Small chunks + high overlap: high precision, high recall, high cost. However, it is expensive and noisy as a result of near-duplicates
- Large chunks + high overlap: wasteful 

#### ❓Question #4

What does a similarity score help you understand, and what does it not prove by itself?

##### ✅ Answer:
The similarity score helps us understand how closely the query and the retrieved chunk align semantically. However, it does not prove the retrieved chunk is relevant to the user's intent or that the answer generated from it is correct.∂∂
---

## 🏗️ Activity #3: Vibe Check Retrieval Quality

Run the notebook's vibe check queries and inspect both:

- The retrieved context
- The generated answer

#### ❓Question #5

For the vibe check queries, did the retrieved context seem relevant before generation? Why or why not?

##### ✅ Answer:
The retrieved context was relevant for all on-topic queries and weak for the off-topic one. Questions 1-3 that were on-topic questions on health, the top hits came from the right sections of the PDF so we know the embedder is doing its job of mapping different vocabulary into neary vectors. For question four, the retrieval returned irrelevant chunks——which is intended since the system prompt instructs the assistant to respond in this manner for a situation like this. 

One observation to note is that vector retireval doesn't need very high cosine scores to be useful. Rather, what matters is that the relevant chunks rank above the irrelevant ones so the LLM can answer more relevantly.

---

## 🏗️ Activity #4: Tune Retrieval

Improve retrieval quality by changing one or more of:

- Chunk size
- Chunk overlap
- Retrieval `k`
- Query wording

Document what changed and whether retrieval improved.

##### Settings Changed:

- increase 'k' size from 4 to 6

##### Results:

1. The additional context resulted in a bullet point change in the result. The additional context included information about behavioral changes in senior cats, which was not present in the original 4-context retrieval.
2. The additional context also resulted in a more detailed answer, including information about pain assessment and musculoskeletal examination.
3. The cost of increasing k was a dropped bullet point and a longer prompt. Furthermore, the retrieved context grew from 911 -> 1239 tokens and the full prompt from 1043 -> 1371 tokens. This increase is negligible at the current scale.

---

## Optional Deep Dive: RAG From Scratch

If you want to look underneath the library abstractions, run the optional reference notebook:

```text
02_Cat_Health_Vector_RAG_From_Scratch.ipynb
```

It builds the same retrieval pipeline again with only:

- `pypdf` for extracting text from the PDF
- Python standard-library HTTP requests for calling OpenAI
- Handcrafted document, chunking, embedding, similarity-search, vector-store, and generation primitives

This notebook is a reference walkthrough, not an additional assignment. Its purpose is to make the responsibilities hidden by LangChain, Qdrant, and provider SDKs visible.

---

## Submitting Your Homework

### Main Assignment

Follow these steps to prepare and submit your homework:

1. Pull the latest updates from upstream into the main branch of your AIE9 repo:

```bash
git checkout main
git pull upstream main
git push origin main
```

2. Start Cursor from the `01_Dense_Vector_Retrieval` folder.
3. Complete the notebook.
4. Answer the questions in this `README.md`.
5. Add, commit, and push your modified work to your origin repository.

When submitting your homework, provide the GitHub URL to your AIE9 repo.
