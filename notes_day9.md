Chunks:
Chunk count is NOT determined by page count.
Chunk count IS determined by text density.

Pages × text_per_page ÷ chunk_size = chunks

A 1-page PDF with 10,000 words
creates more chunks than a 50-page PDF
with mostly images and whitespace.

SmartDocs page (sparse):           CRAG page (dense):
┌─────────────────────┐            ┌─────────────────────┐
│  # Heading          │            │ Lorem ipsum dense   │
│                     │            │ text continues here │
│  [code block]       │            │ with references and │
│                     │            │ citations. Further  │
│  Some explanation   │            │ analysis shows that │
│                     │            │ the model performs  │
│  [diagram]          │            │ better on multiple  │
│                     │            │ benchmarks. Table 1 │
└─────────────────────┘            │ demonstrates this   │
                                   │ clearly. Moreover   │
~500 chars = 1-2 chunks            │ the ablation study  │
                                   └─────────────────────┘
                                   ~4000 chars = 8-10 chunks